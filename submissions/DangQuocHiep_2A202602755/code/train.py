"""train.py - vòng huấn luyện thống nhất cho toàn bộ thí nghiệm (B, T, F).

Tất cả thí nghiệm đi qua MỘT hàm `run(cfg: Config)` (RUBRIC mục H).
Tự động lưu checkpoint tốt nhất theo Macro-F1 val, xuất log, vẽ biểu đồ vào curves/,
và xuất file dự đoán chuẩn cho eval.py.
"""
from __future__ import annotations

import os
import sys
import copy
import json
import random
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

# Thêm đường dẫn tới eval.py của repo gốc
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import eval as ev
from dataset import load_split, build_transforms, make_loader, NUM_CLASSES
from model import build_model, param_groups
from losses import build_criterion, class_weights, mix_batch, mixed_loss


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug
    sampler: Optional[str] = None     # None | balanced
    mix: Optional[str] = None         # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: Optional[float] = None
    # --- tối ưu ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: Optional[float] = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    curves_dir: str = "curves"
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    groups = param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config, steps_per_epoch: int):
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(1e-4, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EMA:
    """Exponential Moving Average của trọng số mô hình."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)
                else:
                    self.shadow[k].copy_(v)

    def apply_to(self, model: nn.Module):
        model.load_state_dict(self.shadow)


def train_one_epoch(model: nn.Module, loader, criterion, optimizer, scheduler, scaler: Optional[GradScaler],
                    cfg: Config, device: torch.device, ema: Optional[EMA] = None) -> Dict[str, float]:
    model.train()
    if cfg.init == "frozen":
        from model import freeze_backbone
        freeze_backbone(model)

    running_loss = 0.0
    total_samples = 0

    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = images.size(0)

        optimizer.zero_grad(set_to_none=True)

        use_mix = (cfg.mix is not None) and (np.random.rand() > 0.3)
        if use_mix:
            images, mixed_targets = mix_batch(images, targets, alpha=cfg.mix_alpha, mode=cfg.mix)

        if cfg.amp and device.type == "cuda":
            with autocast():
                logits = model(images)
                if use_mix:
                    loss = mixed_loss(criterion, logits, mixed_targets)
                else:
                    loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            if use_mix:
                loss = mixed_loss(criterion, logits, mixed_targets)
            else:
                loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        scheduler.step()
        if ema is not None:
            ema.update(model)

        running_loss += loss.item() * batch_size
        total_samples += batch_size

    epoch_loss = running_loss / max(1, total_samples)
    current_lr = optimizer.param_groups[0]["lr"]
    return {"train_loss": epoch_loss, "lr": current_lr}


def evaluate(model: nn.Module, loader, criterion, device: torch.device) -> Tuple[List[str], np.ndarray, np.ndarray, float]:
    """Chạy đánh giá trả về filenames, y_true, logits, loss."""
    model.eval()
    all_filenames = []
    all_targets = []
    all_logits = []
    total_loss = 0.0
    total_samples = 0

    with torch.inference_mode():
        for images, targets, filenames in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_size = images.size(0)

            logits = model(images)
            loss = criterion(logits, targets)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_filenames.extend(filenames)
            all_targets.append(targets.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    avg_loss = total_loss / max(1, total_samples)

    return all_filenames, y_true, logits, avg_loss


def plot_curves(history: List[Dict[str, Any]], path: str | Path, title: str) -> None:
    """Vẽ biểu đồ Loss train/val và Macro-F1 val theo epoch."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_macro_f1 = [h["val_macro_f1"] for h in history]
    val_top1 = [h["val_top1"] for h in history]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.plot(epochs, train_loss, "b-o", label="Train Loss", linewidth=1.8, markersize=4)
    ax1.plot(epochs, val_loss, "r--s", label="Val Loss", linewidth=1.8, markersize=4)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11, color="black")
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_macro_f1, "g-^", label="Val Macro-F1", linewidth=2.0, markersize=5)
    ax2.plot(epochs, val_top1, "m--d", label="Val Top-1 Acc", linewidth=1.5, markersize=4, alpha=0.7)
    ax2.set_ylabel("Metric Score", fontsize=11, color="green")
    ax2.set_ylim([min(val_macro_f1) - 0.05, 1.0])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", framealpha=0.9)

    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def run(cfg: Config) -> Dict[str, Any]:
    """Hàm chạy thực nghiệm thống nhất."""
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"BẮT ĐẦU EXPERIMENT: {cfg.exp_id} (Seed {cfg.seed})")
    print(f"Backbone: {cfg.backbone} | Init: {cfg.init} | Loss: {cfg.loss} | Mix: {cfg.mix}")
    print(f"Thiết bị: {device}")
    print(f"=======================================================")

    # 1. Load split fold
    train_df, val_df, test_df = load_split(cfg.labels_dir, fold=cfg.fold)

    # 2. Transforms & Loaders
    train_tf = build_transforms(train=True, img_size=cfg.img_size, aug=cfg.aug)
    val_tf = build_transforms(train=False, img_size=cfg.img_size)

    train_loader = make_loader(train_df, cfg.images_dir, train_tf, cfg.batch_size, train=True,
                               sampler=cfg.sampler, num_workers=cfg.num_workers)
    val_loader = make_loader(val_df, cfg.images_dir, val_tf, cfg.batch_size, train=False,
                             sampler=None, num_workers=cfg.num_workers)

    # 3. Model
    model = build_model(
        name=cfg.backbone,
        pretrained=True,
        num_classes=NUM_CLASSES,
        drop_rate=cfg.drop_rate,
        init=cfg.init,
    ).to(device)

    # 4. Criterion
    criterion_kwargs = {}
    if cfg.loss == "ls":
        criterion_kwargs["smoothing"] = cfg.label_smoothing
    elif cfg.loss == "focal":
        criterion_kwargs["gamma"] = cfg.focal_gamma
    elif cfg.loss == "ce_weighted":
        counts = train_df["Label"].value_counts().sort_index().values
        w = class_weights(counts, beta=cfg.class_weight_beta or 0.0).to(device)
        criterion_kwargs["weight"] = w

    criterion = build_criterion(cfg.loss, **criterion_kwargs)
    eval_criterion = nn.CrossEntropyLoss()

    # 5. Optimizer, Scheduler, EMA
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    scaler = GradScaler() if (cfg.amp and device.type == "cuda") else None
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None

    history = []
    best_macro_f1 = -1.0
    best_model_weights = None
    best_val_preds = None

    # 6. Training Loop
    for epoch in range(1, cfg.epochs + 1):
        train_res = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        
        # Đánh giá bằng EMA weights nếu có
        eval_model = model
        if ema:
            eval_model = copy.deepcopy(model)
            ema.apply_to(eval_model)

        val_fns, val_y, val_logits, val_loss = evaluate(eval_model, val_loader, eval_criterion, device)
        val_probs = softmax_np(val_logits)
        val_preds_class = val_probs.argmax(axis=-1)
        val_cm = ev.confusion_matrix(val_y, val_preds_class)
        val_metrics = ev.compute_metrics(val_cm)

        val_macro_f1 = val_metrics["macro_f1"]
        val_top1 = val_metrics["top1"]

        log_item = {
            "epoch": epoch,
            "train_loss": train_res["train_loss"],
            "val_loss": val_loss,
            "val_macro_f1": val_macro_f1,
            "val_top1": val_top1,
            "lr": train_res["lr"],
        }
        history.append(log_item)
        print(f"Epoch {epoch:02d}/{cfg.epochs:02d} | Train Loss: {train_res['train_loss']:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Top-1: {val_top1*100:.2f}% | Val Macro-F1: {val_macro_f1:.4f}")

        # Chọn checkpoint theo Macro-F1 val cao nhất (README mục 2.2)
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_model_weights = copy.deepcopy(eval_model.state_dict())
            best_val_preds = (val_fns, val_y, val_probs)

    # 7. Lưu đồ thị training curves
    curve_path = Path(cfg.curves_dir) / f"{cfg.exp_id}_{cfg.backbone}.png"
    plot_curves(history, curve_path, title=f"Training Curve: {cfg.exp_id} ({cfg.backbone})")

    # 8. Lưu Val Predictions
    if best_val_preds is not None:
        val_path = pred_path(cfg, "val")
        ev.save_predictions(val_path, best_val_preds[0], best_val_preds[1], best_val_preds[2])

    # 9. Test Evaluation (CHỈ CHẠY DUY NHẤT Ở BƯỚC 4 KHI SAVE_TEST_PREDICTIONS = TRUE)
    test_metrics = None
    if cfg.save_test_predictions:
        print(f"\n[QUY TẮC S4] Chạy đánh giá Test cho {cfg.exp_id} (Seed {cfg.seed})...")
        best_model = build_model(cfg.backbone, pretrained=False, num_classes=NUM_CLASSES).to(device)
        best_model.load_state_dict(best_model_weights)

        test_loader = make_loader(test_df, cfg.images_dir, val_tf, cfg.batch_size, train=False,
                                  sampler=None, num_workers=cfg.num_workers)
        test_fns, test_y, test_logits, _ = evaluate(best_model, test_loader, eval_criterion, device)
        test_probs = softmax_np(test_logits)

        test_path = pred_path(cfg, "test")
        ev.save_predictions(test_path, test_fns, test_y, test_probs)

        test_cm = ev.confusion_matrix(test_y, test_probs.argmax(axis=-1))
        test_metrics = ev.compute_metrics(test_cm)
        print(f"-> Test Top-1: {test_metrics['top1']*100:.2f}% | Test Macro-F1: {test_metrics['macro_f1']:.4f}")

    return {
        "exp_id": cfg.exp_id,
        "seed": cfg.seed,
        "best_val_macro_f1": best_macro_f1,
        "history": history,
        "test_metrics": test_metrics,
    }


def parse_overrides(args_list: List[str]) -> Dict[str, Any]:
    """Parse key=value CLI overrides."""
    overrides = {}
    for item in args_list:
        if "=" in item:
            k, v = item.split("=", 1)
            # Ép kiểu tự động
            if v.lower() == "true":
                v = True
            elif v.lower() == "false":
                v = False
            elif v.lower() == "none":
                v = None
            else:
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
            overrides[k] = v
    return overrides


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình DeepWeeds")
    parser.add_argument("--set", nargs="+", help="Ghi đè tham số Config: key=value", default=[])
    args = parser.parse_args()

    cfg_dict = asdict(Config())
    if args.set:
        overrides = parse_overrides(args.set)
        cfg_dict.update(overrides)

    cfg = Config(**cfg_dict)
    run(cfg)


if __name__ == "__main__":
    main()
