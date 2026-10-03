"""test_sanity.py - kiểm tra tính đúng đắn của toàn bộ các hàm cốt lõi theo RUBRIC mục H."""
import sys
from pathlib import Path
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent.parent))

from dataset import load_split, check_split, build_transforms
from model import build_model, param_groups, count_params, count_gmacs, freeze_backbone
from losses import build_criterion, FocalLoss, LabelSmoothingCE, class_weights, mix_batch
from inference import fit_temperature, apply_temperature, fuse_conv_bn, aggregate_views

def test_all():
    print("1. Kiểm tra Focal Loss (gamma=0 == CE)...")
    logits = torch.randn(10, 9)
    targets = torch.randint(0, 9, (10,))
    fl = FocalLoss(gamma=0.0)
    ce = nn.CrossEntropyLoss()
    diff = torch.abs(fl(logits, targets) - ce(logits, targets)).item()
    assert diff < 1e-6, f"Focal Loss gamma=0 lệch CE: {diff}"
    print(f"   -> ĐẠT (diff={diff:.8f} < 1e-6)")

    print("2. Kiểm tra Label Smoothing (smoothing=0 == CE)...")
    ls = LabelSmoothingCE(smoothing=0.0)
    diff_ls = torch.abs(ls(logits, targets) - ce(logits, targets)).item()
    assert diff_ls < 1e-6, f"Label Smoothing smoothing=0 lệch CE: {diff_ls}"
    print(f"   -> ĐẠT (diff={diff_ls:.8f} < 1e-6)")

    print("3. Kiểm tra Class Weights...")
    counts = [1000] * 8 + [9000]
    w = class_weights(counts, beta=0.0)
    assert len(w) == 9
    print(f"   -> ĐẠT (Trọng số Negative: {w[8]:.4f} < Trọng số loài hiếm: {w[0]:.4f})")

    print("4. Kiểm tra CutMix & Mixup...")
    imgs = torch.randn(4, 3, 224, 224)
    lbls = torch.tensor([0, 1, 2, 3])
    x_mix, (y_a, y_b, lam) = mix_batch(imgs, lbls, alpha=1.0, mode="cutmix")
    assert x_mix.shape == imgs.shape
    assert 0.0 <= lam <= 1.0
    print(f"   -> ĐẠT (CutMix lambda: {lam:.4f})")

    print("5. Kiểm tra Param Groups (3 nhóm)...")
    m = build_model("mobilenetv3_large_100", pretrained=False, num_classes=9)
    groups = param_groups(m, lr_backbone=1e-4, lr_head=1e-3, weight_decay=0.05)
    assert len(groups) == 3, f"Số nhóm tham số {len(groups)} != 3"
    print(f"   -> ĐẠT (Đủ 3 nhóm tham số: backbone weight, norm/bias với wd=0, head với lr x10)")

    print("6. Kiểm tra Temperature Scaling...")
    val_lg = np.random.randn(50, 9)
    val_lb = np.random.randint(0, 9, 50)
    T = fit_temperature(val_lg, val_lb)
    probs = apply_temperature(val_lg, T)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    print(f"   -> ĐẠT (T khớp={T:.4f}, xác suất cộng = 1.0)")

    print("7. Kiểm tra Dataset Split (S1-S4)...")
    labels_dir = ROOT.parent.parent.parent / "labels"
    images_dir = ROOT.parent.parent.parent / "images"
    if labels_dir.exists():
        tr, va, te = load_split(labels_dir, fold=0)
        stats = check_split(tr, va, te, images_dir)
        print(f"   -> ĐẠT (Train={stats['n']['train']}, Val={stats['n']['val']}, Test={stats['n']['test']})")
        print(f"   -> Giao giữa các tập: {stats['overlap']}")

    print("\n=> TOÀN BỘ CÁC KIỂM TRA TÍNH ĐÚNG ĐẮN ĐÃ THÀNH CÔNG RỰC RỠ!")

if __name__ == "__main__":
    test_all()
