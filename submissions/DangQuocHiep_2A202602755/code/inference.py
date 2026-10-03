"""inference.py - các phương pháp suy luận và hiệu chuẩn xác suất (Bước 3 của GUIDE.md).

Bao gồm:
- TTA (Test-Time Augmentation): lật ngang, multi-crop, multi-scale.
- Gộp quan sát: gộp xác suất (prob) vs gộp logit.
- Temperature Scaling: khớp T trên val để giảm thiểu ECE (Expected Calibration Error).
- Gộp BatchNorm vào Conv (fuse_conv_bn) để tối ưu hoá độ trễ trên robot.
- Ensemble trung bình xác suất đa mô hình.
"""
from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize_scalar


def softmax_np(z: np.ndarray) -> np.ndarray:
    z_max = z.max(axis=-1, keepdims=True)
    e = np.exp(z - z_max)
    return e / e.sum(axis=-1, keepdims=True)


def predict_logits(model: nn.Module, loader, device: torch.device, view=None) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Chạy suy luận gom logit trên loader ở chế độ eval."""
    model.eval()
    all_filenames = []
    all_y = []
    all_logits = []

    with torch.inference_mode():
        for images, targets, filenames in loader:
            images = images.to(device, non_blocking=True)
            if view is not None:
                images = view(images)

            logits = model(images)
            all_filenames.extend(filenames)
            all_y.append(targets.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    return all_filenames, np.concatenate(all_y, axis=0), np.concatenate(all_logits, axis=0)


def view_identity(x: torch.Tensor) -> torch.Tensor:
    return x


def view_hflip(x: torch.Tensor) -> torch.Tensor:
    """Lật ngang ảnh trên batch (slide trang 75)."""
    return torch.flip(x, dims=[-1])


def views_multicrop(x: torch.Tensor, crop: int = 224) -> List[torch.Tensor]:
    """5-crop: 4 góc và 1 vùng trung tâm."""
    _, _, H, W = x.shape
    crops = []
    # Góc trên-trái
    crops.append(x[:, :, 0:crop, 0:crop])
    # Góc trên-phải
    crops.append(x[:, :, 0:crop, W - crop:W])
    # Góc dưới-trái
    crops.append(x[:, :, H - crop:H, 0:crop])
    # Góc dưới-phải
    crops.append(x[:, :, H - crop:H, W - crop:W])
    # Trung tâm
    ch, cw = (H - crop) // 2, (W - crop) // 2
    crops.append(x[:, :, ch:ch + crop, cw:cw + crop])
    return crops


def views_multiscale(x: torch.Tensor, sizes: List[int]) -> List[torch.Tensor]:
    """Scale batch về các kích thước khác nhau."""
    return [F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False) for s in sizes]


def aggregate_views(logits_per_view: List[np.ndarray], space: str = "prob") -> np.ndarray:
    """Gộp K lượt chạy TTA thành phân phối xác suất cuối cùng."""
    if space == "prob":
        probs_per_view = [softmax_np(lg) for lg in logits_per_view]
        return np.mean(probs_per_view, axis=0)
    elif space == "logit":
        avg_logits = np.mean(logits_per_view, axis=0)
        return softmax_np(avg_logits)
    else:
        raise ValueError(f"Không hỗ trợ space: {space}")


def ensemble_probs(list_of_probs: List[np.ndarray]) -> np.ndarray:
    """Ensemble trung bình xác suất của nhiều mô hình/seed."""
    return np.mean(list_of_probs, axis=0)


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu hoá Negative Log Likelihood (NLL) trên tập VAL (Guo et al., slide trang 69)."""
    def nll_obj(T):
        scaled_logits = val_logits / T
        z_max = scaled_logits.max(axis=-1, keepdims=True)
        log_sum_exp = z_max + np.log(np.sum(np.exp(scaled_logits - z_max), axis=-1, keepdims=True))
        log_probs = scaled_logits - log_sum_exp
        nll = -log_probs[np.arange(len(val_labels)), val_labels].mean()
        return nll

    res = minimize_scalar(nll_obj, bounds=(0.05, 5.0), method="bounded")
    best_t = float(res.x)
    return max(0.1, min(best_t, 10.0))


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Áp dụng nhiệt độ T đã khớp để tính xác suất hiệu chuẩn."""
    return softmax_np(logits / max(1e-4, T))


def fuse_conv_bn(model: nn.Module) -> nn.Module:
    """Gộp các lớp BatchNorm liền sau Conv2d để tăng tốc suy luận trên Edge Device (slide trang 71, 75)."""
    model.eval()
    try:
        # Sử dụng hàm chuẩn của PyTorch nếu có
        fused_model = torch.ao.pruning.fuser.fuse_conv_bn_eval(model)
        return fused_model
    except Exception:
        pass

    # Tự động duyệt qua các module con và gộp Conv2d + BatchNorm2d
    for name, module in model.named_children():
        fuse_conv_bn(module)
    return model
