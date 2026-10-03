"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Cài đặt đầy đủ: Label Smoothing, Focal Loss, Class Weights, Mixup và CutMix.
Kiểm tra tính đúng đắn: Focal Loss với gamma=0 và Label Smoothing với eps=0 tương đương tuyệt đối CE.
"""
from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 9


def build_criterion(kind: str = "ce", **kw):
    """Khởi tạo hàm loss theo cấu hình:
    - 'ce': Standard Cross Entropy
    - 'ls': Label Smoothing Cross Entropy
    - 'focal': Focal Loss
    - 'ce_weighted': Weighted Cross Entropy
    """
    if kind == "ce":
        return nn.CrossEntropyLoss()
    elif kind == "ls":
        smoothing = kw.get("smoothing", kw.get("label_smoothing", 0.1))
        return LabelSmoothingCE(smoothing=smoothing)
    elif kind == "focal":
        gamma = kw.get("gamma", kw.get("focal_gamma", 2.0))
        alpha = kw.get("alpha", kw.get("weight", None))
        return FocalLoss(gamma=gamma, alpha=alpha)
    elif kind == "ce_weighted":
        weight = kw.get("weight", None)
        return nn.CrossEntropyLoss(weight=weight)
    else:
        raise ValueError(f"Không hỗ trợ loss: {kind}")


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với Label Smoothing (slide trang 56)."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets)


class FocalLoss(nn.Module):
    """Multi-class Focal Loss (Lin et al., slide trang 57).
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # CE loss không reduction
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        # p_t là xác suất dự đoán của đúng class mục tiêu
        p_t = torch.exp(-ce_loss)
        focal_loss = ((1.0 - p_t) ** self.gamma) * ce_loss
        return focal_loss.mean()


def class_weights(counts, beta: float = 0.0) -> torch.Tensor:
    """Tính trọng số lớp từ số lượng mẫu tập train.
    - beta = 0: Trọng số tỉ lệ nghịch số mẫu (1 / n_c), chuẩn hóa trung bình về 1.
    - beta > 0: Class-Balanced theo effective number of samples (Cui et al., slide trang 57).
    """
    counts = np.array(counts, dtype=np.float64)
    if beta == 0.0:
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.mean()
    else:
        # w_c = (1 - beta) / (1 - beta^n_c)
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / (effective_num + 1e-8)
        weights = weights / weights.sum() * len(counts)
    return torch.tensor(weights, dtype=torch.float32)


def rand_bbox(size: Tuple[int, int, int, int], lam: float) -> Tuple[int, int, int, int]:
    """Tạo bounding box ngẫu nhiên cho CutMix với tỉ lệ diện tích ~ (1 - lam)."""
    W = size[3]
    H = size[2]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Tâm ngẫu nhiên
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def mix_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn mẫu ảnh và nhãn theo batch (Mixup hoặc CutMix)."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    perm = torch.randperm(batch_size, device=x.device)
    y_a = y
    y_b = y[perm]

    if mode == "mixup":
        x_mixed = lam * x + (1.0 - lam) * x[perm]
        return x_mixed, (y_a, y_b, lam)
    elif mode == "cutmix":
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x_mixed = x.clone()
        x_mixed[:, :, bby1:bby2, bbx1:bbx2] = x[perm, :, bby1:bby2, bbx1:bbx2]
        # Điều chỉnh lam theo diện tích cắt thực tế
        box_area = (bbx2 - bbx1) * (bby2 - bby1)
        total_area = x.size(2) * x.size(3)
        actual_lam = 1.0 - (box_area / float(total_area))
        return x_mixed, (y_a, y_b, actual_lam)
    else:
        raise ValueError(f"Không hỗ trợ mix mode: {mode}")


def mixed_loss(criterion, logits: torch.Tensor, targets: Tuple[torch.Tensor, torch.Tensor, float]) -> torch.Tensor:
    """Tính loss cho batch đã trộn nhãn mềm: lam * L(y_a) + (1 - lam) * L(y_b)."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
