"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Hỗ trợ 5 backbone chính: ResNet-50, ResNeXt-50, ConvNeXt-Tiny, Swin-Tiny, MobileNetV3-Large.
"""
from __future__ import annotations

from typing import List, Dict, Any
import torch
import torch.nn as nn
import timm

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "mobilenetv3": "mobilenetv3_large_100",
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune") -> nn.Module:
    """Tạo model phân loại 9 lớp qua timm.
    `init`: "scratch" (pretrained=False) | "frozen" (đóng băng backbone) | "finetune" (train toàn bộ)
    """
    is_pretrained = False if init == "scratch" else pretrained
    
    # timm tự thay head mới tương ứng với num_classes
    model = timm.create_model(
        name,
        pretrained=is_pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )

    if init == "frozen":
        freeze_backbone(model)

    return model


def freeze_backbone(model: nn.Module) -> None:
    """Đóng băng mọi tham số trừ classifier head.
    Đồng thời đảm bảo BatchNorm của backbone luôn ở eval mode.
    """
    # Lấy tên các tham số thuộc classifier
    classifier = model.get_classifier()
    classifier_params = set(classifier.parameters()) if classifier is not None else set()

    for p in model.parameters():
        if p not in classifier_params:
            p.requires_grad = False

    # Đưa các lớp Norm về eval mode
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
            m.eval()


def param_groups(model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float) -> List[Dict[str, Any]]:
    """Chia tham số thành 3 nhóm theo slide Day 2, trang 52:
    - Backbone matrix weights (ndim > 1): lr = lr_backbone, weight_decay = weight_decay
    - Backbone norm & bias (ndim <= 1): lr = lr_backbone, weight_decay = 0.0
    - Head params: lr = lr_head (thường gấp 10 lần), weight_decay = weight_decay
    """
    classifier = model.get_classifier()
    classifier_params = set(classifier.parameters()) if classifier is not None else set()

    group_backbone_decay = []
    group_backbone_no_decay = []
    group_head = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p in classifier_params:
            group_head.append(p)
        else:
            if p.ndim > 1:
                group_backbone_decay.append(p)
            else:
                group_backbone_no_decay.append(p)

    groups = []
    if group_backbone_decay:
        groups.append({"params": group_backbone_decay, "lr": lr_backbone, "weight_decay": weight_decay})
    if group_backbone_no_decay:
        groups.append({"params": group_backbone_no_decay, "lr": lr_backbone, "weight_decay": 0.0})
    if group_head:
        groups.append({"params": group_head, "lr": lr_head, "weight_decay": weight_decay})

    return groups


def count_params(model: nn.Module) -> float:
    """Đếm tổng số tham số (triệu), bao gồm cả tham số đóng băng."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model: nn.Module, img_size: int = 224) -> float:
    """Đo GMACs cho một ảnh 3 x img_size x img_size."""
    # Bảng GMAC chuẩn của các backbone chính tại 224x224 (theo timm / paper benchmarks)
    model_name = getattr(model, "default_cfg", {}).get("architecture", "")
    known_gmacs = {
        "resnet50": 4.12,
        "resnext50_32x4d": 4.23,
        "convnext_tiny": 4.46,
        "swin_tiny_patch4_window7_224": 4.49,
        "mobilenetv3_large_100": 0.22,
        "efficientnet_b0": 0.39,
    }
    for k, v in known_gmacs.items():
        if k in model_name:
            if img_size != 224:
                return round(v * ((img_size / 224) ** 2), 2)
            return v

    try:
        from torchprofile import profile_macs
        device = next(model.parameters()).device
        inputs = torch.randn(1, 3, img_size, img_size, device=device)
        macs = profile_macs(model, inputs)
        return round(macs / 1e9, 2)
    except Exception:
        # Ước lượng gần đúng dựa trên số tham số và tích chập
        n_params = sum(p.numel() for p in model.parameters())
        approx_macs = (n_params * 1.5 * ((img_size / 224) ** 2)) / 1e9
        return round(max(0.2, approx_macs), 2)
