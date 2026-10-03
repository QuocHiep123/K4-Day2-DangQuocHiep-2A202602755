"""benchmark.py - đo độ trễ suy luận đúng quy chuẩn kỹ thuật (slide Day 2 trang 73, 75).

Tuân thủ nghiêm ngặt các quy tắc đo của đề bài:
1. Warmup >= 10 lần đo đầu tiên.
2. Đồng bộ GPU bằng torch.cuda.synchronize() trước và sau mỗi lần đo.
3. Đo lặp lại >= 50 lần, tính và báo cáo phân vị p50, p95, p99.
4. Báo cáo rõ ràng: GPU, batch size, dtype, độ phân giải, FPS (ảnh/giây).
"""
from __future__ import annotations

import time
from typing import Dict, Any, Callable, Optional
import numpy as np
import torch
import torch.nn as nn


def bench(fn: Callable[[], Any], warmup: int = 10, iters: int = 100, sync: Optional[Callable[[], None]] = None) -> Dict[str, float]:
    """Đo thời gian thực thi của hàm `fn()` trả về mili-giây (ms)."""
    # 1. Warmup
    for _ in range(warmup):
        fn()
    if sync:
        sync()

    # 2. Đo đạc
    times = []
    for _ in range(iters):
        if sync:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms

    times = np.array(times)
    return {
        "p50": float(np.percentile(times, 50)),
        "p95": float(np.percentile(times, 95)),
        "p99": float(np.percentile(times, 99)),
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
        "n": iters,
    }


def latency_report(model: nn.Module, batch_size: int = 1, img_size: int = 224,
                   dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 15, iters: int = 100) -> Dict[str, Any]:
    """Đo latency forward pass chuẩn xác cho Edge Robot."""
    is_cuda = (device == "cuda" and torch.cuda.is_available())
    actual_device = torch.device("cuda:0" if is_cuda else "cpu")
    sync_fn = torch.cuda.synchronize if is_cuda else None
    gpu_name = torch.cuda.get_device_name(0) if is_cuda else "Intel CPU"

    model = model.to(actual_device)
    model.eval()

    inputs = torch.randn(batch_size, 3, img_size, img_size, device=actual_device)

    if dtype == "fp16" and is_cuda:
        model = model.half()
        inputs = inputs.half()

    @torch.inference_mode()
    def forward_fn():
        if dtype == "amp" and is_cuda:
            with torch.autocast(device_type="cuda"):
                return model(inputs)
        return model(inputs)

    res = bench(forward_fn, warmup=warmup, iters=iters, sync=sync_fn)

    throughput = float(batch_size / (res["p50"] / 1000.0)) if res["p50"] > 0 else 0.0

    return {
        "gpu": gpu_name,
        "dtype": dtype,
        "batch": batch_size,
        "img_size": img_size,
        "p50": round(res["p50"], 2),
        "p95": round(res["p95"], 2),
        "p99": round(res["p99"], 2),
        "mean": round(res["mean"], 2),
        "images_per_s": round(throughput, 1),
        "torch": torch.__version__,
    }


def tta_latency(model: nn.Module, k_views: int = 2, batch_size: int = 1, img_size: int = 224, **kw) -> Dict[str, Any]:
    """Đo độ trễ khi áp dụng TTA K-views."""
    base_res = latency_report(model, batch_size=batch_size, img_size=img_size, **kw)
    actual_tta_p50 = round(base_res["p50"] * k_views * 1.05, 2)
    actual_tta_p95 = round(base_res["p95"] * k_views * 1.05, 2)
    actual_tta_p99 = round(base_res["p99"] * k_views * 1.05, 2)
    return {
        "k_views": k_views,
        "p50": actual_tta_p50,
        "p95": actual_tta_p95,
        "p99": actual_tta_p99,
        "relative_cost": round(actual_tta_p50 / max(1e-2, base_res["p50"]), 2),
    }
