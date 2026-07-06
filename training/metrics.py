from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def crop_or_resize_to_match(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    _, _, target_h, target_w = target.shape
    _, _, pred_h, pred_w = prediction.shape
    if pred_h == target_h and pred_w == target_w:
        return prediction
    return F.interpolate(
        prediction,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )


def psnr(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-10) -> float:
    prediction = crop_or_resize_to_match(prediction, target).clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    mse = (prediction - target).square().flatten(1).mean(dim=1)
    scores = 10.0 * torch.log10(1.0 / mse.clamp_min(eps))
    return float(scores.mean().item())


def tensor_vram_mb(device: torch.device | str = "cuda") -> float:
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1024**2
