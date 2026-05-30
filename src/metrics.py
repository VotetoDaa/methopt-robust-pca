from __future__ import annotations

import torch


def reconstruction_error(M: torch.Tensor, L: torch.Tensor, S: torch.Tensor) -> float:
    return float(torch.linalg.norm(M - L - S, ord="fro").detach().cpu())


def relative_frobenius_error(M: torch.Tensor, L: torch.Tensor, S: torch.Tensor) -> float:
    denom = torch.linalg.norm(M, ord="fro").clamp_min(1e-12)
    return float((torch.linalg.norm(M - L - S, ord="fro") / denom).detach().cpu())


def sparsity(S: torch.Tensor, eps: float = 1e-6) -> float:
    return float((torch.abs(S) <= eps).to(torch.float32).mean().detach().cpu())


def effective_rank(L: torch.Tensor, threshold: float = 1e-6) -> int:
    if L.numel() == 0:
        return 0
    singular_values = torch.linalg.svdvals(L)
    if singular_values.numel() == 0:
        return 0
    cutoff = threshold * singular_values.max().clamp_min(1e-12)
    return int((singular_values > cutoff).sum().detach().cpu())


def foreground_mask(
    S: torch.Tensor,
    threshold_strategy: str = "quantile",
    quantile: float = 0.985,
) -> torch.Tensor:
    values = torch.abs(S)

    if values.ndim != 2:
        raise ValueError("S must be a 2D matrix with shape [pixels, frames].")

    if threshold_strategy == "quantile":
        q = min(max(float(quantile), 0.0), 1.0)

        # Threshold each frame separately.
        # Global threshold often makes some frames empty and others too noisy.
        thresholds = torch.quantile(values, q, dim=0, keepdim=True)
        mask = values >= thresholds

    elif threshold_strategy == "mean_std":
        means = values.mean(dim=0, keepdim=True)
        stds = values.std(dim=0, keepdim=True)
        thresholds = means + 2.0 * stds
        mask = values >= thresholds

    else:
        raise ValueError("threshold_strategy must be 'quantile' or 'mean_std'.")

    # Avoid marking fully zero columns as foreground.
    column_max = values.max(dim=0, keepdim=True).values
    mask = torch.logical_and(mask, column_max > 1e-12)

    return mask


def precision_recall_f1_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> dict[str, float]:
    pred = pred_mask.to(torch.bool)
    gt = gt_mask.to(torch.bool)
    if pred.shape != gt.shape:
        raise ValueError("pred_mask and gt_mask must have the same shape.")

    tp = torch.logical_and(pred, gt).sum().to(torch.float32)
    fp = torch.logical_and(pred, ~gt).sum().to(torch.float32)
    fn = torch.logical_and(~pred, gt).sum().to(torch.float32)

    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    iou = tp / (tp + fp + fn).clamp_min(1.0)

    return {
        "precision": float(precision.detach().cpu()),
        "recall": float(recall.detach().cpu()),
        "f1": float(f1.detach().cpu()),
        "iou": float(iou.detach().cpu()),
    }
