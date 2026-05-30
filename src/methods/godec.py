from __future__ import annotations

import torch
from tqdm import tqdm

from src.methods.base import BaseMethod, MethodResult


def _low_rank_approximation(M: torch.Tensor, rank: int) -> torch.Tensor:
    U, singular_values, Vh = torch.linalg.svd(M, full_matrices=False)
    r = min(rank, singular_values.numel())
    if r == 0:
        return torch.zeros_like(M)
    return (U[:, :r] * singular_values[:r].unsqueeze(0)) @ Vh[:r, :]


def _quantile_threshold(values: torch.Tensor, quantile: float) -> torch.Tensor:
    flat = values.reshape(-1)
    q = min(max(float(quantile), 0.0), 1.0)

    if flat.numel() == 0:
        return torch.tensor(0.0, device=values.device, dtype=values.dtype)

    if q <= 0.0:
        return flat.min()

    if q >= 1.0:
        return flat.max()

    k = int(q * (flat.numel() - 1)) + 1
    k = max(1, min(k, flat.numel()))

    return torch.kthvalue(flat, k).values


class GoDecMethod(BaseMethod):
    name = "godec"

    def fit_transform(
        self,
        M: torch.Tensor,
        rank: int = 5,
        threshold_quantile: float = 0.95,
        max_iter: int = 100,
        tol: float = 1e-5,
        **kwargs,
    ) -> MethodResult:
        if rank <= 0:
            raise ValueError("rank must be positive.")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if tol <= 0:
            raise ValueError("tol must be positive.")

        L = _low_rank_approximation(M, rank)
        S = torch.zeros_like(M)

        norm_M = torch.linalg.norm(M, ord="fro").clamp_min(1e-12)
        previous_error = float("inf")
        final_error = float("inf")
        converged = False

        iterator = tqdm(range(1, max_iter + 1), desc=self.name, leave=False)

        for iteration in iterator:
            residual = M - L

            abs_residual = torch.abs(residual)
            threshold = _quantile_threshold(abs_residual, threshold_quantile)
            S = torch.where(abs_residual >= threshold, residual, torch.zeros_like(residual))

            L = _low_rank_approximation(M - S, rank)

            error = torch.linalg.norm(M - L - S, ord="fro") / norm_M
            final_error = float(error.detach().cpu())

            iterator.set_postfix(error=f"{final_error:.2e}", rank=rank)

            if abs(previous_error - final_error) < tol:
                converged = True
                break

            previous_error = final_error

        info = {
            "method": self.name,
            "iterations": int(iteration),
            "rank": int(rank),
            "converged": bool(converged),
            "final_error": float(final_error),
            "threshold_quantile": float(threshold_quantile),
        }

        return MethodResult(L=L, S=S, info=info)