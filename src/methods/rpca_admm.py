from __future__ import annotations

import math

import torch
from tqdm import tqdm

from src.methods.base import BaseMethod, MethodResult


def _soft_threshold(X: torch.Tensor, threshold: float) -> torch.Tensor:
    return torch.sign(X) * torch.clamp(torch.abs(X) - threshold, min=0.0)


def _singular_value_threshold(X: torch.Tensor, threshold: float) -> tuple[torch.Tensor, int]:
    U, singular_values, Vh = torch.linalg.svd(X, full_matrices=False)
    shrunk = torch.clamp(singular_values - threshold, min=0.0)
    keep = shrunk > 0
    rank = int(keep.sum().detach().cpu())
    if rank == 0:
        return torch.zeros_like(X), 0
    U_r = U[:, keep]
    s_r = shrunk[keep]
    Vh_r = Vh[keep, :]
    return (U_r * s_r.unsqueeze(0)) @ Vh_r, rank


class RPCAADMMMethod(BaseMethod):
    name = "rpca_admm"

    def fit_transform(
        self,
        M: torch.Tensor,
        lambda_: float | None = None,
        mu: float | None = None,
        max_iter: int = 200,
        tol: float = 1e-5,
        **kwargs,
    ) -> MethodResult:
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if tol <= 0:
            raise ValueError("tol must be positive.")

        if lambda_ is None:
            lambda_ = 1.0 / math.sqrt(max(M.shape))
        if mu is None:
            spectral_norm = torch.linalg.matrix_norm(M, ord=2).detach().clamp_min(1e-12)
            mu = float(1.25 / spectral_norm.cpu())
        if lambda_ <= 0 or mu <= 0:
            raise ValueError("lambda_ and mu must be positive.")

        L = torch.zeros_like(M)
        S = torch.zeros_like(M)
        Y = torch.zeros_like(M)
        norm_M = torch.linalg.norm(M, ord="fro").clamp_min(1e-12)
        converged = False
        final_error = float("inf")
        rank = 0

        iterator = tqdm(range(1, max_iter + 1), desc=self.name, leave=False)
        for iteration in iterator:
            L, rank = _singular_value_threshold(M - S + Y / mu, threshold=1.0 / mu)
            S = _soft_threshold(M - L + Y / mu, threshold=lambda_ / mu)
            residual = M - L - S
            Y = Y + mu * residual

            final_error = float((torch.linalg.norm(residual, ord="fro") / norm_M).detach().cpu())
            iterator.set_postfix(error=f"{final_error:.2e}", rank=rank)
            if final_error < tol:
                converged = True
                break

        info = {
            "method": self.name,
            "iterations": int(iteration),
            "rank": int(rank),
            "converged": bool(converged),
            "final_error": float(final_error),
            "lambda": float(lambda_),
            "mu": float(mu),
        }
        return MethodResult(L=L, S=S, info=info)
