from __future__ import annotations

import torch

from src.methods.base import BaseMethod, MethodResult


class RandomizedPCAMethod(BaseMethod):
    name = "randomized_pca"

    def fit_transform(
        self,
        M: torch.Tensor,
        rank: int = 5,
        oversampling: int = 10,
        n_iter: int = 2,
        **kwargs,
    ) -> MethodResult:
        if rank <= 0:
            raise ValueError("rank must be positive.")
        if oversampling < 0:
            raise ValueError("oversampling must be non-negative.")
        if n_iter < 0:
            raise ValueError("n_iter must be non-negative.")

        m, n = M.shape
        target_rank = min(rank + oversampling, m, n)

        omega = torch.randn(
            n,
            target_rank,
            device=M.device,
            dtype=M.dtype,
        )

        Y = M @ omega

        for _ in range(n_iter):
            Y = M @ (M.T @ Y)

        Q, _ = torch.linalg.qr(Y, mode="reduced")
        B = Q.T @ M

        U_hat, singular_values, Vh = torch.linalg.svd(B, full_matrices=False)

        r = min(rank, singular_values.numel())
        U = Q @ U_hat[:, :r]
        L = (U * singular_values[:r].unsqueeze(0)) @ Vh[:r, :]
        S = M - L

        info = {
            "method": self.name,
            "iterations": int(n_iter),
            "rank": int(r),
            "converged": True,
            "final_error": None,
            "oversampling": int(oversampling),
        }

        return MethodResult(L=L, S=S, info=info)
