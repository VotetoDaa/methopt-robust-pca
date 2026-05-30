from __future__ import annotations

import torch

from src.methods.base import BaseMethod, MethodResult


class PCAMethod(BaseMethod):
    name = "pca"

    def fit_transform(self, M: torch.Tensor, rank: int = 5, **kwargs) -> MethodResult:
        if rank <= 0:
            raise ValueError("rank must be positive.")

        max_rank = min(M.shape)
        used_rank = min(rank, max_rank)

        # The video matrix is not centered so the low-rank component preserves background brightness.
        U, singular_values, Vh = torch.linalg.svd(M, full_matrices=False)
        U_r = U[:, :used_rank]
        s_r = singular_values[:used_rank]
        Vh_r = Vh[:used_rank, :]
        L = (U_r * s_r.unsqueeze(0)) @ Vh_r
        S = M - L

        info = {
            "method": self.name,
            "iterations": 1,
            "rank": int(used_rank),
            "converged": True,
            "final_error": None,
            "centered": False,
            "requested_rank": int(rank),
        }
        return MethodResult(L=L, S=S, info=info)
