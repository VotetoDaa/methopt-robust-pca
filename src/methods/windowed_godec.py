from __future__ import annotations

import torch
from tqdm import tqdm

from src.methods.base import BaseMethod, MethodResult
from src.methods.godec import GoDecMethod


class WindowedGoDecMethod(BaseMethod):
    name = "windowed_godec"

    def fit_transform(
        self,
        M: torch.Tensor,
        rank: int = 5,
        window_size: int = 100,
        threshold_quantile: float = 0.95,
        max_iter: int = 100,
        tol: float = 1e-5,
        **kwargs,
    ) -> MethodResult:
        if rank <= 0:
            raise ValueError("rank must be positive.")
        if window_size <= 0:
            raise ValueError("window_size must be positive.")

        _, n_frames = M.shape

        L_parts = []
        S_parts = []
        total_iterations = 0
        all_converged = True
        ranks = []

        base_method = GoDecMethod()

        starts = list(range(0, n_frames, window_size))
        iterator = tqdm(starts, desc=self.name, leave=False)

        for start in iterator:
            end = min(start + window_size, n_frames)
            block = M[:, start:end]

            result = base_method.fit_transform(
                block,
                rank=rank,
                threshold_quantile=threshold_quantile,
                max_iter=max_iter,
                tol=tol,
            )

            L_parts.append(result.L)
            S_parts.append(result.S)

            total_iterations += int(result.info.get("iterations") or 0)
            all_converged = all_converged and bool(result.info.get("converged"))
            ranks.append(int(result.info.get("rank") or rank))

        L = torch.cat(L_parts, dim=1)
        S = torch.cat(S_parts, dim=1)

        residual = M - L - S
        norm_M = torch.linalg.norm(M, ord="fro").clamp_min(1e-12)
        final_error = float((torch.linalg.norm(residual, ord="fro") / norm_M).detach().cpu())

        info = {
            "method": self.name,
            "iterations": int(total_iterations),
            "rank": int(max(ranks) if ranks else rank),
            "converged": bool(all_converged),
            "final_error": float(final_error),
            "window_size": int(window_size),
            "threshold_quantile": float(threshold_quantile),
        }

        return MethodResult(L=L, S=S, info=info)
