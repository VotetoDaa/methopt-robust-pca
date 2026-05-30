from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MethodResult:
    L: torch.Tensor
    S: torch.Tensor
    info: dict


class BaseMethod:
    name: str = "base"

    def fit_transform(self, M: torch.Tensor, **kwargs) -> MethodResult:
        raise NotImplementedError
