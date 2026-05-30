from __future__ import annotations

from src.methods.base import BaseMethod
from src.methods.godec import GoDecMethod
from src.methods.pca import PCAMethod
from src.methods.randomized_pca import RandomizedPCAMethod
from src.methods.rpca_admm import RPCAADMMMethod
from src.methods.windowed_godec import WindowedGoDecMethod


METHODS = {
    "pca": PCAMethod,
    "randomized_pca": RandomizedPCAMethod,
    "godec": GoDecMethod,
    "windowed_godec": WindowedGoDecMethod,
    "rpca_admm": RPCAADMMMethod,
}


def available_methods() -> list[str]:
    return sorted(METHODS)


def get_method(name: str) -> BaseMethod:
    if name not in METHODS:
        available = ", ".join(available_methods())
        raise ValueError(f"Unknown method '{name}'. Available methods: {available}")
    return METHODS[name]()
