from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data import matrix_column_to_frame
from src.metrics import foreground_mask
from src.utils import ensure_dir


def _to_numpy_frame(vector, height: int, width: int) -> np.ndarray:
    frame = matrix_column_to_frame(vector, height, width)
    return np.asarray(frame)


def _normalize_for_display(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.float32)
    min_value = float(np.min(frame))
    max_value = float(np.max(frame))
    if max_value - min_value < 1e-12:
        return np.zeros_like(frame)
    return (frame - min_value) / (max_value - min_value)


def save_decomposition_grid(
    M,
    L,
    S,
    metadata,
    output_path,
    frame_indices: list[int],
) -> None:
    height = int(metadata["height"])
    width = int(metadata["width"])
    frame_count = int(metadata["frames"])
    valid_indices = [idx for idx in frame_indices if 0 <= idx < frame_count]
    if not valid_indices:
        valid_indices = [0]

    mask = foreground_mask(S, threshold_strategy="quantile", quantile=0.985)
    columns = ["Original", "Low-rank background", "Sparse foreground", "Binary mask"]

    fig, axes = plt.subplots(len(valid_indices), len(columns), figsize=(12, 3 * len(valid_indices)))
    axes_array = np.asarray(axes).reshape(len(valid_indices), len(columns))

    for row, frame_idx in enumerate(valid_indices):
        original = _to_numpy_frame(M[:, frame_idx], height, width)
        background = _to_numpy_frame(L[:, frame_idx], height, width)
        sparse = np.abs(_to_numpy_frame(S[:, frame_idx], height, width))
        binary = _to_numpy_frame(mask[:, frame_idx].to(torch.float32), height, width)
        images = [original, background, _normalize_for_display(sparse), binary]

        for col, (title, image) in enumerate(zip(columns, images, strict=True)):
            ax = axes_array[row, col]
            ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_title(f"{title}\nframe {frame_idx}")
            ax.axis("off")

    fig.tight_layout()
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _read_experiment_csv(csv_path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV is empty: {path}")
    return df


def _line_plot_by_method(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    output_path,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped = df.groupby(["method", x_col], as_index=False)[y_col].mean()
    for method, part in grouped.groupby("method"):
        part = part.sort_values(x_col)
        ax.plot(part[x_col], part[y_col], marker="o", label=method)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_runtime_vs_frames(csv_path, output_path) -> None:
    df = _read_experiment_csv(csv_path)
    _line_plot_by_method(
        df,
        x_col="frames",
        y_col="runtime_sec",
        output_path=output_path,
        xlabel="Frames",
        ylabel="Runtime, seconds",
        title="Runtime vs Frames",
    )


def plot_memory_vs_frames(csv_path, output_path) -> None:
    df = _read_experiment_csv(csv_path)
    _line_plot_by_method(
        df,
        x_col="frames",
        y_col="peak_memory_mb",
        output_path=output_path,
        xlabel="Frames",
        ylabel="Peak memory, MB",
        title="Memory vs Frames",
    )


def plot_runtime_vs_resolution(csv_path, output_path) -> None:
    df = _read_experiment_csv(csv_path).copy()
    df["resolution_pixels"] = df["height"] * df["width"]
    _line_plot_by_method(
        df,
        x_col="resolution_pixels",
        y_col="runtime_sec",
        output_path=output_path,
        xlabel="Resolution, pixels",
        ylabel="Runtime, seconds",
        title="Runtime vs Resolution",
    )


def plot_quality_vs_runtime(csv_path, output_path) -> None:
    df = _read_experiment_csv(csv_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, part in df.groupby("method"):
        ax.scatter(part["runtime_sec"], part["relative_error"], label=method, alpha=0.75)
    ax.set_xlabel("Runtime, seconds")
    ax.set_ylabel("Relative reconstruction error")
    ax.set_title("Quality vs Runtime")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
