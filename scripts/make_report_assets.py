from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import ensure_dir


METHOD_ORDER = ["pca", "randomized_pca", "godec", "windowed_godec", "rpca_admm"]


def parse_target_frames(run_id: str, actual_frames: int) -> int:
    match = re.search(r"_f(\d+)$", str(run_id))
    if match:
        return int(match.group(1))
    return int(actual_frames)


def method_order_key(method: str) -> int:
    try:
        return METHOD_ORDER.index(method)
    except ValueError:
        return len(METHOD_ORDER)


def line_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    output_path: Path,
    xlabel: str,
    ylabel: str,
    title: str,
    log_y: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    grouped = df.groupby(["method", x_col], as_index=False)[y_col].mean()
    methods = sorted(grouped["method"].unique(), key=method_order_key)

    for method in methods:
        part = grouped[grouped["method"] == method].sort_values(x_col)
        ax.plot(part[x_col], part[y_col], marker="o", label=method)

    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def scatter_plot_quality_runtime(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = sorted(df["method"].unique(), key=method_order_key)

    for method in methods:
        part = df[df["method"] == method]
        ax.scatter(
            part["runtime_sec"],
            part["relative_error"],
            label=method,
            alpha=0.75,
        )

    ax.set_xlabel("Runtime, seconds")
    ax.set_ylabel("Relative decomposition residual")
    ax.set_title("Decomposition Residual vs Runtime")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def bar_plot_mean_runtime(summary: pd.DataFrame, output_path: Path) -> None:
    summary = summary.copy()
    summary["order"] = summary["method"].map(lambda x: method_order_key(str(x)))
    summary = summary.sort_values("order")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(summary["method"], summary["mean_runtime_sec"])
    ax.set_yscale("log")
    ax.set_xlabel("Method")
    ax.set_ylabel("Mean runtime, seconds, log scale")
    ax.set_title("Mean Runtime by Method")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    table_path = ROOT / "outputs" / "tables" / "experiments.csv"
    plots_dir = ROOT / "outputs" / "plots"
    tables_dir = ROOT / "outputs" / "tables"

    ensure_dir(plots_dir)
    ensure_dir(tables_dir)

    if not table_path.exists():
        raise SystemExit(f"Missing {table_path}. Run experiments first.")

    df = pd.read_csv(table_path)
    if df.empty:
        raise SystemExit(f"{table_path} is empty. Run experiments first.")

    df["target_frames"] = [
        parse_target_frames(run_id, frames)
        for run_id, frames in zip(df["run_id"], df["frames"], strict=True)
    ]
    df["resolution_pixels"] = df["height"] * df["width"]

    summary = (
        df.groupby("method", dropna=False)
        .agg(
            mean_runtime_sec=("runtime_sec", "mean"),
            mean_memory_mb=("peak_memory_mb", "mean"),
            mean_relative_error=("relative_error", "mean"),
            mean_reconstruction_error=("reconstruction_error", "mean"),
            mean_sparsity=("sparsity", "mean"),
            mean_effective_rank=("effective_rank", "mean"),
            success_count=("method", "size"),
        )
        .reset_index()
    )

    summary["order"] = summary["method"].map(lambda x: method_order_key(str(x)))
    summary = summary.sort_values("order").drop(columns=["order"])

    summary_path = tables_dir / "summary_by_method.csv"
    summary.to_csv(summary_path, index=False)

    line_plot(
        df=df,
        x_col="target_frames",
        y_col="runtime_sec",
        output_path=plots_dir / "runtime_vs_frames.png",
        xlabel="Requested frames",
        ylabel="Runtime, seconds, log scale",
        title="Runtime vs Frames",
        log_y=True,
    )

    line_plot(
        df=df,
        x_col="target_frames",
        y_col="peak_memory_mb",
        output_path=plots_dir / "memory_vs_frames.png",
        xlabel="Requested frames",
        ylabel="Peak memory, MB",
        title="Memory vs Frames",
        log_y=False,
    )

    line_plot(
        df=df,
        x_col="resolution_pixels",
        y_col="runtime_sec",
        output_path=plots_dir / "runtime_vs_resolution.png",
        xlabel="Resolution, pixels",
        ylabel="Runtime, seconds, log scale",
        title="Runtime vs Resolution",
        log_y=True,
    )

    scatter_plot_quality_runtime(df, plots_dir / "quality_vs_runtime.png")
    bar_plot_mean_runtime(summary, plots_dir / "mean_runtime_by_method.png")

    print(f"Saved {plots_dir / 'runtime_vs_frames.png'}")
    print(f"Saved {plots_dir / 'memory_vs_frames.png'}")
    print(f"Saved {plots_dir / 'runtime_vs_resolution.png'}")
    print(f"Saved {plots_dir / 'quality_vs_runtime.png'}")
    print(f"Saved {plots_dir / 'mean_runtime_by_method.png'}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
