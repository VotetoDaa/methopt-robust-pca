from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_video_as_matrix
from src.metrics import (
    effective_rank,
    reconstruction_error,
    relative_frobenius_error,
    sparsity,
)
from src.methods.registry import available_methods, get_method
from src.utils import (
    ensure_dir,
    get_device,
    get_peak_memory_mb,
    measure_runtime,
    reset_peak_memory,
    save_json,
    set_seed,
)
from src.visualization import save_decomposition_grid


def build_run_id(video_path: str, method: str, height: int, width: int, frames: int | None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    video_name = Path(video_path).stem
    frame_tag = "all" if frames is None else str(frames)
    return f"{timestamp}_{video_name}_{method}_{width}x{height}_f{frame_tag}"


def append_csv(row: dict[str, Any], csv_path: Path) -> None:
    ensure_dir(csv_path.parent)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def method_kwargs(method_name: str, rank: int, max_iter: int, tol: float) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if method_name in {"pca", "godec"}:
        kwargs["rank"] = rank
    if method_name in {"rpca_admm", "godec"}:
        kwargs["max_iter"] = max_iter
        kwargs["tol"] = tol
    return kwargs


def choose_frame_indices(frame_count: int) -> list[int]:
    if frame_count <= 0:
        return []
    candidates = [0, frame_count // 2, frame_count - 1]
    return sorted(set(idx for idx in candidates if 0 <= idx < frame_count))


def run_single_experiment(
    video: str,
    method: str,
    height: int,
    width: int,
    frames: int | None,
    device: str,
    rank: int,
    max_iter: int,
    tol: float,
    output_dir: str | Path = "outputs",
    seed: int = 42,
) -> dict[str, Any]:
    set_seed(seed)

    output_root = Path(output_dir)
    images_dir = output_root / "images"
    tables_dir = output_root / "tables"
    runs_dir = output_root / "runs"
    for directory in [images_dir, tables_dir, runs_dir]:
        ensure_dir(directory)

    resolved_device = get_device(device)
    run_id = build_run_id(video, method, height, width, frames)

    print(f"Loading video: {video}")
    reset_peak_memory(resolved_device)
    M, metadata = load_video_as_matrix(
        video_path=video,
        height=height,
        width=width,
        max_frames=frames,
        grayscale=True,
        normalize=True,
        device=str(resolved_device),
    )

    method_obj = get_method(method)
    kwargs = method_kwargs(method, rank=rank, max_iter=max_iter, tol=tol)

    print(f"Running method: {method}")
    with measure_runtime(resolved_device) as timer:
        result = method_obj.fit_transform(M, **kwargs)

    peak_memory_mb = get_peak_memory_mb(resolved_device)
    runtime_sec = timer["runtime_sec"]

    print("Computing metrics")
    row = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "method": method,
        "video_name": Path(video).name,
        "video_path": str(video),
        "height": metadata["height"],
        "width": metadata["width"],
        "frames": metadata["frames"],
        "pixels": metadata["pixels"],
        "device": str(resolved_device),
        "runtime_sec": runtime_sec,
        "peak_memory_mb": peak_memory_mb,
        "reconstruction_error": reconstruction_error(M, result.L, result.S),
        "relative_error": relative_frobenius_error(M, result.L, result.S),
        "sparsity": sparsity(result.S),
        "effective_rank": effective_rank(result.L),
        "iterations": result.info.get("iterations"),
        "converged": result.info.get("converged"),
        "rank": result.info.get("rank"),
        "final_error": result.info.get("final_error"),
    }

    image_path = images_dir / f"{run_id}.png"
    save_decomposition_grid(
        M=M,
        L=result.L,
        S=result.S,
        metadata=metadata,
        output_path=image_path,
        frame_indices=choose_frame_indices(metadata["frames"]),
    )

    run_json = {
        "run_id": run_id,
        "metadata": metadata,
        "method_info": result.info,
        "metrics": row,
        "image_path": str(image_path),
    }
    json_path = runs_dir / f"{run_id}.json"
    save_json(run_json, json_path)

    csv_path = tables_dir / "experiments.csv"
    append_csv(row, csv_path)

    print(f"Saved run JSON: {json_path}")
    print(f"Saved image: {image_path}")
    print(f"Appended CSV: {csv_path}")
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Robust PCA video experiment.")
    parser.add_argument("--video", required=True, help="Path to input video.")
    parser.add_argument("--method", required=True, choices=available_methods())
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_single_experiment(
        video=args.video,
        method=args.method,
        height=args.height,
        width=args.width,
        frames=args.frames,
        device=args.device,
        rank=args.rank,
        max_iter=args.max_iter,
        tol=args.tol,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
