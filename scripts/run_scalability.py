from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_experiment import run_single_experiment
from src.methods.registry import available_methods


def parse_resolution(value: str) -> tuple[int, int]:
    normalized = value.lower().replace(" ", "")
    if "x" not in normalized:
        raise argparse.ArgumentTypeError("Resolution must be formatted as widthxheight, for example 160x120.")
    width_str, height_str = normalized.split("x", maxsplit=1)
    try:
        width = int(width_str)
        height = int(height_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Resolution width and height must be integers.") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Resolution width and height must be positive.")
    return width, height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scalability experiments.")
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", default=["pca", "rpca_admm", "godec"], choices=available_methods())
    parser.add_argument("--frames", nargs="+", required=True, type=int)
    parser.add_argument("--resolutions", nargs="+", required=True, type=parse_resolution)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = [
        (video, method, frame_count, resolution)
        for video in args.videos
        for method in args.methods
        for frame_count in args.frames
        for resolution in args.resolutions
    ]

    failures: list[str] = []
    for video, method, frame_count, (width, height) in tqdm(jobs, desc="Experiments"):
        try:
            run_single_experiment(
                video=video,
                method=method,
                height=height,
                width=width,
                frames=frame_count,
                device=args.device,
                rank=args.rank,
                max_iter=args.max_iter,
                tol=args.tol,
                output_dir=args.output_dir,
                seed=args.seed,
            )
        except Exception as exc:  # noqa: BLE001 - keep long benchmark moving and report failures clearly.
            message = f"{video} | {method} | {width}x{height} | frames={frame_count}: {exc}"
            failures.append(message)
            print(f"FAIL: {message}")

    if failures:
        print("")
        print("Completed with failures:")
        for failure in failures:
            print(f"  {failure}")
        raise SystemExit(1)

    print("All scalability experiments completed.")


if __name__ == "__main__":
    main()
