from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}


def load_video_as_matrix(
    video_path: str,
    height: int,
    width: int,
    max_frames: int | None,
    grayscale: bool = True,
    normalize: bool = True,
    device: str = "cuda",
) -> tuple[torch.Tensor, dict[str, Any]]:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video extension '{path.suffix}'. Use .mp4, .avi, or .mov.")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive.")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive or None.")

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    original_fps = capture.get(cv2.CAP_PROP_FPS)
    frames: list[np.ndarray] = []

    try:
        while True:
            if max_frames is not None and len(frames) >= max_frames:
                break

            ok, frame = capture.read()
            if not ok:
                break

            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            if grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame_array = frame.astype(np.float32)
            if normalize:
                frame_array /= 255.0

            frames.append(frame_array.reshape(-1))
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"No frames were read from video: {video_path}")

    matrix = np.stack(frames, axis=1)
    M = torch.from_numpy(matrix).to(device=device, dtype=torch.float32)
    channels = 1 if grayscale else 3

    metadata: dict[str, Any] = {
        "video_path": str(path),
        "height": height,
        "width": width,
        "frames": M.shape[1],
        "pixels": M.shape[0],
        "original_fps": float(original_fps) if original_fps else None,
        "grayscale": grayscale,
        "normalized": normalize,
        "channels": channels,
    }
    return M, metadata


def matrix_column_to_frame(vector, height: int, width: int):
    if isinstance(vector, torch.Tensor):
        array = vector.detach().cpu().numpy()
    else:
        array = np.asarray(vector)

    array = array.reshape(-1)
    if array.size == height * width:
        return array.reshape(height, width)
    if array.size % (height * width) == 0:
        channels = array.size // (height * width)
        return array.reshape(height, width, channels)
    raise ValueError(f"Cannot reshape vector of size {array.size} into frame {height}x{width}.")
