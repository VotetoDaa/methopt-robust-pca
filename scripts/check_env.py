from __future__ import annotations

import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FOLDERS = [
    "data/raw",
    "data/processed",
    "outputs/images",
    "outputs/plots",
    "outputs/tables",
    "outputs/runs",
]


def import_required(module_name: str, display_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        print(f"{display_name}: FAIL, missing package '{module_name}'")
        raise SystemExit(1) from exc


def main() -> None:
    major, minor, micro = sys.version_info[:3]
    if (major, minor) >= (3, 11):
        print(f"Python: OK, {major}.{minor}.{micro}")
    else:
        print(f"Python: FAIL, {major}.{minor}.{micro} (expected >= 3.11)")
        raise SystemExit(1)

    torch = import_required("torch", "Torch")
    cv2 = import_required("cv2", "OpenCV")
    np = import_required("numpy", "NumPy")
    pd = import_required("pandas", "pandas")
    psutil = import_required("psutil", "psutil")

    print(f"Torch: OK, {torch.__version__}")
    if torch.cuda.is_available():
        print("CUDA: OK")
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        gpu_memory_gb = props.total_memory / (1024**3)
        print(f"GPU: {gpu_name}")
        print(f"GPU memory: {gpu_memory_gb:.2f} GB")
    else:
        print("CUDA: NOT AVAILABLE")
        print("GPU: NOT AVAILABLE")
        print("GPU memory: NOT AVAILABLE")

    print(f"CPU cores: {psutil.cpu_count(logical=True)}")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print(f"OpenCV: OK, {cv2.__version__}")
    print(f"NumPy: OK, {np.__version__}")
    print(f"pandas: OK, {pd.__version__}")

    missing = [folder for folder in REQUIRED_FOLDERS if not (ROOT / folder).is_dir()]
    if missing:
        print(f"Folders: FAIL, missing {', '.join(missing)}")
        raise SystemExit(1)

    print("Folders: OK")


if __name__ == "__main__":
    main()
