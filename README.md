# Robust PCA for Video Background Subtraction

Course project for **Optimization Methods**.

The project compares several low-rank + sparse decomposition methods for video background subtraction. Each video is converted into a matrix `M = pixels × frames`, then decomposed into a low-rank background component `L` and a sparse foreground/motion component `S`.

Main focus:

- application of Robust PCA to video data;
- scalability by number of frames and resolution;
- runtime and memory comparison;
- GPU-based linear algebra with PyTorch;
- visual comparison of background/foreground separation.

## Methods

Implemented methods:

```text
pca             classical truncated SVD baseline
randomized_pca  randomized low-rank baseline
godec           approximate batch Robust PCA
windowed_godec  windowed/block GoDec for better scalability
rpca_admm       classical Robust PCA via ADMM / PCP
```

The methods share a common interface and are registered in `src/methods/registry.py`, so a new method can be added by creating a new method module and registering it there.

## Repository Structure

```text
data/raw/              input videos, ignored by git
data/processed/        optional processed data
outputs/images/        decomposition examples
outputs/plots/         generated plots
outputs/tables/        experiment CSV files
outputs/runs/          per-run JSON metadata

scripts/check_env.py
scripts/init_project.sh
scripts/run_experiment.py
scripts/run_scalability.py
scripts/make_report_assets.py

src/data.py
src/metrics.py
src/visualization.py
src/utils.py
src/methods/
```

## Environment

Recommended:

```text
Python >= 3.11
PyTorch with CUDA support
OpenCV
NumPy
pandas
matplotlib
scipy
scikit-learn
psutil
tqdm
PyYAML
```

The project was tested on:

```text
Python 3.12.9
PyTorch 2.6.0+cu126
GPU: NVIDIA A100-SXM4-80GB
```

## Setup

On a machine with internet access and conda:

```bash
bash scripts/init_project.sh
conda activate rpca-video
```

If conda activation fails after initialization:

```bash
conda init bash
source ~/.bashrc
conda activate rpca-video
```

On a prepared server environment where dependencies are already installed, the project can be used without creating a new conda environment:

```bash
bash scripts/init_project.sh
python scripts/check_env.py
```

## Data

Put input videos manually into:

```text
data/raw/
```

Expected names used in experiments:

```text
data/raw/video_01_simple.mp4
data/raw/video_02_medium.mp4
data/raw/video_03_hard.mp4
```

Supported formats:

```text
.mp4
.avi
.mov
```

Video files are ignored by git.

## Run One Experiment

```bash
python scripts/run_experiment.py \
  --video data/raw/video_01_simple.mp4 \
  --method rpca_admm \
  --height 120 \
  --width 160 \
  --frames 200 \
  --device cuda
```

Available methods:

```text
pca
randomized_pca
godec
windowed_godec
rpca_admm
```

## Run Full Scalability Benchmark

```bash
python scripts/run_scalability.py \
  --videos data/raw/video_01_simple.mp4 data/raw/video_02_medium.mp4 data/raw/video_03_hard.mp4 \
  --methods pca randomized_pca godec windowed_godec rpca_admm \
  --frames 50 100 200 400 \
  --resolutions 80x60 160x120 320x240 \
  --device cuda \
  2>&1 | tee outputs/runs/scalability_full.log
```

Resolution values are written as:

```text
widthxheight
```

## Generate Report Assets

```bash
python scripts/make_report_assets.py
```

This reads:

```text
outputs/tables/experiments.csv
```

and generates:

```text
outputs/tables/summary_by_method.csv
outputs/plots/runtime_vs_frames.png
outputs/plots/runtime_vs_resolution.png
outputs/plots/memory_vs_frames.png
outputs/plots/quality_vs_runtime.png
outputs/plots/mean_runtime_by_method.png
```

## Experiment Outputs

Each run saves:

```text
outputs/runs/*.json       run metadata and metrics
outputs/images/*.png      original / background / foreground / mask grids
outputs/tables/*.csv      experiment tables
outputs/plots/*.png       presentation-ready plots
```

## Final Benchmark Summary

Full benchmark setup:

```text
3 videos
5 methods
4 frame counts
3 resolutions
180 total runs
```

Average results from the final run:

| Method | Mean runtime, sec | Mean memory, MB | Mean relative residual |
|---|---:|---:|---:|
| randomized_pca | 0.0048 | 74.75 | 0.0000 |
| pca | 0.0227 | 107.85 | 0.0000 |
| windowed_godec | 1.9767 | 170.97 | 0.0192 |
| godec | 2.3948 | 211.26 | 0.0311 |
| rpca_admm | 3.4021 | 211.10 | 0.0085 |

Important note: for `pca` and `randomized_pca`, the relative residual is zero because their sparse component is defined as `S = M - L`. This does not mean they provide the best foreground separation. They are used as low-rank speed baselines.

## Main Observations

- `randomized_pca` is the fastest method, but it is not robust.
- `pca` is a simple baseline and is also very fast.
- `rpca_admm` gives the lowest decomposition residual among Robust PCA methods, but it is the slowest.
- `godec` is a faster approximate Robust PCA method.
- `windowed_godec` improves scalability compared to batch GoDec: it is faster, uses less memory on average, and gives lower residual in the final benchmark.
- Memory consumption grows with the number of frames because video is represented as a `pixels × frames` matrix.
- GoDec and ADMM use similar memory in this implementation because both rely on SVD-based low-rank updates and keep similar GPU tensors.

## Git Notes

Raw videos and generated heavy outputs should not be committed. The repository keeps only `.gitkeep` files inside data/output folders.
