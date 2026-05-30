#!/usr/bin/env bash
set -e

echo "== Robust PCA Video Project initializer =="

mkdir -p data/raw data/processed outputs/images outputs/plots outputs/tables outputs/runs

touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch outputs/images/.gitkeep
touch outputs/plots/.gitkeep
touch outputs/tables/.gitkeep
touch outputs/runs/.gitkeep

echo "OK: project folders created"

echo ""
echo "Checking current Python environment..."
python scripts/check_env.py

echo ""
echo "Project is ready in the current environment."
echo "Run experiments with:"
echo "  python scripts/run_experiment.py --video data/raw/video_01_simple.mp4 --method pca --height 120 --width 160 --frames 200 --device cuda"
