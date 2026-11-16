#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

python3 -m venv venv
./venv/bin/python -m pip install \
    torch \
    torchvision \
    monai \
    mlflow \
    flwr \
    gdown \
    numpy \
    Pillow \
    PyYAML \
    scikit-learn \
    matplotlib
git pull
git checkout -b alf_baseline origin/alf_baseline
