#!/bin/bash
# Installation script for promise-bench data curation
#
# Creates two conda environments from scratch:
#   1. promise        - Main env (Python 3.12) + installs promise_data CLI
#   2. prodigy-cryst  - Separate env for prodigy_cryst (Python 3.8)
#
# Usage:
#   bash install.sh          # first-time setup
#   bash install.sh --update # update existing envs
#
# Requires: conda/mamba, and uv (https://docs.astral.sh/uv/getting-started/installation/)
#
# After installation:
#   conda activate promise
#   promise_data run --spec spec.json --mmcif-store /path/to/mmcif_files

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Installing promise-bench curation"
echo "========================================"

# 1. Main environment
echo ""
echo "[1/3] Creating 'promise' conda environment..."
conda env update --name promise --file environment.yaml --prune
echo "  OK: promise environment ready"

# 2. prodigy-cryst environment (Python 3.8, separate due to dependency conflicts)
echo ""
echo "[2/3] Creating 'prodigy-cryst' conda environment..."
conda env update --name prodigy-cryst --file environment-prodigy.yaml --prune
echo "  OK: prodigy-cryst environment ready"

# 3. Install promise-data package into the promise environment (uv)
echo ""
echo "[3/3] Installing promise-data CLI with uv..."
if ! command -v uv >/dev/null 2>&1; then
  echo "  ERROR: uv is not installed. Install it from https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi
UV_PYTHON="$(conda run -n promise which python)"
uv pip install --python "$UV_PYTHON" -e "$PROJECT_ROOT"
echo "  OK: promise_data command installed"

echo ""
echo "========================================"
echo "Installation complete!"
echo "========================================"
echo ""
echo "Quick start:"
echo ""
echo "  conda activate promise"
echo "  promise_data steps                  # list pipeline steps"
echo "  promise_data run \\\\                  # run full pipeline"
echo "    --spec spec.json \\\\"
echo "    --mmcif-store /path/to/mmcif_files"
echo ""
