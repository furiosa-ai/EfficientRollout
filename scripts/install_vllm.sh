#!/bin/bash
# Install vendored vLLM 0.11.2 from third_party/vllm
#
# Builds C++/CUDA kernels from source (csrc/).
# Python source is installed in editable mode — changes to vllm/ take effect immediately.
# CUDA kernel changes require re-running this script.
#
# Prerequisites:
#   - PyTorch 2.9.0 already installed
#   - CUDA toolkit 12.8+ (nvcc)
#
# Usage:
#   bash scripts/install_vllm.sh

set -euo pipefail


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VLLM_DIR="$PROJECT_ROOT/third_party/vllm"

if [ ! -d "$VLLM_DIR/vllm" ]; then
    echo "ERROR: vendored vLLM not found at $VLLM_DIR"
    exit 1
fi

# Build dependencies (--no-build-isolation skips pyproject.toml auto-install)
pip install cmake ninja setuptools setuptools-scm packaging wheel jinja2

# Hardcode version since we removed .git (setuptools_scm needs it)
export SETUPTOOLS_SCM_PRETEND_VERSION="0.11.2"
export VLLM_VERSION_OVERRIDE="0.11.2"

echo "=== Installing vendored vLLM 0.11.2 (editable + full C++/CUDA build) ==="
pip install -e "$VLLM_DIR" --no-build-isolation

echo ""
echo "=== Verifying installation ==="
python -c "import vllm; print(f'vLLM {vllm.__version__} installed from: {vllm.__file__}')"

echo ""
echo "=== Done ==="
echo "Vendored vLLM 0.11.2 is now installed in editable mode."
echo "Python changes to $VLLM_DIR/vllm/ take effect immediately."
echo "CUDA kernel changes in $VLLM_DIR/csrc/ require re-running this script."
