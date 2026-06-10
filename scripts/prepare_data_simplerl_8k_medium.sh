#!/bin/bash
# Prepare SimpleRL-Zoo MEDIUM bucket: MATH level 1-4 (~8K examples)
#
# Models trained on this bucket: Qwen-2.5-0.5B
# Source: https://huggingface.co/datasets/hkust-nlp/SimpleRL-Zoo-Data
# Reference: SimpleRL-Zoo (arXiv:2503.18892)
#
# Usage:
#   bash scripts/prepare_data_simplerl_8k_medium.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/prepare_simplerl_zoo.py" --level medium
