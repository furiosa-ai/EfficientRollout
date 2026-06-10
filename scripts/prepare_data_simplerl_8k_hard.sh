#!/bin/bash
# Prepare SimpleRL-Zoo HARD bucket: MATH level 3-5 (~8K examples)
#
# Models trained on this bucket: Qwen-2.5-1.5B/7B/14B/32B, Qwen-2.5-Math-7B, Mistral-Small-24B
# Source: https://huggingface.co/datasets/hkust-nlp/SimpleRL-Zoo-Data
# Reference: SimpleRL-Zoo (arXiv:2503.18892)
#
# Usage:
#   bash scripts/prepare_data_simplerl_8k_hard.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/prepare_simplerl_zoo.py" --level hard
