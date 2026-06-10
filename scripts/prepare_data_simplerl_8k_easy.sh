#!/bin/bash
# Prepare SimpleRL-Zoo EASY bucket: GSM8K + MATH level 1 (~8K examples)
#
# Models trained on this bucket: Llama-3.1-8B, Mistral-v0.1-7B, DeepSeek-Math-7B
# Source: https://huggingface.co/datasets/hkust-nlp/SimpleRL-Zoo-Data
# Reference: SimpleRL-Zoo (arXiv:2503.18892)
#
# Usage:
#   bash scripts/prepare_data_simplerl_8k_easy.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/prepare_simplerl_zoo.py" --level easy
