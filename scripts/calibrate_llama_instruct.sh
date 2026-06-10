#!/bin/bash
# Calibrate SD toggle roofline model for LLaMA-3.1-8B-Instruct
#
# Usage:
#   bash scripts/calibrate_llama_instruct.sh                         # A100 default, uses 5 GPUs (AR + 4 γ)
#   bash scripts/calibrate_llama_instruct.sh --gpus 0,1,2,3,4        # custom GPU subset (match pass count)
#
# Pipeline: multi-GPU sweep → fit → save config
# Output:  sd_toggle/configs/${gpu_short}_tp1_llama318binstruct.json (overwrite)
#          e.g. a100_tp1_llama318binstruct.json
#
# GAMMAS=3,7,11,15 → 5 passes (AR/γ=0, γ=3, γ=7, γ=11, γ=15). 5 GPUs match 1:1.
# Time estimate: ~20-25 min (5 GPUs, 1 pass/GPU)

set -euo pipefail

# Ensure conda env is active (needed for nohup / cron execution)
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "$CONDA_DEFAULT_ENV" != "efficientrollout" ]]; then
    source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
    conda activate efficientrollout 2>/dev/null || true
fi

MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SHORT="Llama-3.1-8B-Instruct"
GPU_NAME="${GPU_NAME:-A100}"
GPU_SHORT=$(echo "$GPU_NAME" | tr '[:upper:]' '[:lower:]' | cut -d- -f1)
TP=1

# Parse --gpus argument; default to 5 GPUs matching 5 passes (AR + γ∈{3,7,11,15})
GPUS_ARG="0,1,2,3,4"
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus) GPUS_ARG="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done
GPUS="--gpus $GPUS_ARG"

# Preflight checks
python -c "import vllm; import sd_toggle" 2>/dev/null || { echo "ERROR: activate efficientrollout conda env first"; exit 1; }
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('$MODEL', trust_remote_code=True)" 2>/dev/null || { echo "ERROR: model not downloaded. Run: huggingface-cli download $MODEL"; exit 1; }
NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
echo "Detected ${NUM_GPUS} GPUs"

GAMMAS="3,7,11,15"
BATCHES="1,2,4,8,16,24,32,48,64"
SEQ_LENS="256,512,1024,2048,4096"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SWEEP_CSV="results/recalibrate_${TIMESTAMP}/llama_instruct_sweep.csv"
CONFIG_DIR="sd_toggle/configs"
RESULTS_DIR=$(dirname "$SWEEP_CSV")

mkdir -p "$RESULTS_DIR" "$CONFIG_DIR"

echo "============================================================"
echo "SD Toggle Calibration: ${MODEL_SHORT}"
echo "============================================================"
echo "  GPUs:     ${GPUS_ARG}"
echo "  TP:       ${TP}"
echo "  Gammas:   ${GAMMAS}"
echo "  Batches:  ${BATCHES}"
echo "  Seq lens: ${SEQ_LENS}"
echo "  Output:   ${CONFIG_DIR}/"
echo "============================================================"
echo ""

# --- Step 1: Multi-GPU sweep ---
echo "[Step 1/3] Running multi-GPU sweep..."
python -m sd_toggle sweep \
    --model "$MODEL" \
    $GPUS \
    --gammas "$GAMMAS" \
    --batches "$BATCHES" \
    --seq-lens "$SEQ_LENS" \
    --tp "$TP" \
    --output "$SWEEP_CSV"

if [ ! -f "$SWEEP_CSV" ]; then
    echo "ERROR: Sweep CSV not produced at $SWEEP_CSV"
    exit 1
fi

NLINES=$(wc -l < "$SWEEP_CSV")
echo ""
echo "  Sweep complete: ${NLINES} lines in ${SWEEP_CSV}"

# Also mirror CSV to configs/ dir for convenient co-location with fitted JSON
FINAL_CSV="${CONFIG_DIR}/sweep_${GPU_SHORT}_tp${TP}_llama318binstruct.csv"
cp "$SWEEP_CSV" "$FINAL_CSV"
echo "  Sweep CSV also saved: ${FINAL_CSV}"
echo ""

# --- Step 2: Fit roofline model ---
echo "[Step 2/3] Fitting roofline model..."
python -m sd_toggle calibrate \
    --csv "$SWEEP_CSV" \
    --model "$MODEL_SHORT" \
    --gpu "$GPU_NAME" \
    --tp "$TP" \
    --gammas "$GAMMAS" \
    --F-eff-bench sd_toggle/configs/F_eff_bench_${GPU_SHORT}.json \
    --output "$CONFIG_DIR/"

CONFIG_FILE="${CONFIG_DIR}/${GPU_SHORT}_tp${TP}_llama318binstruct.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config not produced at $CONFIG_FILE"
    exit 1
fi

echo ""
echo "  Config saved: ${CONFIG_FILE}"
echo ""

# --- Step 3: Print summary ---
echo "[Step 3/3] Calibration summary:"
python -m sd_toggle info --config "$CONFIG_FILE"

echo ""
echo "============================================================"
echo "DONE. Config: ${CONFIG_FILE}"
echo "Sweep data: ${SWEEP_CSV}"
echo "============================================================"
