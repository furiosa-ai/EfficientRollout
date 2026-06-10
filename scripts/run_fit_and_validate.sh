#!/bin/bash
# Fit roofline models from sweep CSVs and validate
# Usage: bash scripts/run_fit_and_validate.sh
set -e

CONDA_ENV="efficientrollout"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="${PROJECT_ROOT}/results/lv1_roofline"
CONFIG_DIR="${PROJECT_ROOT}/sd_toggle/configs"
PLOT_DIR="${PROJECT_ROOT}/results/sd_toggle_plots"
QWEN14B_CSV="$RESULTS_DIR/qwen2514b_all_gammas.csv"

if [ ! -f "$QWEN14B_CSV" ] && [ -f "$RESULTS_DIR/qwen2514b_tp2_all_gammas.csv" ]; then
    echo "WARNING: using legacy 14B sweep CSV: $RESULTS_DIR/qwen2514b_tp2_all_gammas.csv"
    QWEN14B_CSV="$RESULTS_DIR/qwen2514b_tp2_all_gammas.csv"
fi

mkdir -p "$CONFIG_DIR" "$PLOT_DIR"

echo "=========================================="
echo " Phase 2: Fitting roofline models"
echo "=========================================="

# --- Qwen2.5-7B TP=1 ---
echo ""
echo "--- Qwen2.5-7B TP=1 ---"
conda run -n $CONDA_ENV python -m sd_toggle calibrate \
    --csv "$RESULTS_DIR/qwen257b_all_gammas.csv" \
    --model Qwen2.5-7B \
    --gpu A100 --tp 1 \
    --gammas 3,7,11,15 \
    --output "$CONFIG_DIR/a100_tp1_qwen257b.json"

# --- LLaMA-3.1-8B-Instruct TP=1 ---
echo ""
echo "--- LLaMA-3.1-8B-Instruct TP=1 ---"
conda run -n $CONDA_ENV python -m sd_toggle calibrate \
    --csv "$RESULTS_DIR/llama318b_instruct_all_gammas.csv" \
    --model Llama-3.1-8B-Instruct \
    --gpu A100 --tp 1 \
    --gammas 3,7,11,15 \
    --output "$CONFIG_DIR/a100_tp1_llama318binstruct.json"

# --- Qwen2.5-14B TP=1 ---
echo ""
echo "--- Qwen2.5-14B TP=1 ---"
conda run -n $CONDA_ENV python -m sd_toggle calibrate \
    --csv "$QWEN14B_CSV" \
    --model Qwen2.5-14B \
    --gpu A100 --tp 1 \
    --gammas 3,7,11,15 \
    --output "$CONFIG_DIR/a100_tp1_qwen2514b.json"

echo ""
echo "=========================================="
echo " Phase 3: Validation"
echo "=========================================="

# --- Validate BW_eff, R², sign accuracy ---
conda run -n $CONDA_ENV python -c "
from sd_toggle.config import load_config
from sd_toggle.predict import evaluate_sign_accuracy
import json, glob

configs = {
    'Qwen2.5-7B (TP=1)': ('$CONFIG_DIR/a100_tp1_qwen257b.json', '$RESULTS_DIR/qwen257b_all_gammas.csv'),
    'LLaMA-Instruct (TP=1)': ('$CONFIG_DIR/a100_tp1_llama318binstruct.json', '$RESULTS_DIR/llama318b_instruct_all_gammas.csv'),
    'Qwen2.5-14B (TP=1)': ('$CONFIG_DIR/a100_tp1_qwen2514b.json', '$QWEN14B_CSV'),
}

print('=' * 70)
print('VALIDATION RESULTS')
print('=' * 70)

all_pass = True
for name, (cfg_path, csv_path) in configs.items():
    cfg = load_config(cfg_path)
    print(f'\n--- {name} ---')
    print(f'  BW_eff = {cfg.hardware.BW_eff/1e12:.3f} TB/s (expect ~1.6)')
    print(f'  F_eff  = {cfg.calibration.F_eff/1e12:.0f} TFLOPS')
    print(f'  eta_d  = {cfg.calibration.eta_d:.3f}')
    print(f'  kappa  = {cfg.calibration.kappa_eff:.0f}')

    per_gamma = cfg.calibration.per_gamma
    gammas = sorted(per_gamma.keys()) if per_gamma else sorted(cfg.metadata.get('gammas', []))

    if per_gamma:
        for g in gammas:
            cal = per_gamma[g]
            r2_ok = '✓' if cal.R2 >= 0.80 else '✗'
            print(f'  gamma={g}: R²={cal.R2:.3f} {r2_ok}  c_D={cal.c_D:.1f}  c_V={cal.c_V:.1f}')
            if cal.R2 < 0.80:
                all_pass = False
    elif gammas:
        s2 = cfg.metadata.get('s2_diagnostics', {})
        r2 = s2.get('R2')
        r2_txt = f'{r2:.3f}' if isinstance(r2, (int, float)) else 'n/a'
        r2_ok = '✓' if isinstance(r2, (int, float)) and r2 >= 0.80 else '✗'
        print(f'  shared-fit gammas={gammas}: R²={r2_txt} {r2_ok}  c_D={cfg.calibration.c_D:.1f}  c_V={cfg.calibration.c_V:.1f}')
        if not (isinstance(r2, (int, float)) and r2 >= 0.80):
            all_pass = False

    # Sign accuracy at L=γ (trained-policy regime)
    for g in gammas:
        L = float(g)
        try:
            acc = evaluate_sign_accuracy(cfg, csv_path, L, [g])
            sign_ok = '✓' if acc >= 0.90 else '✗'
            print(f'  gamma={g}: sign_acc={acc:.1%} {sign_ok} (L=γ)')
            if acc < 0.90:
                all_pass = False
        except Exception as e:
            print(f'  gamma={g}: sign_acc ERROR: {e}')

print(f'\n{\"=\" * 70}')
print(f'OVERALL: {\"ALL PASS ✓\" if all_pass else \"SOME FAILED ✗\"}')
print(f'{\"=\" * 70}')
"

# --- Boundary plots ---
echo ""
echo "=========================================="
echo " Phase 3b: Boundary curve plots"
echo "=========================================="

for cfg_file in "$CONFIG_DIR/a100_tp1_qwen257b.json" \
                "$CONFIG_DIR/a100_tp1_llama318binstruct.json" \
                "$CONFIG_DIR/a100_tp1_qwen2514b.json"; do

    model_name=$(basename "$cfg_file" .json)
    csv_file=""
    case "$model_name" in
        *qwen257b*) csv_file="$RESULTS_DIR/qwen257b_all_gammas.csv" ;;
        *llama*) csv_file="$RESULTS_DIR/llama318b_instruct_all_gammas.csv" ;;
        *qwen2514b*) csv_file="$QWEN14B_CSV" ;;
    esac

    echo "Plotting $model_name..."
    conda run -n $CONDA_ENV python -m sd_toggle plot \
        --config "$cfg_file" \
        --csv "$csv_file" \
        --output "$PLOT_DIR/$model_name/" \
        --L-accepts 3.5,4.0,4.5,5.0,5.5,6.0 \
        --no-pow2
done

echo ""
echo "=========================================="
echo " Done! Plots saved to: $PLOT_DIR"
echo "=========================================="
echo "Review boundary curves and confirm validation."
