# sd_toggle

Offline calibration for roofline-aware speculative decoding (SD) toggle decisions.

Measures component latencies on your GPU, fits a roofline cost model, and
produces a config JSON that predicts whether SD is beneficial for any
`(batch, seq, gamma, L_accept)` operating point.

**Who it's for:** ML engineers running vLLM-based RL rollouts who want to auto-toggle
speculative decoding based on current batch size and sequence length.

---

## Quick Start

The e2e pipeline runs TP=1 on all three models. The easiest path is the per-model helper scripts (see repo-root `README.md` §4), which default to 4-GPU parallel sweeps. For ad-hoc / single-GPU use:

```bash
python -m sd_toggle calibrate --model Qwen/Qwen2.5-7B --gpu-id 0
```

This runs a GPU sweep (~30-40min), fits the roofline model, and saves a calibrated config JSON.

> TP>1 is still supported in `sweep.py` / `fit.py` (pass `--tp N`) for cross-TP ablations, but it is not part of the production calibration flow documented below.

---

## New Server Calibration

When moving to a new server (or a new GPU type), **re-calibrate** — `BW_eff`, `F_eff`, `eta_d`, and overhead
constants are hardware-instance-specific.

**Recommended path (per-model helpers)**: the `scripts/calibrate_*.sh` scripts run Steps 0–3 in one go.

```bash
bash scripts/calibrate_qwen_7b.sh
bash scripts/calibrate_llama_instruct.sh
bash scripts/calibrate_qwen_14b.sh
```

The launcher scripts load the matching config automatically:

```bash
bash scripts/run_qwen2.5_7b_sd.sh toggle
```

If you prefer to run the pipeline manually, the individual steps are below.

### Step 0 — Measure F_eff (once per GPU type)

```bash
python scripts/profiling/measure_gemm_effective_tflops.py \
    --models "Qwen2.5-7B:1,Llama-3.1-8B-Instruct:1,Qwen2.5-14B:1" \
    --gpu 0
# Output defaults to sd_toggle/configs/F_eff_bench_<gpu_short>.json
# (e.g. F_eff_bench_a100.json) — auto-derived from torch.cuda.get_device_name().
```

Measures FLOPs-weighted saturation throughput from representative GEMM shapes.
This is a **hardware constant** shared across all models on the same GPU.
Pre-existing A100 bench (`F_eff_bench_a100.json`) ships with the repo.

### Step 0b — Measure c_comm (TP>1 only)

Skip this step if all models run TP=1 (the current e2e pipeline default). Only needed
when any model uses `tensor_model_parallel_size >= 2`:

```bash
torchrun --nproc_per_node=2 scripts/profiling/measure_nccl_allreduce.py \
    --output sd_toggle/configs/c_comm_bench_a100.json   # adjust suffix per GPU
```

### Step 1 — Sweep (per model)

```bash
# Single-GPU sequential (AR + each γ in series on one GPU)
python -m sd_toggle sweep --model Qwen/Qwen2.5-7B --gpu 0 --gammas 3,7,11,15

# Multi-pass parallel (recommended — matches production contention)
# 5 GPUs → AR + γ=3 + γ=7 + γ=11 + γ=15 run concurrently (one pass per GPU).
python -m sd_toggle sweep --model Qwen/Qwen2.5-7B              --gpus 0,1,2,3,4 --gammas 3,7,11,15
python -m sd_toggle sweep --model meta-llama/Llama-3.1-8B-Instruct --gpus 0,1,2,3,4 --gammas 3,7,11,15
python -m sd_toggle sweep --model Qwen/Qwen2.5-14B             --gpus 0,1,2,3,4 --gammas 3,7,11,15
```

> **Why parallel?** Production RL rollouts run 8 vLLM workers simultaneously. Running
> sweep passes concurrently matches that CPU/NCCL contention level, so the fitted
> `c_T`, `c_D`, `c_V`, and `c_comm` reflect conditions the toggle will face at runtime.
> Single-process sweeps under-estimate overhead → optimistic `predict_speedup` near B\*.

### Step 2 — Calibrate

The `--gpu` argument is a label used for the output filename prefix
(e.g. `a100_tp1_*.json`) and config metadata; it does not change the fit itself.

```bash
python -m sd_toggle calibrate --csv sweep_qwen257b.csv --model Qwen2.5-7B \
    --gpu A100 --F-eff-bench sd_toggle/configs/F_eff_bench_a100.json
```

Repeat for `Llama-3.1-8B-Instruct` and `Qwen2.5-14B`.

### Step 3 — Validate

After calibration, verify the model accuracy against sweep data:

```bash
# Pass measured L_accept from your runtime (e.g. rollout/sd/mean_acceptance_length)
python -m sd_toggle validate \
    --config sd_toggle/configs/a100_tp1_qwen257b.json \
    --csv sd_toggle/configs/sweep_a100_tp1_qwen257b.csv \
    --L-accept 6.7
```

Reports per-gamma: T_D/T_V accuracy (R², MAPE), ratio accuracy (r, v),
T_cycle MAPE, speedup bias/MAE, toggle sign accuracy. Auto-checks:
- T_T MAPE < 10%
- Sign accuracy ≥ 90%
- BW_eff and η_d in physical range

### Step 4 — Predict / Use

```bash
python -m sd_toggle predict --config sd_toggle/configs/a100_tp1_qwen257b.json \
    --batch 16 --seq 2048 --gamma 7 --L-accept 6.1
```

### GPU allocation on multi-GPU nodes

| Model | Minimum | Recommended (production-match) | `--gpus` |
|---|---|---|---|
| 7B / 8B-Instruct / 14B (all TP=1) | 1 GPU sequential | 5 GPUs parallel (5 passes) | `0,1,2,3,4` |
| GEMM bench | 1 GPU only | — | `--gpu 0` |

- **One pass (AR or SD γ=N) per GPU.** 5 passes map 1:1 onto 5 GPUs for γ∈{3,7,11,15}.
- **Do not mix**: cross-model concurrent sweeps on the same node stress CPU
  differently than production; keep one model at a time.
- **Per-worker temp files** (`/tmp/sd_timing_output_gpu{gpu}.txt`,
  `/tmp/torchinductor_gpu{gpu}`) are already disambiguated — no manual
  `CUDA_VISIBLE_DEVICES` needed.

---

## Calibration Pipeline

### Stage 0: F_eff (hardware constant)

F_eff is the GPU's sustained FP16 tensor-core throughput in the compute-bound
saturation regime. Measured once per GPU type via GEMM micro-benchmarks.

Each GEMM shape (attn_qkv, attn_out, ffn_gate_up, ffn_down) is weighted by
its share of per-token dense compute (C_dense). The cross-model median of
FLOPs-weighted saturation throughput is the hardware F_eff.

**A100-SXM4-80GB: F_eff = 212 TFLOPS** (68% of peak 312 TFLOPS)

### Stage 1: BW_eff, κ, c_T (from T_T)

Fits effective memory bandwidth, KV cache coefficient, and target decode
overhead from target decode (γ=0) sweep data. Log-MSE loss.

### Stage 2: η_d, c_D, c_V (from T_D + T_V, hybrid loss)

Fits drafter overhead multiplier and batch overheads from drafter + verify
sweep data across all gammas simultaneously. **All 3 params are γ-shared.**

Hybrid loss = 0.5 · component accuracy + 0.5 · Lv2 primitive accuracy:
- Component: log-MSE on T_D and T_V individually
- Lv2 primitive: log-MSE on T_cycle (γ·T_D + T_V) + MSE on speedup ratio

### Total: 6 fitted params + 1 measured

| Param | Stage | Meaning | Scope |
|-------|-------|---------|-------|
| F_eff | S0 | Sustained FP16 throughput | hardware |
| BW_eff | S1 | Effective memory bandwidth | hardware |
| κ | S1 | Effective KV cache coefficient | model |
| c_T | S1 | Target decode overhead (μs/batch) | model |
| η_d | S2 | Drafter memory overhead multiplier | model |
| c_D | S2 | Drafter overhead (μs/batch) | model |
| c_V | S2 | Verify overhead (μs/batch) | model |

---

## Formula Reference

### Component latency model

All modes share the roofline form: `T = max(M, C) + h·B·1e-6`

```
T_T(B, S)    = max(M_T, C_T) + c_T · B · 1e-6       [target decode]
T_D(B, S)    = max(M_D, C_D) + c_D · B · 1e-6       [drafter decode]
T_V(B, S, γ) = max(M_V, C_V) + c_V · B · 1e-6       [verify]

where:
  M_T = (W_t + κ · B · S) / BW_eff
  M_D = (W_d + η_d · κ · B · S) / BW_eff
  M_V = (W_t + κ · B · S) / BW_eff
  C_T = C_D = (B · C_dense + B · S · C_attn) / F_eff
  C_V = (B·(γ+1)·C_dense + B·S·(γ+1)·C_attn) / F_eff
```

### Speedup and toggle decision

```
speedup = L̄ / (γ · r + v)

r = T_D / T_T   [drafter cost ratio]
v = T_V / T_T   [verify cost ratio]
L̄ = (1 - α^(γ+1)) / (1 - α)   [expected accepted tokens]

SD ON  when speedup > 1.0
SD OFF when speedup ≤ 1.0
```

### Lv2 makespan cost

```
AR cost:  T_step = T_T
SD cost:  T_step = (γ · T_D + T_V) / L̄
```

---

## Config Schema

```json
{
  "hardware": {
    "gpu": "A100", "tp": 1,
    "BW_eff": 1.504e12, "BW_peak": 2.039e12, "F_peak": 3.12e14
  },
  "model": {
    "name": "Qwen2.5-7B",
    "W_t": 14769689600, "W_d": 4860646400,
    "C_dense": 13990625280, "C_attn": 458752,
    "kappa_theoretical": 57344, "rho": 0.329, "gqa": 7
  },
  "calibration": {
    "eta_d": 1.373, "kappa_eff": 12204,
    "F_eff": 211937000000000,
    "c_T": 22.3, "c_D": 66.5, "c_V": 71.8,
    "beta": 0.0
  },
  "metadata": {
    "created": "2026-04-18",
    "csv_source": "sweep_a100_tp1_qwen257b.csv",
    "F_eff_tflops": 211.9,
    "F_eff_source": "sd_toggle/configs/F_eff_bench_${gpu_short}.json",
    "gammas": [3, 7, 11, 15]
  }
}
```

### Pre-calibrated configs

Naming: `${gpu_short}_tp${TP}_${model_short}.json` (lowercase, first token of GPU name — `A100-SXM4` → `a100`).

| Config | Model | TP | Notes |
|--------|-------|----|-------|
| `a100_tp1_qwen257b.json` | Qwen2.5-7B | 1 | Calibrated on γ∈{3,7,11,15} |
| `a100_tp1_llama318binstruct.json` | LLaMA-3.1-8B-Instruct | 1 | Calibrated on γ∈{3,7,11,15} |
| `a100_tp1_qwen2514b.json` | Qwen2.5-14B | 1 | Calibrated on γ∈{3,7,11,15} |

---

## CLI Reference

### `sweep` — measure component latencies

```
python -m sd_toggle sweep
    --model MODEL       HuggingFace model ID (required)
    --gpu INT           GPU index for single-GPU mode (default: 0)
    --gpus LIST         Comma-separated GPU indices for parallel sweep.
                        For TP>1, list PRIMARY indices only; each primary p
                        expands to pair (p, p+1, …, p+tp-1).
                        Examples:
                          TP=1, 4 passes parallel:  --gpus 0,1,2,3
                          TP=2, 4 pairs parallel:   --gpus 0,2,4,6 --tp 2
    --tp INT            Tensor parallelism (default: 1)
    --gammas LIST       Comma-separated gamma values (default: 3,7,11,15)
    --batches LIST      Comma-separated batch sizes
    --seq-lens LIST     Comma-separated sequence lengths
    --output PATH       Output CSV path
```

### `calibrate` — fit roofline model

```
python -m sd_toggle calibrate
    --model MODEL       Model name (required)
    --csv PATH          Sweep CSV (if omitted, auto-sweep with --gpu-id)
    --gpu-id IDS        GPU index(es) for auto-sweep (e.g. '0' or '0,1')
    --gpu NAME          GPU name for metadata (default: A100)
    --tp INT            Tensor parallelism (auto-detected from --gpu-id)
    --gammas LIST       Comma-separated gammas (auto-detect from CSV)
    --F-eff FLOAT       Fixed F_eff in FLOPS (default: 200e12)
    --F-eff-bench PATH  GEMM bench JSON (overrides --F-eff)
    --output PATH       Output path or directory (default: sd_toggle/configs/)
```

### `predict` — single-point toggle decision

```
python -m sd_toggle predict
    --config PATH       Config JSON (required)
    --batch INT         Batch size (required)
    --seq INT           Sequence length (required)
    --gamma INT         Draft token count (required)
    --L-accept FLOAT    Expected accepted length (required — pass measured value)
```

### `validate` — check calibration accuracy

```
python -m sd_toggle validate
    --config PATH       Config JSON (required)
    --csv PATH          Sweep CSV (required)
    --L-accept FLOAT    Expected accepted length (required — pass measured value)
```

Reports per-gamma: T_D/T_V R² and MAPE, ratio bias/MAE, T_cycle MAPE,
speedup bias/MAE, toggle sign accuracy. Auto-checks T_T MAPE, sign ≥ 90%,
BW_eff and η_d in physical range.

### `plot` — boundary visualization

```
python -m sd_toggle plot
    --config PATH       Config JSON (required)
    --csv PATH          Sweep CSV for empirical overlay
    --output DIR        Output directory
    --L-accepts LIST    L_accept values (default: 4.5,5.0,5.5)
```

### `info` — config summary

```
python -m sd_toggle info --config PATH
```
