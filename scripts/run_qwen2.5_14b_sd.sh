#!/bin/bash
# veRL E2E: Qwen2.5-14B + GRPO with Quantized Self-Speculative Decoding
#
# Usage:
#   bash scripts/run_qwen2.5_14b_sd.sh no-sd      # No SD baseline
#   bash scripts/run_qwen2.5_14b_sd.sh rtn        # SD with RTN quantized drafter (always-on, fixed γ)
#   bash scripts/run_qwen2.5_14b_sd.sh toggle     # SD with roofline toggle (fixed γ)
#   bash scripts/run_qwen2.5_14b_sd.sh adaptive   # roofline toggle + adaptive-γ (full EfficientRollout)
#
# Hardware: 8x A100-SXM4-80GB (1 node)
# Data: simplerl-8k-hard (MATH lv.3-5)

set -x

MODE=${1:-rtn}
shift  # Remove mode arg so $@ only contains hydra overrides

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

DATA_DIR=${DATA_DIR:-"${PROJECT_ROOT}/data/simplerl-8k-hard-qwen"}

# --- Hardware ---
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# --- Adaptive-γ ---
export VLLM_GAMMA_AR_THRESHOLD=${VLLM_GAMMA_AR_THRESHOLD:-0.94}              # α_up: elevate γ when accept-rate ≥ this
export VLLM_GAMMA_AR_THRESHOLD_LOWER=${VLLM_GAMMA_AR_THRESHOLD_LOWER:-0.85}  # α_down: lower γ when accept-rate ≤ this
export VLLM_GAMMA_PERSISTENCE=${VLLM_GAMMA_PERSISTENCE:-2}
SPEC_GAMMA_LADDER=${SPEC_GAMMA_LADDER:-5,7,9,11}     # 'adaptive' mode: γ ladder (uniform Δγ=2); γ starts at ladder[0]

# --- SD config ---
SD_TOGGLE_THRESHOLD=${SD_TOGGLE_THRESHOLD:-32}       # ramp-up guard: prevent premature toggle during prefill
SD_TOGGLE_MARGIN=${SD_TOGGLE_MARGIN:-0.05}
SPEC_NUM_DRAFT_TOKENS=${SPEC_NUM_DRAFT_TOKENS:-5}
GPU_SHORT="a100"
SD_TOGGLE_CONFIG=${SD_TOGGLE_CONFIG:-"${PROJECT_ROOT}/sd_toggle/configs/${GPU_SHORT}_tp1_qwen2514b.json"}

SD_ARGS=""
case $MODE in
  no-sd)
    echo "=== Qwen2.5-14B No-SD Baseline ==="
    SD_ARGS=""
    EXPERIMENT_NAME="qwen14b_baseline_nosd"
    ;;
  rtn)
    echo "=== Qwen2.5-14B SD with RTN Quantized Drafter (always-on) ==="
    SD_ARGS="+actor_rollout_ref.rollout.spec_method=quant_self +actor_rollout_ref.rollout.spec_num_draft_tokens=${SPEC_NUM_DRAFT_TOKENS} +actor_rollout_ref.rollout.spec_quant_method=rtn"
    EXPERIMENT_NAME="qwen14b_sd_rtn"
    ;;
  toggle)
    echo "=== Qwen2.5-14B SD with roofline toggle (margin=${SD_TOGGLE_MARGIN}) ==="
    SD_ARGS="+actor_rollout_ref.rollout.spec_method=quant_self +actor_rollout_ref.rollout.spec_num_draft_tokens=${SPEC_NUM_DRAFT_TOKENS} +actor_rollout_ref.rollout.spec_quant_method=rtn +actor_rollout_ref.rollout.spec_sd_toggle_mode=roofline +actor_rollout_ref.rollout.spec_sd_toggle_config=${SD_TOGGLE_CONFIG} +actor_rollout_ref.rollout.spec_sd_toggle_threshold=${SD_TOGGLE_THRESHOLD} +actor_rollout_ref.rollout.spec_sd_toggle_margin=${SD_TOGGLE_MARGIN}"
    EXPERIMENT_NAME="qwen14b_sd_toggle_roofline"
    ;;
  adaptive)
    echo "=== Qwen2.5-14B EfficientRollout: roofline toggle + adaptive-γ (ladder=${SPEC_GAMMA_LADDER}) ==="
    SD_ARGS="+actor_rollout_ref.rollout.spec_method=quant_self +actor_rollout_ref.rollout.spec_num_draft_tokens=${SPEC_GAMMA_LADDER%%,*} +actor_rollout_ref.rollout.spec_quant_method=rtn +actor_rollout_ref.rollout.spec_sd_toggle_mode=roofline +actor_rollout_ref.rollout.spec_sd_toggle_config=${SD_TOGGLE_CONFIG} +actor_rollout_ref.rollout.spec_sd_toggle_threshold=${SD_TOGGLE_THRESHOLD} +actor_rollout_ref.rollout.spec_sd_toggle_margin=${SD_TOGGLE_MARGIN} +actor_rollout_ref.rollout.spec_gamma_ladder=${SPEC_GAMMA_LADDER}"
    EXPERIMENT_NAME="qwen14b_sd_adaptive"
    ;;
  *)
    echo "Usage: $0 {no-sd|rtn|toggle|adaptive} [hydra overrides...]"
    exit 1
    ;;
esac

# TP=1, optimizer_offload=True (14B needs more memory headroom)
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    trainer.val_before_train=False \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    data.shuffle=True \
    data.seed=42 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-14B \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.max_model_len=10240 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    +actor_rollout_ref.rollout.frequency_penalty=0.05 \
    algorithm.rollout_correction.rollout_is=null \
    algorithm.rollout_correction.rollout_rs=token \
    algorithm.rollout_correction.rollout_rs_threshold=2.0 \
    algorithm.rollout_correction.rollout_rs_threshold_lower=0.5 \
    algorithm.rollout_correction.bypass_mode=false \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=efficientrollout \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.rollout_data_dir=${PROJECT_ROOT}/rollouts/${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=2 \
    ${SD_ARGS} \
    "$@"
