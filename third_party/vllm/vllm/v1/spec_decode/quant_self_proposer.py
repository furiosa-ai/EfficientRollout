"""Quantized Self-Drafter Proposer for self-speculative decoding.

Phase 2e: AWQ-native architecture.
- Drafter initialized with AWQMarlinLinearMethod from the start
- Production compile + CUDA graph path (no do_not_compile bypass)
- Dynamic subclass for independent compilation from target
- Online requantize: target FP16 → RTN W4 → copy_() into AWQ buffers
- KV cache shared via kv_sharing_target_layer_name (zero extra KV memory)

Architecture follows SparseAttnProposer pattern:
- Autoregressive draft loop with build_for_drafting()
- Returns (draft_token_ids, draft_probs) for rejection sampling
"""

from __future__ import annotations

import copy
import logging
import os
import time
from dataclasses import replace
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from vllm.compilation.backends import set_model_tag
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.quantization.awq_marlin import (
    AWQMarlinConfig,
    AWQMarlinLinearMethod,
)
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.spec_decode.online_quantizer import OnlineQuantizer
from vllm.v1.spec_decode.quant_utils import quantize_to_marlin

PADDING_SLOT_ID = -1

logger = logging.getLogger(__name__)


class QuantizedSelfDrafterProposer:
    """Self-speculative proposer using AWQ-native W4 Marlin drafter.

    Phase 2e: Drafter is initialized with AWQMarlinLinearMethod from the start,
    using vLLM's production compile + CUDA graph path. A dynamic subclass gives
    the drafter independent compilation state from the target model.

    Lifecycle:
    1. __init__: Create quantizer, allocate buffers
    2. load_model: Create AWQ-native drafter + KV sharing + requantize
    3. propose: Draft gamma tokens (CUDA graph enabled)
    4. requantize: Refresh W4 weights after veRL weight sync
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        self.runner = runner
        self.device = device
        self.max_model_len = (
            self.speculative_config.draft_model_config.max_model_len
        )

        self.gamma = self.speculative_config.num_speculative_tokens

        quant_method = getattr(self.speculative_config, "quant_method", "rtn")
        group_size = getattr(self.speculative_config, "quant_group_size", 128)

        self.quantizer = OnlineQuantizer(
            method=quant_method,
            group_size=group_size,
            quant_interval=1,
        )

        # AWQ activation-aware quantization state
        self._awq_enabled = quant_method == "awq"
        # No vLLM-side collector — stats come from FSDP pipeline
        self._stale_stats: dict[str, torch.Tensor] = {}  # CPU, CuMem-safe
        self._stale_samples: dict[str, torch.Tensor] = {}  # CPU, [<=128, in]
        self._awq_scales: dict[str, torch.Tensor] = {}  # Computed per requantize
        self._stats_step: int = 0  # Counter for stats-to-disk logging

        self.block_size = self.vllm_config.cache_config.block_size

        max_num_tokens = self.vllm_config.scheduler_config.max_num_seqs
        self.input_ids = torch.zeros(
            max_num_tokens, dtype=torch.int32, device=device
        )
        self.positions = torch.zeros(
            max_num_tokens, dtype=torch.long, device=device
        )
        self.arange = torch.arange(
            max_num_tokens + 1, dtype=torch.int32, device=device
        )
        self.token_arange_np = np.arange(max_num_tokens + 1, dtype=np.int32)

        self.model: Optional[nn.Module] = None
        self.attn_metadata_builder: Optional[AttentionMetadataBuilder] = None
        self.attn_layer_names: list[str] = []
        self._target_model: Optional[nn.Module] = None

        # CUDA graph config
        self.use_cuda_graph = False
        self.cudagraph_capture_sizes: list[int] = []
        compilation_config = self.vllm_config.compilation_config
        if hasattr(compilation_config, 'mode'):
            from vllm.config import CompilationMode
            if compilation_config.mode == CompilationMode.VLLM_COMPILE:
                cudagraph_mode = compilation_config.cudagraph_mode
                self.use_cuda_graph = (
                    cudagraph_mode.has_mode(CUDAGraphMode.PIECEWISE)
                    and not getattr(
                        self.speculative_config, 'enforce_eager', False
                    )
                )
        if self.use_cuda_graph:
            self.cudagraph_capture_sizes = list(
                compilation_config.cudagraph_capture_sizes or []
            )

        logger.info(
            "QuantizedSelfDrafterProposer (Phase 2e AWQ-native): gamma=%d, "
            "method=%s, group_size=%d, cuda_graph=%s",
            self.gamma, quant_method, group_size, self.use_cuda_graph,
        )

    def set_gamma(self, gamma: int) -> None:
        """Update γ at runtime for adaptive-γ SD.

        Called by GPUModelRunner.set_current_gamma() when elevation fires.
        The propose() loop reads self.gamma dynamically, so subsequent
        draft steps use the new γ. Pre-captured CUDA graphs for the new γ
        must already exist (ensured by capture_model iterating over ladder).
        """
        if gamma < 1:
            raise ValueError(f"gamma must be >= 1, got {gamma}")
        self.gamma = gamma
        logger.info("QuantizedSelfDrafterProposer: γ elevated to %d", gamma)

    def load_model(self, target_model: nn.Module) -> None:
        """Create AWQ-native drafter with production compile/CUDA graph path.

        Key change from Phase 2d: model is initialized with AWQMarlinLinearMethod
        from the start (not swapped at runtime). This ensures vLLM's compile/graph
        system sees the same code path as production AWQ serving.

        Steps:
        1. Create AWQMarlinConfig + drafter-specific vllm_config
        2. Initialize model with AWQ linear layers (empty weight shells)
        3. Process weights (set up Marlin buffer structure)
        4. Dynamic subclass for independent compilation
        5. Fix attention backend, copy non-quantized params
        6. Share embeddings, set up KV sharing
        7. Requantize from target FP16 weights
        """
        from vllm.model_executor.model_loader.utils import (
            initialize_model,
            process_weights_after_loading,
        )

        logger.info("Loading AWQ-native quantized self-drafter...")
        self._target_model = target_model

        # Step 1: Create AWQ quant config
        awq_config = AWQMarlinConfig(
            weight_bits=4,
            group_size=self.quantizer.group_size,
            zero_point=True,
            lm_head_quantized=False,
            modules_to_not_convert=[],
            full_config={
                "bits": 4,
                "group_size": self.quantizer.group_size,
                "zero_point": True,
                "quant_method": "awq",
            },
        )

        # Step 2: Create drafter-specific vllm_config with AWQ quantization
        # shallow copy — shares all sub-configs except quant_config
        drafter_vllm_config = copy.copy(self.vllm_config)
        drafter_vllm_config.quant_config = awq_config

        # Step 3: Initialize model with AWQ linear layers (empty weight shells)
        # AWQMarlinLinearMethod.create_weights() creates PackedvLLMParameter shells
        with set_model_tag("quant_self_drafter"):
            self.model = initialize_model(
                vllm_config=drafter_vllm_config,
                prefix="quant_drafter",
            )

        # Step 4: Move to GPU and fix dtypes.
        # Can't use model.to(dtype=bf16) because that would corrupt int32
        # qweight/qzeros. Instead: move everything to GPU, then selectively
        # cast float32 params to model dtype.
        self.model = self.model.to(device=self.device)
        model_dtype = self.vllm_config.model_config.dtype
        for name, param in self.model.named_parameters():
            if param.dtype == torch.float32:
                param.data = param.data.to(dtype=model_dtype)
        for name, buf in self.model.named_buffers():
            if buf is not None and buf.dtype == torch.float32:
                # Skip int32 buffers (workspace, g_idx) — they're created later
                # Only cast actual float32 buffers (e.g., inv_freq)
                pass
        logger.info(
            "Drafter model on device: %s, float params cast to %s",
            self.device, model_dtype,
        )

        # Step 5: Process weights — converts PackedvLLMParameter → Parameter,
        # sets up workspace/g_idx, repacks to Marlin format.
        # Works on uninitialized data (just rearranges bytes).
        model_config = drafter_vllm_config.model_config
        process_weights_after_loading(self.model, model_config, self.device)
        logger.info("Marlin weight buffers initialized")

        # Step 6: Dynamic subclass for independent compilation from target.
        # vLLM's @support_torch_compile uses TorchCompileWithNoGuardsWrapper
        # which shares compiled graphs between instances of the same class.
        # Different class → independent compile state → own CUDA graphs.
        self._create_independent_compile_state()

        # Step 7: Fix attention backend (must match target for KV sharing)
        self._fix_attention_backend(target_model)

        # Step 8: Copy non-quantized params (layernorms, biases, rotary_emb)
        # AWQ model's qweight/qzeros/scales won't match target's weight names → skipped
        self._copy_non_quantized_params(target_model)

        # Step 9: Share embeddings with target (FP16, saves memory)
        self._share_embeddings(target_model)

        # Step 10: KV sharing (BEFORE get_kv_cache_spec runs!)
        self._setup_kv_sharing()

        # Step 11: AWQ activation stats come from FSDP compute_log_prob()
        # hooks (torch.compile-safe). No hooks on vLLM target model —
        # torch.compile breaks forward_pre_hooks on compiled modules.
        if self._awq_enabled:
            logger.info(
                "AWQ enabled — activation stats from FSDP "
                "compute_log_prob() hooks (no vLLM-side hooks)"
            )

        # Step 12: Requantize from target FP16 weights (initial, replaces empty buffers)
        elapsed = self._requantize_from_target(initial=True)
        logger.info("Initial requantization from target: %.2fs", elapsed)

        logger.info(
            "AWQ-native quantized self-drafter loaded "
            "(compile+graph=%s, awq=%s)",
            self.use_cuda_graph, self._awq_enabled,
        )

    def _create_independent_compile_state(self) -> None:
        """Create dynamic subclass so drafter compiles independently from target.

        vLLM's @support_torch_compile with TorchCompileWithNoGuardsWrapper
        shares compiled graphs between instances of the same class. Since
        target (unquantized) and drafter (AWQ) have different forward ops,
        sharing compiled state would cause errors.

        Fix: Change drafter inner model's class to a dynamic subclass.
        Different class → different function identity → independent compilation.

        This approach failed in Phase 2c because OnlineMarlinQuantMethod used
        register_buffer (compile-incompatible). With AWQ-native approach,
        AWQMarlinLinearMethod is production code → compile-compatible.
        """
        inner = getattr(self.model, 'model', None)
        if inner is None:
            logger.warning("No inner model found, skipping subclass creation")
            return

        original_cls = type(inner)
        drafter_cls = type(
            f"Drafter{original_cls.__name__}",
            (original_cls,),
            {},
        )
        inner.__class__ = drafter_cls
        logger.info(
            "Drafter inner model class: %s (independent from target %s)",
            drafter_cls.__name__,
            original_cls.__name__,
        )

    def _share_embeddings(self, target_model: nn.Module) -> None:
        """Share embed_tokens and lm_head with target (FP16)."""
        target_inner = _get_inner_model(target_model)
        drafter_inner = _get_inner_model(self.model)

        if hasattr(drafter_inner, "embed_tokens"):
            del drafter_inner.embed_tokens
            drafter_inner.embed_tokens = target_inner.embed_tokens
            logger.info("Shared embed_tokens with target")

        if hasattr(self.model, "lm_head") and hasattr(target_model, "lm_head"):
            del self.model.lm_head
            self.model.lm_head = target_model.lm_head
            logger.info("Shared lm_head with target")

    def _fix_attention_backend(self, target_model: nn.Module) -> None:
        """Copy attention backend/impl from target to drafter.

        Drafter and target must use the same attention backend because they
        share KV cache (same layout required).
        """
        from vllm.attention.layer import Attention as AttentionLayer

        target_attns = sorted(
            [(n, m) for n, m in target_model.named_modules()
             if isinstance(m, AttentionLayer)],
            key=lambda x: x[0]
        )
        drafter_attns = sorted(
            [(n, m) for n, m in self.model.named_modules()
             if isinstance(m, AttentionLayer)],
            key=lambda x: x[0]
        )

        assert len(target_attns) == len(drafter_attns), (
            f"Target has {len(target_attns)} attn layers, "
            f"drafter has {len(drafter_attns)}"
        )

        fixed = 0
        for (t_name, t_attn), (d_name, d_attn) in zip(
            target_attns, drafter_attns
        ):
            if type(d_attn.impl) != type(t_attn.impl):
                d_attn.impl = t_attn.impl
                d_attn.attn_backend = t_attn.attn_backend
                d_attn.use_output = t_attn.use_output
                d_attn.use_direct_call = t_attn.use_direct_call
                d_attn.backend = t_attn.backend
                fixed += 1

        if fixed > 0:
            logger.info(
                "Fixed %d drafter attn layers → %s",
                fixed,
                type(target_attns[0][1].impl).__name__,
            )

    def _copy_non_quantized_params(self, target_model: nn.Module) -> None:
        """Copy non-quantized parameters from target to drafter.

        Copies layernorms, biases, rotary_emb buffers, etc.
        AWQ model's qweight/qzeros/scales names don't exist in target → skipped.

        IMPORTANT: After copying, biases on AWQ linear layers must be permuted
        via marlin_permute_bias(). The gptq_marlin_gemm kernel expects permuted
        bias (it adds bias internally in permuted column order).

        Safe to call repeatedly (from requantize): target biases are always
        unpermuted, so copy → permute is correct every time. Uses copy_() for
        CUDA graph address stability.
        """
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_permute_bias,
        )

        target_params = dict(target_model.named_parameters())
        target_buffers = dict(target_model.named_buffers())
        copied = 0

        for name, param in self.model.named_parameters():
            if name in target_params:
                try:
                    param.data.copy_(target_params[name].data)
                    copied += 1
                except RuntimeError as e:
                    if "layernorm" in name.lower():
                        logger.warning(
                            "LayerNorm copy FAILED for %s: %s — "
                            "AWQ scale absorption may drift", name, e)
                    elif "size mismatch" not in str(e) and "dtype" not in str(e):
                        logger.debug("Unexpected copy error for %s: %s", name, e)

        for name, buf in self.model.named_buffers():
            if name in target_buffers and not isinstance(buf, nn.Parameter):
                try:
                    buf.copy_(target_buffers[name])
                    copied += 1
                except RuntimeError as e:
                    if "size mismatch" not in str(e) and "dtype" not in str(e):
                        logger.debug("Unexpected copy error for %s: %s", name, e)

        # Permute biases on AWQ linear layers for gptq_marlin_gemm kernel.
        # The kernel adds bias internally in permuted column order.
        # Target biases are always unpermuted (FP16 model), so copy brings
        # fresh unpermuted values → permute is correct on every call.
        # Use copy_() to preserve tensor addresses for CUDA graph safety.
        bias_permuted = 0
        for name, module in self.model.named_modules():
            if (hasattr(module, 'quant_method')
                    and isinstance(module.quant_method, AWQMarlinLinearMethod)
                    and hasattr(module, 'bias') and module.bias is not None):
                permuted = marlin_permute_bias(module.bias.data)
                module.bias.data.copy_(permuted)
                bias_permuted += 1

        logger.info(
            "Copied %d non-quantized params/buffers from target, "
            "permuted %d biases for Marlin kernel",
            copied, bias_permuted,
        )

    def _setup_kv_sharing(self) -> None:
        """Set kv_sharing_target_layer_name on drafter Attention layers.

        MUST run during load_model() — before get_kv_cache_spec().
        """
        from vllm.attention.layer import Attention as AttentionLayer

        compilation_config = self.vllm_config.compilation_config
        drafter_attn_names = []
        sharing_count = 0

        for name, module in self.model.named_modules():
            if isinstance(module, AttentionLayer):
                target_layer_name = name
                if target_layer_name not in compilation_config.static_forward_context:
                    logger.warning(
                        "Target layer %s not in static_forward_context, "
                        "skipping KV sharing",
                        target_layer_name,
                    )
                    continue

                module.kv_sharing_target_layer_name = target_layer_name
                drafter_registered_name = f"quant_drafter.{name}"
                drafter_attn_names.append(drafter_registered_name)
                sharing_count += 1

        self.attn_layer_names = drafter_attn_names
        logger.info(
            "KV sharing configured: %d drafter layers → target layers",
            sharing_count,
        )

    def _resolve_awq_scales(self, weight_key: str) -> torch.Tensor | None:
        """Resolve AWQ scales for a drafter layer (vLLM fused names).

        Maps vLLM fused names to FSDP unfused names stored in _awq_scales:
        - qkv_proj → q_proj scales (shared by q, k, v)
        - gate_up_proj → gate_proj scales (shared by gate, up)
        - down_proj → direct lookup
        - o_proj → RTN (GQA dimension mismatch)
        """
        # o_proj: no absorption mechanism (GQA prevents FC-FC, no preceding LN)
        # Must use RTN to avoid uncompensated W*s scaling.
        if "o_proj" in weight_key:
            return None

        # Direct lookup (works for down_proj which is not fused)
        if weight_key in self._awq_scales:
            return self._awq_scales[weight_key]

        # Fused → unfused mapping
        layer_prefix = weight_key.rsplit(".", 2)[0]  # e.g. "model.layers.0.self_attn"

        if "qkv_proj" in weight_key:
            # qkv_proj uses q_proj's AWQ scales (shared input from LN)
            q_key = f"{layer_prefix}.q_proj.weight"
            return self._awq_scales.get(q_key)

        if "gate_up_proj" in weight_key:
            # gate_up_proj uses gate_proj's AWQ scales (shared input from LN)
            gate_key = f"{layer_prefix}.gate_proj.weight"
            return self._awq_scales.get(gate_key)

        return None  # o_proj, others → RTN

    def _absorb_awq_scales_into_layernorms(self) -> None:
        """Absorb AWQ per-channel scales into preceding LayerNorm weights.

        For layers with a preceding LayerNorm (q_proj <- input_layernorm,
        gate_proj <- post_attention_layernorm), divide LN weight by AWQ scales.
        This makes inference mathematically equivalent:
          (LN_out / s) @ dequant(quant(W * s)).T ≈ LN_out @ W.T

        k_proj/v_proj share input_layernorm with q_proj → same scales applied.
        up_proj shares post_attention_layernorm with gate_proj → same scales.
        down_proj uses FC-FC absorption (handled in _requantize_from_target).
        o_proj has no absorption → RTN fallback.
        """
        if not self._awq_scales:
            return

        absorbed = 0
        for name, module in self.model.named_modules():
            if name.endswith('input_layernorm') and hasattr(module, 'weight'):
                layer_prefix = name.rsplit('.input_layernorm', 1)[0]
                q_proj_key = f"{layer_prefix}.self_attn.q_proj.weight"
                if q_proj_key in self._awq_scales:
                    scales = self._awq_scales[q_proj_key].to(
                        module.weight.device
                    )
                    module.weight.data.div_(scales)
                    absorbed += 1

            elif (name.endswith('post_attention_layernorm')
                  and hasattr(module, 'weight')):
                layer_prefix = name.rsplit('.post_attention_layernorm', 1)[0]
                gate_proj_key = f"{layer_prefix}.mlp.gate_proj.weight"
                if gate_proj_key in self._awq_scales:
                    scales = self._awq_scales[gate_proj_key].to(
                        module.weight.device
                    )
                    module.weight.data.div_(scales)
                    absorbed += 1

        if absorbed > 0:
            logger.info(
                "Absorbed AWQ scales into %d LayerNorm weights", absorbed
            )

    def _requantize_from_target(self, initial: bool = False) -> float:
        """Requantize all AWQ linear layers from target FP16 weights.

        Uses copy_() into existing CuMem-tagged buffers to preserve tensor
        addresses. This maintains CuMem tag integrity (sleep/wake cycles) and
        CUDA graph compatibility (graphs capture fixed addresses).
        """
        assert self._target_model is not None

        # Ensure all GPU operations complete before reading weights.
        # veRL FSDP weight sync may use async CUDA copies.
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        target_params = dict(self._target_model.named_parameters())
        count = 0

        # Pre-compute FC-FC absorption: down_proj scales → applied to up
        # portion of fused gate_up_proj. Clone target weight to avoid
        # in-place modification of the target model.
        fc_fc_modified_weights: dict[str, torch.Tensor] = {}
        if self._awq_enabled and self._awq_scales:
            for mod_name, module in self.model.named_modules():
                if not mod_name.endswith("mlp.down_proj"):
                    continue
                down_key = f"{mod_name}.weight"
                s_down = self._awq_scales.get(down_key)
                if s_down is None:
                    continue

                # Find the corresponding gate_up_proj
                layer_prefix = mod_name.rsplit(".mlp.down_proj", 1)[0]
                gate_up_key = f"{layer_prefix}.mlp.gate_up_proj.weight"
                if gate_up_key not in target_params:
                    continue

                # Clone and modify only the up portion (last intermediate rows)
                gate_up_weight = target_params[gate_up_key].data.clone()
                intermediate_size = s_down.shape[0]
                gate_up_weight[-intermediate_size:].div_(
                    s_down.to(gate_up_weight.device).unsqueeze(1)
                )
                fc_fc_modified_weights[gate_up_key] = gate_up_weight
                logger.debug(
                    "FC-FC absorption: %s scales → %s up portion (%d rows)",
                    down_key, gate_up_key, intermediate_size,
                )

        for name, module in self.model.named_modules():
            if not hasattr(module, 'quant_method'):
                continue
            if not isinstance(module.quant_method, AWQMarlinLinearMethod):
                continue

            weight_key = f"{name}.weight"
            if weight_key not in target_params:
                continue

            # Use pre-modified weight if available (FC-FC absorption)
            if weight_key in fc_fc_modified_weights:
                fp16_weight = fc_fc_modified_weights[weight_key]
            else:
                fp16_weight = target_params[weight_key].data

            # Resolve AWQ scales: handles shared inputs (k/v→q, up→gate)
            # and returns None for o_proj (RTN fallback).
            layer_scales = (
                self._resolve_awq_scales(weight_key)
                if self._awq_enabled else None
            )
            marlin_qw, marlin_s, marlin_zp = quantize_to_marlin(
                fp16_weight,
                group_size=self.quantizer.group_size,
                scales_override=layer_scales,
            )

            # Marlin kernel requires scales dtype to match input dtype.
            model_dtype = self.vllm_config.model_config.dtype
            if marlin_s.dtype != model_dtype:
                marlin_s = marlin_s.to(model_dtype)

            # copy_() into existing CuMem-tagged buffers. Safe because:
            # 1. Initial load runs inside use_memory_pool(tag='weights')
            #    (gpu_worker.py:270-273 → model_runner.load_model → drafter.load_model)
            # 2. CuMem sleep preserves "weights"-tagged tensor addresses
            # 3. CuMem wake restores memory at same addresses
            # 4. copy_() overwrites with fresh quantized values
            # 5. CUDA graphs remain valid (addresses unchanged)
            module.qweight.data.copy_(marlin_qw)
            module.scales.data.copy_(marlin_s)
            module.qzeros.data.copy_(marlin_zp)

            # Bias handling is done by _copy_non_quantized_params() which
            # always runs before this method (copy + marlin_permute_bias).

            count += 1

        elapsed = time.perf_counter() - t0
        awq_count = sum(1 for k in self._awq_scales if k in target_params)
        logger.info(
            "Requantized %d layers (awq=%d, rtn=%d) in %.2fs",
            count, awq_count, count - awq_count, elapsed,
        )

        return elapsed

    def _get_attention_metadata_builder(self) -> AttentionMetadataBuilder:
        """Find attention metadata builder for drafter layers."""
        builder = None
        if self.attn_layer_names:
            chosen_layer = self.attn_layer_names[0]
            for kv_cache_group in self.runner.attn_groups:
                for attn_group in kv_cache_group:
                    if chosen_layer in attn_group.layer_names:
                        builder = attn_group.get_metadata_builder()
                        break
                if builder is not None:
                    break
        assert builder is not None, (
            "Failed to find attention metadata builder for quantized drafter. "
            f"Looking for: "
            f"{self.attn_layer_names[0] if self.attn_layer_names else 'none'}"
        )
        return builder

    @torch.inference_mode()
    def dummy_run(self, num_tokens: int, use_cudagraphs: bool = True) -> None:
        """Warmup for CUDA graph capture (EAGLE pattern)."""
        cudagraphs_enabled = use_cudagraphs and self.use_cuda_graph
        if cudagraphs_enabled and self.cudagraph_capture_sizes:
            if num_tokens <= self.cudagraph_capture_sizes[-1]:
                num_tokens = self.vllm_config.pad_for_cudagraph(num_tokens)

        with set_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=(
                CUDAGraphMode.PIECEWISE if cudagraphs_enabled
                else CUDAGraphMode.NONE
            ),
        ):
            self.model(
                input_ids=self.input_ids[:num_tokens],
                positions=self.positions[:num_tokens],
                intermediate_tensors=None,
            )

    def propose(
        self,
        target_token_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Draft gamma tokens using AWQ-native quantized model."""
        del target_token_ids, hidden_states

        assert self.model is not None
        if self.attn_metadata_builder is None:
            self.attn_metadata_builder = self._get_attention_metadata_builder()

        batch_size = next_token_ids.shape[0]
        draft_positions = (
            common_attn_metadata.seq_lens[:batch_size].long() - 1
        )

        common_attn_metadata.num_actual_tokens = batch_size
        common_attn_metadata.max_query_len = 1
        common_attn_metadata.query_start_loc = self.arange[:batch_size + 1]
        common_attn_metadata.query_start_loc_cpu = torch.from_numpy(
            self.token_arange_np[:batch_size + 1]
        ).clone()

        self.input_ids[:batch_size] = next_token_ids.int()

        draft_token_ids_list: list[torch.Tensor] = []
        draft_probs_list: list[torch.Tensor] = []
        local_spec_token_ids: list[list[int]] = [
            [] for _ in range(batch_size)
        ]

        # ── SD timing instrumentation ──
        _sd_timing = os.environ.get("VLLM_SD_TIMING", "0") == "1"
        if _sd_timing:
            _draft_total_start = torch.cuda.Event(enable_timing=True)
            _draft_total_end = torch.cuda.Event(enable_timing=True)
            _draft_fwd_events: list[tuple] = []  # (start, end) per step
            _draft_total_start.record()

        for draft_index in range(self.gamma):
            draft_positions = draft_positions + 1
            exceeds = draft_positions >= self.max_model_len
            clamped = torch.where(
                exceeds,
                torch.zeros_like(draft_positions),
                draft_positions,
            )

            common_attn_metadata.seq_lens += 1
            common_attn_metadata.seq_lens_cpu = (
                common_attn_metadata.seq_lens_cpu + 1
            )
            common_attn_metadata.seq_lens.masked_fill_(exceeds, 1)
            common_attn_metadata.num_computed_tokens_cpu = (
                common_attn_metadata.seq_lens_cpu - 1
            )

            block_numbers = clamped // self.block_size
            block_ids = (
                common_attn_metadata.block_table_tensor.gather(
                    dim=1, index=block_numbers.view(-1, 1)
                ).view(-1)
            )
            common_attn_metadata.slot_mapping = (
                block_ids * self.block_size + clamped % self.block_size
            )
            common_attn_metadata.slot_mapping.masked_fill_(
                exceeds, PADDING_SLOT_ID
            )

            draft_attn_meta = (
                self.attn_metadata_builder.build_for_drafting(
                    common_attn_metadata=common_attn_metadata,
                    draft_index=draft_index,
                )
            )
            per_layer_attn_metadata = {
                n: draft_attn_meta for n in self.attn_layer_names
            }

            self.positions[:batch_size] = clamped

            cudagraph_mode = CUDAGraphMode.NONE
            num_input = batch_size
            if self.use_cuda_graph and self.cudagraph_capture_sizes:
                if batch_size <= self.cudagraph_capture_sizes[-1]:
                    num_input = self.vllm_config.pad_for_cudagraph(batch_size)
                    cudagraph_mode = CUDAGraphMode.PIECEWISE

            if _sd_timing:
                _fwd_s = torch.cuda.Event(enable_timing=True)
                _fwd_e = torch.cuda.Event(enable_timing=True)
                _fwd_s.record()

            with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=num_input,
                cudagraph_runtime_mode=cudagraph_mode,
            ):
                model_hidden = self.model(
                    input_ids=self.input_ids[:num_input],
                    positions=self.positions[:num_input],
                    intermediate_tensors=None,
                )

            if _sd_timing:
                _fwd_e.record()
                _draft_fwd_events.append((_fwd_s, _fwd_e))

            logits = self.model.compute_logits(model_hidden[:batch_size])

            processed_logits = self._process_draft_logits(
                logits, sampling_metadata, local_spec_token_ids
            )

            draft_probs = processed_logits.softmax(
                dim=-1, dtype=torch.float32
            )

            if sampling_metadata.all_greedy:
                draft_token = processed_logits.argmax(dim=-1)
            else:
                draft_token = torch.multinomial(
                    draft_probs, num_samples=1
                ).squeeze(-1)

            draft_token_ids_list.append(draft_token)
            draft_probs_list.append(draft_probs)

            for req_idx, token_id in enumerate(draft_token.tolist()):
                local_spec_token_ids[req_idx].append(token_id)

            self.input_ids[:batch_size] = draft_token.int()

        # ── SD timing: log T_D (forward-only and full-step) ──
        # NOTE: no synchronize here — events are read later after cycle end sync
        if _sd_timing:
            _draft_total_end.record()
            # Defer elapsed_time reads to avoid flushing GPU pipeline
            # Store events for later consumption
            self._sd_draft_events = {
                "total_start": _draft_total_start,
                "total_end": _draft_total_end,
                "fwd_events": _draft_fwd_events,
                "batch_size": batch_size,
                "gamma": self.gamma,
                "avg_seq": float(common_attn_metadata.seq_lens_cpu.float().mean()),
            }
            # Print is deferred — events will be read in sample_tokens cycle end

        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        draft_probs = torch.stack(draft_probs_list, dim=1).reshape(
            batch_size * self.gamma, -1
        )

        return draft_token_ids, draft_probs

    def _process_draft_logits(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        local_spec_token_ids: list[list[int]],
    ) -> torch.Tensor:
        """Process draft logits with temperature/top-k/top-p."""
        assert self.runner is not None

        base_output_token_ids = list(
            sampling_metadata.output_token_ids[:len(local_spec_token_ids)]
        )
        if len(base_output_token_ids) < len(local_spec_token_ids):
            base_output_token_ids.extend(
                [
                    []
                    for _ in range(
                        len(local_spec_token_ids) - len(base_output_token_ids)
                    )
                ]
            )
        step_output_token_ids = [
            [*base_output, *local_spec]
            for base_output, local_spec in zip(
                base_output_token_ids, local_spec_token_ids
            )
        ]
        step_sampling_metadata = replace(
            sampling_metadata,
            output_token_ids=step_output_token_ids,
            spec_token_ids=None,
        )

        processed_logits = logits.to(torch.float32)
        processed_logits = self.runner.sampler.apply_logits_processors(
            processed_logits,
            step_sampling_metadata,
            predict_bonus_token=False,
        )
        if step_sampling_metadata.all_greedy:
            return processed_logits

        processed_logits = self.runner.sampler.apply_temperature(
            processed_logits,
            step_sampling_metadata.temperature,
            step_sampling_metadata.all_random,
        )
        for processor in step_sampling_metadata.logitsprocs.argmax_invariant:
            processed_logits = processor.apply(processed_logits)
        return apply_top_k_top_p(
            processed_logits,
            step_sampling_metadata.top_k,
            step_sampling_metadata.top_p,
        )

    def requantize(
        self, target_model: nn.Module, fsdp_activation_stats=None,
        fsdp_activation_samples=None,
    ) -> Optional[float]:
        """Requantize from updated target model weights.

        Called after veRL weight sync. Copies all non-quantized params
        (layernorms, biases) and requantizes all AWQ layers every time.

        If AWQ is enabled and activation stats are available from the
        previous rollout, computes per-layer AWQ scales via beta search
        before requantizing. Scale absorption into LayerNorms is applied
        after quantization.

        Args:
            target_model: The target FP16 model to requantize from.
            fsdp_activation_stats: Optional dict of per-channel activation
                magnitudes collected from FSDP training phase. When provided,
                takes priority over vLLM-side collector stats.
            fsdp_activation_samples: Optional dict of activation samples
                [<=128, in_features] on CPU for block output MSE search.
        """
        self._target_model = target_model

        # --- Measurement: weight delta per layer ---
        _measure = os.environ.get("SD_MEASURE_WEIGHT_DELTA", "")
        if _measure:
            self._measure_weight_delta(target_model)

        # --- Activation stats from FSDP compute_log_prob() pipeline ---
        if fsdp_activation_stats is not None:
            # Normalize FSDP layer names: strip _fsdp_wrapped_module segments
            # that FSDP v1 auto_wrap_policy inserts into named_modules() paths.
            # vLLM target model uses clean names (no FSDP wrappers).
            normalized = {
                k.replace("._fsdp_wrapped_module", ""): v
                for k, v in fsdp_activation_stats.items()
            }
            self._stale_stats = normalized  # CPU tensors
            logger.info(
                "Using FSDP activation stats: %d layers (normalized from %d)",
                len(normalized), len(fsdp_activation_stats),
            )
            self._save_stats_to_disk(normalized)
        elif self._awq_enabled:
            logger.warning(
                "AWQ enabled but no activation stats available — "
                "using RTN fallback (first step or FSDP stats not flowing)"
            )

        # --- Activation samples from FSDP pipeline ---
        if fsdp_activation_samples is not None:
            normalized_samples = {
                k.replace("._fsdp_wrapped_module", ""): v
                for k, v in fsdp_activation_samples.items()
            }
            self._stale_samples = normalized_samples
            logger.info(
                "Using FSDP activation samples: %d layers", len(normalized_samples),
            )
        else:
            self._stale_samples = {}

        # Always copy non-quantized params (layernorm, bias, rotary).
        # Must happen BEFORE scale absorption — gives us fresh LN weights.
        t0 = time.perf_counter()
        self._copy_non_quantized_params(target_model)
        t1 = time.perf_counter()

        # --- AWQ beta search (if enabled and stats available) ---
        self._awq_scales = {}
        if self._awq_enabled and self._stale_stats:
            from vllm.v1.spec_decode.awq_search import (
                compute_awq_scales_all_layers,
            )
            self._awq_scales = compute_awq_scales_all_layers(
                target_model,
                self._stale_stats,
                stale_samples=self._stale_samples or None,
                group_size=self.quantizer.group_size,
                num_candidates=20,
            )
        t2 = time.perf_counter()

        elapsed = self._requantize_from_target(initial=False)

        # Absorb AWQ scales into LayerNorm weights AFTER quantization.
        # Order: copy fresh LN → quantize W*s → LN /= s
        self._absorb_awq_scales_into_layernorms()
        t3 = time.perf_counter()
        total_sec = t3 - t0
        logger.info(
            "Requantize breakdown: copy_params=%.3fs, awq_search=%.3fs, "
            "quantize+absorb=%.3fs, total=%.3fs",
            t1 - t0, t2 - t1, t3 - t2, total_sec,
        )
        self._last_requantize_sec = total_sec
        return total_sec

    def _save_stats_to_disk(self, stats: dict[str, torch.Tensor]) -> None:
        """Save activation stats to disk for stability analysis."""
        stats_log_dir = os.environ.get("SD_ACT_STATS_LOG_DIR")
        if not stats_log_dir:
            return
        os.makedirs(stats_log_dir, exist_ok=True)
        step = getattr(self, '_stats_step', 0)
        self._stats_step = step + 1
        try:
            torch.save(stats, f"{stats_log_dir}/stats_step_{step:04d}.pt")
        except OSError as e:
            logger.warning("Failed to save stats: %s", e)

    def _measure_weight_delta(self, target_model: nn.Module) -> None:
        """Measurement hook: record per-layer weight change between steps."""
        _measure = os.environ.get("SD_MEASURE_WEIGHT_DELTA", "")
        if not _measure:
            if hasattr(self, '_prev_weights'):
                del self._prev_weights
            return

        if not hasattr(self, '_prev_weights'):
            self._prev_weights = {}

        step = getattr(self, '_measure_step', 0)
        self._measure_step = step + 1
        diag_path = _measure
        lines = []

        for name, param in target_model.named_parameters():
            if 'weight' not in name or param.dim() != 2:
                continue
            current_cpu = param.data.float().cpu()
            if name in self._prev_weights:
                diff = current_cpu - self._prev_weights[name]
                prev_norm = self._prev_weights[name].norm().item()
                rel_change = diff.norm().item() / max(prev_norm, 1e-8)
                max_change = diff.abs().max().item()
                lines.append(
                    f"step={step} {name}: rel={rel_change:.6f}, "
                    f"max={max_change:.6f}, norm={prev_norm:.2f}\n"
                )
            self._prev_weights[name] = current_cpu.clone()

        if lines:
            try:
                with open(diag_path, "a") as f:
                    f.writelines(lines)
            except OSError as e:
                logger.warning("Failed to write weight delta to %s: %s", diag_path, e)


def _get_inner_model(model: nn.Module) -> nn.Module:
    """Get the inner transformer model (handles different model wrappers)."""
    if hasattr(model, "model"):
        return model.model
    return model
