# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SparseAttnProposer: self-speculative decoding via sparse attention.

Two guidance modes:
- draft_query: legacy step-0 draft query collection.
- verify_collect2_query: verification-guided collect-2-query (first+bonus).
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from vllm.config import (
    CompilationMode,
    CUDAGraphMode,
    VllmConfig,
    get_layers_from_vllm_config,
)
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)

PADDING_SLOT_ID = -1


class SparseAttnProposer:
    """Self-speculative proposer: same model weights, sparse attention."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None

        self.runner = runner
        self.device = device
        self.max_model_len = vllm_config.model_config.max_model_len
        self.block_size = vllm_config.cache_config.block_size

        self.gamma = self.speculative_config.num_speculative_tokens

        sparse_window = self.speculative_config.sparse_window_size
        assert sparse_window is not None and sparse_window > 0
        self.sparse_window_size: int = sparse_window
        # Use (-1, -1) for full attention when window covers entire context.
        # FlashAttention's sliding window kernel can produce incorrect results
        # for very large window values, so we avoid it when unnecessary.
        if self.sparse_window_size >= self.max_model_len:
            self.draft_window_size: tuple[int, int] = (-1, -1)
        else:
            self.draft_window_size: tuple[int, int] = (
                self.sparse_window_size - 1, 0)

        # Block-level sparse attention config.
        self.topk_budget: int = getattr(self.speculative_config, "topk_budget", 0)
        self.guidance_mode: str = getattr(
            self.speculative_config, "guidance_mode", "verify_collect2_query"
        )
        self.collect2q_enabled: bool = bool(
            getattr(self.speculative_config, "collect2q_enabled", True)
        )
        self.topk_update_stride: int = int(
            getattr(self.speculative_config, "topk_update_stride", 1)
        )
        self.topk_candidate_cap_blocks: int = int(
            getattr(self.speculative_config, "topk_candidate_cap_blocks", 0)
        )
        self.guidance_layer_mode: str = str(
            getattr(self.speculative_config, "guidance_layer_mode", "last_n")
        )
        self.guidance_num_layers: int = int(
            getattr(self.speculative_config, "guidance_num_layers", 4)
        )

        # Number of recent blocks to always include (covers window + draft).
        self.num_recent_blocks: int = (
            (self.sparse_window_size + self.gamma + self.block_size - 1)
            // self.block_size
        )

        # CUDA graph config.
        self.use_cuda_graph = False
        compilation_config = vllm_config.compilation_config
        if compilation_config.mode == CompilationMode.VLLM_COMPILE:
            cudagraph_mode = compilation_config.cudagraph_mode
            if (cudagraph_mode != CUDAGraphMode.NONE
                    and not cudagraph_mode.has_mode(
                        CUDAGraphMode.PIECEWISE)):
                logger.warning(
                    "SparseAttnProposer only supports PIECEWISE cudagraph.")
            self.use_cuda_graph = (
                cudagraph_mode.has_mode(CUDAGraphMode.PIECEWISE)
                and not self.speculative_config.enforce_eager)
        self.cudagraph_batch_sizes = (
            sorted(compilation_config.cudagraph_capture_sizes)
            if self.use_cuda_graph else [])
        self.use_cuda_graph = (
            self.use_cuda_graph and bool(self.cudagraph_batch_sizes))

        self.max_num_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens)
        max_batch_size = vllm_config.scheduler_config.max_num_seqs

        self.token_arange_np = np.arange(max_batch_size + 1)
        self.input_ids = torch.zeros(
            self.max_num_tokens, dtype=torch.int32, device=device)
        self.positions = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device)
        self.arange = torch.arange(
            max_batch_size + 1, device=device, dtype=torch.int32)

        # KV cache refs for block scoring (set lazily, keyed by layer name).
        self._kv_cache_by_layer: dict[str, torch.Tensor] = {}

        # verify-guided state.
        self._verify_guidance: dict[str, object] | None = None
        self._verify_update_counter: int = 0

        # Set in load_model().
        self.model: nn.Module | None = None
        self.attn_metadata_builder: AttentionMetadataBuilder | None = None
        self.attn_layer_names: list[str] = []
        self.guidance_layer_names: list[str] = []
        self.guidance_source_by_layer: dict[str, str] = {}

    def load_model(self, target_model: nn.Module) -> None:
        """Reuse target model directly (zero extra GPU memory)."""
        self.model = self.runner.model
        target_attn_layers = get_layers_from_vllm_config(
            self.vllm_config, AttentionLayerBase)
        self.attn_layer_names = list(target_attn_layers.keys())
        assert self.attn_layer_names
        self.guidance_layer_names = self._select_guidance_layers(
            self.attn_layer_names,
            mode=self.guidance_layer_mode,
            num_layers=self.guidance_num_layers,
        )
        self.guidance_source_by_layer = self._build_guidance_source_map(
            self.attn_layer_names, self.guidance_layer_names)
        logger.info(
            "SparseAttnProposer: %d attn layers, window=%d, gamma=%d, "
            "topk_blocks=%d, recent_blocks=%d, guidance_mode=%s, "
            "guidance_layers=%d/%d (%s)",
            len(self.attn_layer_names), self.sparse_window_size, self.gamma,
            self.topk_budget, self.num_recent_blocks, self.guidance_mode,
            len(self.guidance_layer_names), len(self.attn_layer_names),
            self.guidance_layer_mode)

    @staticmethod
    def _select_guidance_layers(layer_names: list[str],
                                mode: str = "last_n",
                                num_layers: int = 4) -> list[str]:
        if not layer_names:
            return []
        if mode == "all":
            return list(layer_names)
        count = min(max(1, num_layers), len(layer_names))
        if mode == "last_n":
            return list(layer_names[-count:])
        if mode == "evenly_spaced":
            if count == 1:
                return [layer_names[-1]]
            indices = np.linspace(0, len(layer_names) - 1, num=count, dtype=int)
            dedup_indices = sorted(set(int(idx) for idx in indices))
            return [layer_names[idx] for idx in dedup_indices]
        logger.warning("Unknown guidance_layer_mode=%s, falling back to last_n", mode)
        return list(layer_names[-count:])

    def _build_guidance_source_map(
        self,
        all_layer_names: list[str],
        guidance_layer_names: list[str],
    ) -> dict[str, str]:
        if not guidance_layer_names:
            return {}
        guidance_positions = {
            layer_name: idx for idx, layer_name in enumerate(all_layer_names)
            if layer_name in set(guidance_layer_names)
        }
        mapping: dict[str, str] = {}
        for idx, layer_name in enumerate(all_layer_names):
            best_layer = min(
                guidance_layer_names,
                key=lambda candidate: abs(guidance_positions[candidate] - idx),
            )
            mapping[layer_name] = best_layer
        return mapping

    def _get_attention_metadata_builder(self) -> AttentionMetadataBuilder:
        chosen_layer = self.attn_layer_names[0]
        for kv_cache_group in self.runner.attn_groups:
            for attn_group in kv_cache_group:
                if chosen_layer in attn_group.layer_names:
                    return attn_group.get_metadata_builder()
        raise RuntimeError(
            f"No metadata builder for layer '{chosen_layer}'")

    def _init_kv_cache_refs(self) -> None:
        """Cache references to each attention layer's KV cache tensor."""
        if self._kv_cache_by_layer:
            return
        assert self.model is not None
        for _, module in self.model.named_modules():
            if (hasattr(module, "kv_cache")
                    and hasattr(module, "layer_name")
                    and module.layer_name in self.attn_layer_names):
                self._kv_cache_by_layer[module.layer_name] = module.kv_cache[0]
        missing_layers = [
            layer_name for layer_name in self.attn_layer_names
            if layer_name not in self._kv_cache_by_layer
        ]
        if missing_layers:
            logger.warning(
                "SparseAttnProposer: missing KV cache refs for %d layers; "
                "block-level sparse disabled for those layers: %s",
                len(missing_layers),
                missing_layers[:4],
            )

    def _build_sparse_override(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        batch_size: int,
        query_mask: torch.Tensor | None = None,
    ) -> dict[str, object] | None:
        """Score blocks and build sparse block_table override."""
        from vllm.v1.spec_decode.kv_selector import (
            build_sparse_block_table,
            score_blocks,
        )

        seq_lens = common_attn_metadata.seq_lens[:batch_size]
        block_table = common_attn_metadata.block_table_tensor[:batch_size]

        # Sparse override is batch-shaped. If even one request lacks valid
        # guidance, we conservatively fall back instead of synthesizing
        # per-row fake top-k slots that can duplicate KV blocks.
        if query_mask is not None and not bool(query_mask.all().item()):
            return None

        min_blocks_for_sparse = self.num_recent_blocks + self.topk_budget
        num_blocks_per_req = (seq_lens + self.block_size - 1) // self.block_size
        if num_blocks_per_req.max().item() <= min_blocks_for_sparse:
            return None

        try:
            block_scores = score_blocks(
                query=query,
                kv_cache=kv_cache,
                block_table=block_table,
                seq_lens=seq_lens,
                block_size=self.block_size,
                query_mask=query_mask,
            )

            (
                sparse_bt,
                sparse_sl,
                sparse_capacity_tokens,
                sparse_max_seqlen_k,
            ) = build_sparse_block_table(
                block_table=block_table,
                block_scores=block_scores,
                seq_lens=seq_lens,
                num_topk_blocks=self.topk_budget,
                num_recent_blocks=self.num_recent_blocks,
                block_size=self.block_size,
                candidate_cap_blocks=self.topk_candidate_cap_blocks,
                query_mask=query_mask,
            )

            return {
                "block_table": sparse_bt,
                "seq_lens": sparse_sl,
                "sparse_capacity_tokens": sparse_capacity_tokens,
                "max_seqlen_k": sparse_max_seqlen_k,
            }

        except Exception as e:
            logger.warning("Block scoring failed (%s), using sliding window", e)
            return None

    def _runtime_sparse_override(
        self,
        sparse_base: dict[str, object],
        curr_seq_lens: torch.Tensor,
    ) -> dict[str, object]:
        """Build runtime sparse override with dynamic seq_lens."""
        capacity = sparse_base["sparse_capacity_tokens"]
        runtime_seq_lens = torch.minimum(curr_seq_lens.int(), capacity)
        return {
            "block_table": sparse_base["block_table"],
            "seq_lens": runtime_seq_lens,
            "sparse_capacity_tokens": capacity,
            "max_seqlen_k": sparse_base["max_seqlen_k"],
        }

    def update_guidance_from_verify(
        self,
        first_queries_by_layer: dict[str, torch.Tensor] | None,
        bonus_queries_by_layer: dict[str, torch.Tensor] | None,
        valid_req_indices: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        req_ids: list[str] | None = None,
    ) -> None:
        """Update verification-guided sparse state (collect-2-query).

        Called by GPUModelRunner after verification forward pass.
        """
        if self.topk_budget <= 0:
            self._verify_guidance = None
            return
        if self.guidance_mode != "verify_collect2_query" or not self.collect2q_enabled:
            return

        self._verify_update_counter += 1
        if self.topk_update_stride > 1 and (
            self._verify_update_counter % self.topk_update_stride != 0
        ):
            return

        if (first_queries_by_layer is None
                or bonus_queries_by_layer is None
                or valid_req_indices is None):
            self._verify_guidance = None
            return
        if not first_queries_by_layer or not bonus_queries_by_layer:
            self._verify_guidance = None
            return

        batch_size = common_attn_metadata.seq_lens.shape[0]
        valid_req_indices = valid_req_indices.to(device=self.device, dtype=torch.long)

        sparse_base_by_layer: dict[str, dict[str, object]] = {}
        query_mask = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
        query_mask.index_fill_(0, valid_req_indices, True)

        for layer_name in self.guidance_layer_names:
            first_queries = first_queries_by_layer.get(layer_name)
            bonus_queries = bonus_queries_by_layer.get(layer_name)
            kv_cache = self._kv_cache_by_layer.get(layer_name)
            if first_queries is None or bonus_queries is None or kv_cache is None:
                continue
            if first_queries.numel() == 0 or bonus_queries.numel() == 0:
                continue

            n = min(
                first_queries.shape[0],
                bonus_queries.shape[0],
                valid_req_indices.shape[0],
            )
            if n <= 0:
                continue

            fused_query = (first_queries[:n] + bonus_queries[:n]) * 0.5
            full_query = torch.zeros(
                batch_size,
                fused_query.shape[1],
                fused_query.shape[2],
                device=self.device,
                dtype=fused_query.dtype,
            )
            full_query.index_copy_(0, valid_req_indices[:n], fused_query)

            sparse_base = self._build_sparse_override(
                query=full_query,
                kv_cache=kv_cache,
                common_attn_metadata=common_attn_metadata,
                batch_size=batch_size,
                query_mask=query_mask,
            )
            if sparse_base is not None:
                sparse_base_by_layer[layer_name] = sparse_base

        if not sparse_base_by_layer:
            self._verify_guidance = None
            return

        self._verify_guidance = {
            "batch_size": int(batch_size),
            "req_ids": tuple(req_ids) if req_ids is not None else None,
            "sparse_base_by_layer": sparse_base_by_layer,
        }

    def _get_verify_guidance_runtime_override(
        self,
        batch_size: int,
        curr_seq_lens: torch.Tensor,
    ) -> dict[str, dict[str, object]] | None:
        if self.guidance_mode != "verify_collect2_query":
            return None
        if self._verify_guidance is None:
            return None
        if self._verify_guidance.get("batch_size") != int(batch_size):
            return None
        cached_req_ids = self._verify_guidance.get("req_ids")
        if cached_req_ids is not None:
            current_req_ids = tuple(self.runner.input_batch.req_ids[:batch_size])
            if cached_req_ids != current_req_ids:
                return None
        sparse_base_by_layer = self._verify_guidance.get("sparse_base_by_layer")
        if sparse_base_by_layer is None:
            return None
        runtime_override: dict[str, dict[str, object]] = {}
        for layer_name in self.attn_layer_names:
            source_layer = self.guidance_source_by_layer.get(layer_name)
            if source_layer is None:
                continue
            sparse_base = sparse_base_by_layer.get(source_layer)
            if sparse_base is None:
                continue
            runtime_override[layer_name] = self._runtime_sparse_override(
                sparse_base, curr_seq_lens)
        return runtime_override or None

    def propose(
        self,
        target_token_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Draft gamma tokens with sparse attention."""
        del target_token_ids, hidden_states

        assert self.model is not None
        if self.attn_metadata_builder is None:
            self.attn_metadata_builder = (
                self._get_attention_metadata_builder())

        use_block_sparse = self.topk_budget > 0
        if use_block_sparse and not self._kv_cache_by_layer:
            self._init_kv_cache_refs()
            if not self._kv_cache_by_layer:
                use_block_sparse = False

        batch_size = next_token_ids.shape[0]

        # Derive starting position from seq_lens (which has been adjusted
        # for rejected draft tokens by the caller). seq_lens[i] - 1 gives
        # the position of the last valid token for each request.
        draft_positions = (
            common_attn_metadata.seq_lens[:batch_size].long() - 1)

        common_attn_metadata.num_actual_tokens = batch_size
        common_attn_metadata.max_query_len = 1
        common_attn_metadata.query_start_loc = (
            self.arange[:batch_size + 1])
        common_attn_metadata.query_start_loc_cpu = torch.from_numpy(
            self.token_arange_np[:batch_size + 1]).clone()

        self.input_ids[:batch_size] = next_token_ids.int()

        draft_token_ids_list: list[torch.Tensor] = []
        draft_probs_list: list[torch.Tensor] = []
        local_spec_token_ids: list[list[int]] = [[] for _ in range(batch_size)]
        # Legacy draft-query mode base sparse override (built after step-0).
        sparse_base_legacy: dict[str, dict[str, object]] | None = None

        for draft_index in range(self.gamma):
            draft_positions = draft_positions + 1
            exceeds = draft_positions >= self.max_model_len
            clamped = torch.where(
                exceeds, torch.zeros_like(draft_positions), draft_positions)

            common_attn_metadata.seq_lens += 1
            common_attn_metadata.seq_lens_cpu = (
                common_attn_metadata.seq_lens_cpu + 1)
            common_attn_metadata.seq_lens.masked_fill_(exceeds, 1)
            common_attn_metadata.num_computed_tokens_cpu = (
                common_attn_metadata.seq_lens_cpu - 1)

            block_numbers = clamped // self.block_size
            block_ids = common_attn_metadata.block_table_tensor.gather(
                dim=1, index=block_numbers.view(-1, 1)).view(-1)
            common_attn_metadata.slot_mapping = (
                block_ids * self.block_size + clamped % self.block_size)
            common_attn_metadata.slot_mapping.masked_fill_(
                exceeds, PADDING_SLOT_ID)

            draft_attn_meta = (
                self.attn_metadata_builder.build_for_drafting(
                    common_attn_metadata=common_attn_metadata,
                    draft_index=draft_index))
            per_layer_attn_metadata = {
                n: draft_attn_meta for n in self.attn_layer_names}

            self.positions[:batch_size] = clamped

            # Sparse override selection.
            sparse_override: dict[str, dict[str, object]] | None = None
            if use_block_sparse:
                # Prefer verification-guided state.
                sparse_override = self._get_verify_guidance_runtime_override(
                    batch_size=batch_size,
                    curr_seq_lens=common_attn_metadata.seq_lens[:batch_size],
                )
                if sparse_override is None and sparse_base_legacy is not None:
                    # Legacy draft_query mode (step1+).
                    sparse_override = {
                        layer_name: self._runtime_sparse_override(
                            sparse_base,
                            common_attn_metadata.seq_lens[:batch_size],
                        )
                        for layer_name, sparse_base in sparse_base_legacy.items()
                    }
                    if not sparse_override:
                        sparse_override = None

            use_sparse_this_step = sparse_override is not None

            # Legacy fallback: collect draft step-0 query when top-k is enabled,
            # verification-guided state is unavailable, and guidance_mode is
            # draft_query (or collect2q disabled).
            collect_queries = (
                use_block_sparse
                and not use_sparse_this_step
                and draft_index == 0
                and (
                    self.guidance_mode == "draft_query"
                    or not self.collect2q_enabled
                )
            )

            collected_buf = {} if collect_queries else None

            # CUDA graph: only when NOT using sparse override.
            cudagraph_mode = CUDAGraphMode.NONE
            num_input = batch_size
            if (not use_sparse_this_step
                    and self.use_cuda_graph
                    and batch_size <= self.cudagraph_batch_sizes[-1]):
                num_input = self.vllm_config.pad_for_cudagraph(batch_size)
                cudagraph_mode = CUDAGraphMode.PIECEWISE

            with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=num_input,
                cudagraph_runtime_mode=cudagraph_mode,
                draft_window_size=(
                    self.draft_window_size
                    if not use_sparse_this_step else (-1, -1)),
                collect_queries=collect_queries,
                collect_query_layers=(
                    frozenset(self.guidance_layer_names) if collect_queries else None),
                collected_queries=collected_buf,
                sparse_kv_override=sparse_override,
            ):
                hidden = self.model(
                    input_ids=self.input_ids[:num_input],
                    positions=self.positions[:num_input],
                    intermediate_tensors=None,
                )
            # Build legacy sparse base from step-0 query for each layer.
            if collect_queries and collected_buf:
                sparse_base_legacy = {}
                for layer_name, collected_query in collected_buf.items():
                    kv_cache = self._kv_cache_by_layer.get(layer_name)
                    if kv_cache is None:
                        continue
                    sparse_base = self._build_sparse_override(
                        query=collected_query,
                        kv_cache=kv_cache,
                        common_attn_metadata=common_attn_metadata,
                        batch_size=batch_size,
                    )
                    if sparse_base is not None:
                        sparse_base_legacy[layer_name] = sparse_base
                if not sparse_base_legacy:
                    sparse_base_legacy = None
                else:
                    sparse_base_legacy = {
                        layer_name: sparse_base_legacy[source_layer]
                        for layer_name, source_layer in self.guidance_source_by_layer.items()
                        if source_layer in sparse_base_legacy
                    } or None

            logits = self.model.compute_logits(hidden[:batch_size])
            processed_logits = self._process_draft_logits(
                logits,
                sampling_metadata,
                local_spec_token_ids,
            )
            # Compute draft distribution with temperature/top-k/top-p applied.
            draft_probs = processed_logits.softmax(dim=-1, dtype=torch.float32)
            # Sample from the draft distribution (NOT argmax).
            # This is mathematically required for rejection sampling:
            # the token must be drawn from the same distribution p(x)
            # that we report as draft_probs to the rejection sampler.
            if sampling_metadata.all_greedy:
                draft_token = processed_logits.argmax(dim=-1)
            else:
                draft_token = torch.multinomial(
                    draft_probs, num_samples=1).squeeze(-1)
            draft_token_ids_list.append(draft_token)
            draft_probs_list.append(draft_probs)

            for req_idx, token_id in enumerate(draft_token.tolist()):
                local_spec_token_ids[req_idx].append(token_id)

            self.input_ids[:batch_size] = draft_token.int()

        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        # Reorder probs to request-major layout:
        # [step0 batch, step1 batch, ...] -> [req0 all_steps, req1 all_steps, ...]
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
        assert self.runner is not None

        base_output_token_ids = list(
            sampling_metadata.output_token_ids[: len(local_spec_token_ids)]
        )
        if len(base_output_token_ids) < len(local_spec_token_ids):
            base_output_token_ids.extend(
                [[] for _ in range(len(local_spec_token_ids) - len(base_output_token_ids))]
            )
        step_output_token_ids = [
            [*base_output, *local_spec]
            for base_output, local_spec in zip(
                base_output_token_ids,
                local_spec_token_ids,
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
