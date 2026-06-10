# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from itertools import product

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor
from vllm.logger import init_logger

logger = init_logger(__name__)


class CudagraphDispatcher:
    """
    Runtime cudagraph dispatcher to dispatch keys for multiple set of
    cudagraphs.

    The dispatcher stores two sets of dispatch keys, one for PIECEWISE and one
    for FULL cudagraph runtime mode. The keys are initialized depending on
    attention support and what cudagraph mode is set in CompilationConfig. The
    keys stored in dispatcher are the only source of truth for valid
    cudagraphs that can be dispatched at runtime.

    At runtime, the dispatch method generates the runtime cudagraph mode (FULL,
    PIECEWISE, or NONE for no cudagraph) and the valid key (batch descriptor)
    based on the input key. After dispatching (communicated via forward
    context), the cudagraph wrappers will trust the dispatch key to either
    capture or replay (if the mode matches), or pass through to the underlying
    runnable without cudagraph (if the mode does not match or mode is NONE).
    """

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.cudagraph_mode = self.compilation_config.cudagraph_mode

        # Dict to store valid cudagraph dispatching keys.
        self.cudagraph_keys: dict[CUDAGraphMode, set[BatchDescriptor]] = {
            CUDAGraphMode.PIECEWISE: set(),
            CUDAGraphMode.FULL: set(),
        }

        not_use_piecewise_compilation = (
            not self.cudagraph_mode.requires_piecewise_compilation()
        )

        assert (
            not_use_piecewise_compilation
            or self.compilation_config.is_attention_compiled_piecewise()
        ), (
            "Compilation mode should be CompilationMode.VLLM_COMPILE when "
            "cudagraph_mode piecewise cudagraphs is used, "
            "and attention should be in splitting_ops or "
            "inductor splitting should be used. "
            f"cudagraph_mode={self.cudagraph_mode}, "
            f"compilation_mode={self.compilation_config.mode}, "
            f"splitting_ops={self.compilation_config.splitting_ops}"
        )

        self.keys_initialized = False

    def add_cudagraph_key(
        self, runtime_mode: CUDAGraphMode, batch_descriptor: BatchDescriptor
    ):
        assert runtime_mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL], (
            f"Invalid cudagraph runtime mode for keys: {runtime_mode}"
        )
        self.cudagraph_keys[runtime_mode].add(batch_descriptor)

    def initialize_cudagraph_keys(
        self, cudagraph_mode: CUDAGraphMode, uniform_decode_query_len: int,
        sd_toggle_threshold: int | None = None,
        gamma_ladder: list[int] | None = None,
    ):
        # This should be called only after attention backend is initialized.

        # LoRA activation cases to specialize the cuda graphs on
        if self.vllm_config.lora_config:
            if self.compilation_config.cudagraph_specialize_lora:
                lora_cases = [True, False]
            else:
                lora_cases = [True]
        else:
            lora_cases = [False]

        # Note: we create all valid keys for cudagraph here but do not
        # guarantee all keys would be used. For example, if we allow lazy
        # capturing in future PR, some keys may never be triggered.
        if cudagraph_mode.mixed_mode() != CUDAGraphMode.NONE:
            for bs, has_lora in product(
                self.compilation_config.cudagraph_capture_sizes, lora_cases
            ):
                self.add_cudagraph_key(
                    cudagraph_mode.mixed_mode(),
                    BatchDescriptor(
                        num_tokens=bs, uniform_decode=False, has_lora=has_lora
                    ),
                )

        # if decode cudagraph mode is FULL, and we don't already have mixed
        # mode full cudagraphs then add them here.
        if (
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and cudagraph_mode.separate_routine()
        ):
            max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
            # Adaptive-γ: register SD FULL keys for EACH γ on the ladder
            # (decode_query_len = γ+1). Without this, runtime dispatch after
            # elevation would fail to find the pre-captured graph for the
            # higher γ, falling back to NONE/eager.
            if gamma_ladder:
                _sd_qlens = [1 + g for g in gamma_ladder]
            else:
                _sd_qlens = [uniform_decode_query_len]

            # Use MAX qlen to bound max_num_tokens (upper bound across ladder).
            max_num_tokens = max(_sd_qlens) * max_num_seqs
            cudagraph_capture_sizes_for_decode = [
                x
                for x in self.compilation_config.cudagraph_capture_sizes
                if x <= max_num_tokens and x >= 1
            ]
            # SD FULL keys (decode_query_len = uniform_decode_query_len).
            # When toggle is active, SD only fires at batch ≤ threshold,
            # so max SD tokens = threshold × uniform_decode_query_len.
            # Include the ceiling capture size (pad_for_cudagraph result)
            # because batch=threshold pads UP to next capture size.
            for _qlen in _sd_qlens:
                _sizes_for_qlen = [
                    x for x in cudagraph_capture_sizes_for_decode
                    if x >= _qlen
                ]
                if (
                    sd_toggle_threshold is not None
                    and sd_toggle_threshold > 0
                ):
                    max_sd_tokens = sd_toggle_threshold * _qlen
                    sd_ceiling = next(
                        (x for x in _sizes_for_qlen if x >= max_sd_tokens),
                        max(_sizes_for_qlen) if _sizes_for_qlen else max_num_tokens,
                    )
                    sd_capture_sizes = [
                        x for x in _sizes_for_qlen if x <= sd_ceiling
                    ]
                else:
                    sd_capture_sizes = _sizes_for_qlen
                for bs, has_lora in product(sd_capture_sizes, lora_cases):
                    self.add_cudagraph_key(
                        CUDAGraphMode.FULL,
                        BatchDescriptor(
                            num_tokens=bs, uniform_decode=True, has_lora=has_lora,
                            decode_query_len=_qlen,
                        ),
                    )
            # AR FULL keys for SD-toggle fallback (decode_query_len=1).
            # Only needed when toggle is active (threshold > 0).
            # AR fallback fires when active_batch > threshold, so
            # min AR num_tokens = threshold+1, padded up to capture size.
            # Max AR num_tokens = max_num_seqs (1 token/request).
            # Include ceiling: pad_for_cudagraph(max_num_seqs) may round
            # above max_num_seqs (e.g. 256→258). _dummy_run handles
            # this via virtual padding (last request absorbs excess).
            if (
                uniform_decode_query_len > 1
                and sd_toggle_threshold is not None
                and sd_toggle_threshold > 0
            ):
                # Find the smallest capture size >= max_num_seqs
                # (avoids pad_for_cudagraph which may IndexError if
                # max_num_seqs > max_cudagraph_capture_size)
                ar_ceiling = next(
                    (x for x in cudagraph_capture_sizes_for_decode
                     if x >= max_num_seqs),
                    max_num_tokens,  # fallback to SD ceiling
                )
                ar_capture_sizes = [
                    x for x in cudagraph_capture_sizes_for_decode
                    if x > sd_toggle_threshold and x <= ar_ceiling
                ]
                for bs, has_lora in product(ar_capture_sizes, lora_cases):
                    self.add_cudagraph_key(
                        CUDAGraphMode.FULL,
                        BatchDescriptor(
                            num_tokens=bs, uniform_decode=True,
                            has_lora=has_lora, decode_query_len=1,
                        ),
                    )
        self.keys_initialized = True

    def dispatch(
        self, batch_descriptor: BatchDescriptor, use_cascade_attn: bool = False
    ) -> tuple[CUDAGraphMode, BatchDescriptor | None]:
        """
        Given conditions(e.g.,batch descriptor and if using cascade attention),
        dispatch to a cudagraph runtime mode and the valid batch descriptor.
        A new batch descriptor is returned as we might dispatch a uniform batch
        to a graph that supports a more general batch (uniform to non-uniform).
        """
        # if not initialized, just skip dispatching.
        if not self.keys_initialized:
            return CUDAGraphMode.NONE, None

        non_uniform_key = batch_descriptor.non_uniform
        # if a batch use cascade attention, bypass checking full cudagraphs
        if not use_cascade_attn:
            # check if key exists for full cudagraph
            if batch_descriptor in self.cudagraph_keys[CUDAGraphMode.FULL]:
                return CUDAGraphMode.FULL, batch_descriptor

            # otherwise, check if non-uniform key exists
            if non_uniform_key in self.cudagraph_keys[CUDAGraphMode.FULL]:
                return CUDAGraphMode.FULL, non_uniform_key

        # also check if non-uniform key exists for more "general"
        # piecewise cudagraph
        if non_uniform_key in self.cudagraph_keys[CUDAGraphMode.PIECEWISE]:
            return CUDAGraphMode.PIECEWISE, non_uniform_key

        # finally, just return no cudagraphs
        return CUDAGraphMode.NONE, None
