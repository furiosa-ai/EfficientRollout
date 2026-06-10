# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import hashlib
from typing import TYPE_CHECKING, Any, Literal, get_args

from pydantic import Field, SkipValidation, model_validator
from pydantic.dataclasses import dataclass
from typing_extensions import Self

from vllm.config.parallel import ParallelConfig
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.utils.import_utils import LazyLoader, has_arctic_inference

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    import vllm.model_executor.layers.quantization as me_quant
    from vllm.config import ModelConfig
else:
    PretrainedConfig = Any
    ModelConfig = Any

    me_quant = LazyLoader(
        "model_executor", globals(), "vllm.model_executor.layers.quantization"
    )

logger = init_logger(__name__)

MTPModelTypes = Literal[
    "deepseek_mtp",
    "mimo_mtp",
    "glm4_moe_mtp",
    "ernie_mtp",
    "qwen3_next_mtp",
    "longcat_flash_mtp",
    "mtp",
    "pangu_ultra_moe_mtp",
]
EagleModelTypes = Literal["eagle", "eagle3", MTPModelTypes]
SpeculativeMethod = Literal[
    "ngram",
    "medusa",
    "mlp_speculator",
    "draft_model",
    "suffix",
    EagleModelTypes,
    "sparse_attn",
    "quant_self",
]


@config
@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""

    enforce_eager: bool | None = None
    """Override the default enforce_eager from model_config"""
    # General speculative decoding control
    num_speculative_tokens: int = Field(default=None, gt=0)
    """The number of speculative tokens, if provided. It will default to the
    number in the draft model config if present, otherwise, it is required."""
    model: str | None = None
    """The name of the draft model, eagle head, or additional weights, if
    provided."""
    method: SpeculativeMethod | None = None
    """The name of the speculative method to use. If users provide and set the
    `model` param, the speculative method type will be detected automatically
    if possible, if `model` param is not provided, the method name must be
    provided.

    If using `ngram` method, the related configuration `prompt_lookup_max` and
    `prompt_lookup_min` should be considered."""
    draft_tensor_parallel_size: int | None = Field(default=None, ge=1)
    """The degree of the tensor parallelism for the draft model. Can only be 1
    or the same as the target model's tensor parallel size."""

    # Draft model configuration
    quantization: me_quant.QuantizationMethods | None = None
    """Quantization method that was used to quantize the draft model weights.
    If `None`, we assume the model weights are not quantized. Note that it only
    takes effect when using the draft model-based speculative method."""
    max_model_len: int | None = Field(default=None, ge=1)
    """The maximum model length of the draft model. Used when testing the
    ability to skip speculation for some sequences."""
    revision: str | None = None
    """The specific model version to use for the draft model. It can be a
    branch name, a tag name, or a commit id. If unspecified, will use the
    default version."""
    code_revision: str | None = None
    """The specific revision to use for the draft model code on Hugging Face
    Hub. It can be a branch name, a tag name, or a commit id. If unspecified,
    will use the default version."""

    # Advanced control
    disable_by_batch_size: int | None = Field(default=None, ge=2)
    """Disable speculative decoding for new incoming requests when the number
    of enqueued requests is larger than this value, if provided."""
    disable_padded_drafter_batch: bool = False
    """Disable input padding for speculative decoding. If set to True,
    speculative input batches can contain sequences of different lengths,
    which may only be supported by certain attention backends. This currently
    only affects the EAGLE method of speculation."""

    # Sparse attention self-speculative decoding configuration
    sparse_window_size: int | None = None
    """Sliding window size for sparse_attn self-speculative decoding method."""

    # Quantized self-drafter config (quant_self method)
    quant_method: str = "rtn"
    """Quantization method: 'rtn' (round-to-nearest), 'fixed_beta'
    (activation-aware β=0.5), or 'replay_awq' (faithful AWQ, K-step)."""

    quant_interval: int = 1
    """Requantize drafter every N weight updates. 1 = every step."""

    quant_group_size: int = 128
    """Group size for W4 quantization."""

    sd_toggle_mode: str = "off"
    """SD toggle mode: "off" (no toggle), "threshold" (batch count),
    "roofline" (sd_toggle model-based decision)."""

    sd_toggle_threshold: int | None = None
    """Monotone SD toggle: activate speculative decoding when n_active
    requests <= this threshold. Used when sd_toggle_mode="threshold"."""

    sd_toggle_config_path: str | None = None
    """Path to sd_toggle config JSON for roofline mode."""

    sd_toggle_margin: float = 0.05
    """Safety margin for roofline toggle: require predicted speedup >= 1 + margin
    before enabling SD. Absorbs unmodeled overhead (γ-scaling proposer loop,
    post-toggle first-tick cold start). Default 0.05 tracks empirically estimated
    overhead of ~2-4%; set higher for more conservative toggling."""

    gamma_ladder: list[int] | None = None
    """Adaptive-γ elevation ladder: list of γ values to pre-capture CUDA graphs
    for, in ascending order with uniform step Δγ=2 (e.g., [5,7,9,11]). When
    set, engine starts at ladder[0] and monotonically elevates γ between RL
    rollouts using an AR-threshold rule: elevate when observed acceptance
    rate (= (MAL-1)/γ) reaches VLLM_GAMMA_AR_THRESHOLD (default 0.94).
    See EngineCore.wake_up for the elevation logic.
    When None (default), behavior matches fixed-γ: num_speculative_tokens is
    used as-is with no elevation. Requires quant_self method.
    Invariant: ladder[0] == num_speculative_tokens."""

    topk_budget: int = 0
    """Number of important past tokens to select via verification-guided
    attention scoring. 0 = disabled (use sliding window only).
    When > 0, drafting uses recent window + top-K important past tokens."""

    guidance_mode: Literal["draft_query", "verify_collect2_query"] = (
        "verify_collect2_query"
    )
    """Guidance source for sparse_attn top-k selection.

    - draft_query: collect draft step-0 query inside proposer (legacy).
    - verify_collect2_query: collect first-target and bonus queries from
      verification pass (SpecAttn-style collect-2-query).
    """

    collect2q_enabled: bool = True
    """If True, verification-guided mode collects first-target + bonus
    query vectors only. If False, falls back to legacy draft_query behavior."""

    topk_update_stride: int = Field(default=1, ge=1)
    """Refresh sparse top-k guidance every N verification updates.
    1 means update every iteration."""

    topk_candidate_cap_blocks: int = Field(default=0, ge=0)
    """Optional cap on candidate past blocks during top-k selection.
    0 means no cap."""

    guidance_layer_mode: Literal["all", "last_n", "evenly_spaced"] = "last_n"
    """Layer subset policy for verification-guided sparse attention.

    - all: all attention layers
    - last_n: the last guidance_num_layers attention layers
    - evenly_spaced: guidance_num_layers layers spaced over depth
    """

    guidance_num_layers: int = Field(default=4, ge=1)
    """Number of representative layers when guidance_layer_mode != all."""

    # Ngram proposer configuration
    prompt_lookup_max: int | None = Field(default=None, ge=1)
    """Maximum size of ngram token window when using Ngram proposer, required
    when method is set to ngram."""
    prompt_lookup_min: int | None = Field(default=None, ge=1)
    """Minimum size of ngram token window when using Ngram proposer, if
    provided. Defaults to 1."""

    speculative_token_tree: str | None = None
    """Specifies the tree structure for speculative token generation.
    """
    # required configuration params passed from engine
    target_model_config: SkipValidation[ModelConfig] = None  # type: ignore
    """The configuration of the target model."""
    target_parallel_config: SkipValidation[ParallelConfig] = None  # type: ignore
    """The parallel configuration for the target model."""

    # params generated in the post-init stage
    draft_model_config: SkipValidation[ModelConfig] = None  # type: ignore
    """The configuration of the draft model initialized internal."""
    draft_parallel_config: SkipValidation[ParallelConfig] = None  # type: ignore
    """The parallel configuration for the draft model initialized internal."""

    # Suffix decoding configuration
    suffix_decoding_max_tree_depth: int = 24
    """The maximum depth of the suffix decoding global and prompt trees. The
    tree depth limits the sum of the prefix match and speculation lengths."""

    suffix_decoding_max_cached_requests: int = 10000
    """The maximum number of requests to cache in the global suffix tree. If
    exceeded, will trigger eviction in FIFO order. If set to 0, the global
    suffix tree is disabled and past responses are not cached (prompt trees
    are still used)."""

    suffix_decoding_max_spec_factor: float = 1.0
    """The maximum spec factor for suffix decoding. The spec factor controls
    speculation lengths based on the prefix match length: max_spec_tokens =
    max_spec_factor * prefix_match_length."""

    suffix_decoding_min_token_prob: float = 0.1
    """The minimum token probability for suffix decoding. Will only speculate
    tokens with estimated probability (based on frequency counts) greater than
    or equal to this value."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        # Eagle3 affects the computation graph because it returns intermediate
        # hidden states in addition to the final hidden state.
        factors.append(self.method == "eagle3")
        factors.extend([
            self.method == "sparse_attn",
            self.sparse_window_size,
            self.topk_budget,
            self.guidance_mode,
            self.collect2q_enabled,
            self.guidance_layer_mode,
            self.guidance_num_layers,
        ])
        # Quantized self-speculative decoding uses AWQ W4 Marlin kernels
        # which affect the computation graph differently from FP16.
        factors.append(self.method == "quant_self")
        factors.append(getattr(self, 'quant_group_size', 128))
        # Adaptive-γ: ladder changes which verify-decode query lengths
        # have captured CUDA graphs. Must be part of the compile cache key
        # to prevent serving legacy (single-γ) artifacts for a ladder run.
        factors.append(tuple(self.gamma_ladder) if self.gamma_ladder else None)
        hash_str = hashlib.md5(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    @staticmethod
    def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:
        if hf_config.model_type in ("deepseek_v3", "deepseek_v32"):
            hf_config.model_type = "deepseek_mtp"
        if hf_config.model_type == "deepseek_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["DeepSeekMTPModel"]}
            )
        if hf_config.model_type in ("pangu_ultra_moe"):
            hf_config.model_type = "pangu_ultra_moe_mtp"
        if hf_config.model_type == "pangu_ultra_moe_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["OpenPanguMTPModel"]}
            )

        if hf_config.architectures[0] == "MiMoForCausalLM":
            hf_config.model_type = "mimo_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "num_hidden_layers": 0,
                    "n_predict": n_predict,
                    "architectures": ["MiMoMTPModel"],
                }
            )

        if hf_config.architectures[0] == "Glm4MoeForCausalLM":
            hf_config.model_type = "glm4_moe_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "num_hidden_layers": 0,
                    "n_predict": n_predict,
                    "architectures": ["Glm4MoeMTPModel"],
                }
            )

        if hf_config.model_type == "ernie4_5_moe":
            hf_config.model_type = "ernie_mtp"
        if hf_config.model_type == "ernie_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["ErnieMTPModel"]}
            )

        if hf_config.model_type == "qwen3_next":
            hf_config.model_type = "qwen3_next_mtp"
        if hf_config.model_type == "qwen3_next_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["Qwen3NextMTP"]}
            )
        if hf_config.model_type == "longcat_flash":
            hf_config.model_type = "longcat_flash_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["LongCatFlashMTPModel"]}
            )

        return hf_config

    def __post_init__(self):
        # Note: "method" is a new parameter that helps to extend the
        # configuration of non-model-based proposers, and the "model" parameter
        # will be used to set the draft model, eagle head, or additional weight
        # when needed. If users do not specify "method", the speculative method
        # will be detected automatically if possible. If the speculative method
        # can not be detected, it will be considered as the "draft_model" by
        # default.

        if self.sd_toggle_margin < 0.0 or self.sd_toggle_margin > 1.0:
            raise ValueError(
                "sd_toggle_margin must be in [0.0, 1.0], "
                f"got {self.sd_toggle_margin}"
            )

        if self.method in get_args(MTPModelTypes) and self.method != "mtp":
            logger.warning(
                "method `%s` is deprecated and replaced with mtp.", self.method
            )
            self.method = "mtp"

        if self.model is None and self.num_speculative_tokens is not None:
            if self.method == "mtp":
                if self.target_model_config is None:
                    raise ValueError("target_model_config must be present for mtp")
                if self.target_model_config.hf_text_config.model_type == "deepseek_v32":
                    # FIXME(luccafong): cudgraph with v32 MTP is not supported,
                    # remove this when the issue is fixed.
                    self.enforce_eager = True
                # use the draft model from the same model:
                self.model = self.target_model_config.model
                # Align the quantization of draft model for cases such as
                # --quantization fp8 with a bf16 checkpoint.
                if not self.quantization:
                    self.quantization = self.target_model_config.quantization
            elif self.method in ("ngram", "[ngram]"):
                self.model = "ngram"
            elif self.method == "suffix":
                self.model = "suffix"
            elif self.method == "sparse_attn":
                # Self-speculative: draft model = target model (no separate loading)
                self.model = self.target_model_config.model
            elif self.method == "quant_self":
                # Quantized self-speculative: draft = quantized target
                self.model = self.target_model_config.model
            else:
                raise ValueError(
                    "num_speculative_tokens was provided but without speculative model."
                )

        # Automatically configure the method for ngram when "model" is used
        # instead of "method"
        if self.method is None and (
            self.model is not None and self.model in ("ngram", "[ngram]")
        ):
            self.method = "ngram"

        if self.method in ("ngram", "[ngram]"):
            # Unified to "ngram" internally
            self.method = "ngram"
            # Set default values if not provided
            if self.prompt_lookup_min is None and self.prompt_lookup_max is None:
                # TODO(woosuk): Tune these values. They are arbitrarily chosen.
                self.prompt_lookup_min = 5
                self.prompt_lookup_max = 5
            elif self.prompt_lookup_min is None:
                if self.prompt_lookup_max is None:
                    raise ValueError(
                        "Either prompt_lookup_max or prompt_lookup_min must be "
                        "provided when using the ngram method."
                    )
                self.prompt_lookup_min = self.prompt_lookup_max
            elif self.prompt_lookup_max is None:
                if self.prompt_lookup_min is None:
                    raise ValueError(
                        "Either prompt_lookup_max or prompt_lookup_min must be "
                        "provided when using the ngram method."
                    )
                self.prompt_lookup_max = self.prompt_lookup_min

            # Validate values
            if self.prompt_lookup_min > self.prompt_lookup_max:
                raise ValueError(
                    f"prompt_lookup_min={self.prompt_lookup_min} must "
                    f"be <= prompt_lookup_max={self.prompt_lookup_max}"
                )

            # TODO: current we still need extract vocab_size from target model
            # config, in future, we may try refactor it out, and set
            # draft related config as None here.
            self.draft_model_config = self.target_model_config
            self.draft_parallel_config = self.target_parallel_config
        elif self.method == "suffix":
            self._validate_suffix_decoding()
        elif self.method == "sparse_attn":
            # Self-speculative decoding: reuse target model, no draft loading.
            self.prompt_lookup_max = 0
            self.prompt_lookup_min = 0
            self.draft_model_config = self.target_model_config
            self.draft_parallel_config = self.target_parallel_config
        elif self.method == "quant_self":
            # Quantized self-speculative: same architecture, W4 Marlin weights.
            self.prompt_lookup_max = 0
            self.prompt_lookup_min = 0
            self.draft_model_config = self.target_model_config
            self.draft_parallel_config = self.target_parallel_config
        else:
            self.prompt_lookup_max = 0
            self.prompt_lookup_min = 0

            if self.model is not None:
                # TODO: Move this import to the top once `ModelConfig`
                # lives in `vllm.config.model`.
                from vllm.config import ModelConfig

                self.draft_model_config = ModelConfig(
                    model=self.model,
                    runner="draft",
                    tokenizer=self.target_model_config.tokenizer,
                    tokenizer_mode=self.target_model_config.tokenizer_mode,
                    trust_remote_code=self.target_model_config.trust_remote_code,
                    allowed_local_media_path=self.target_model_config.allowed_local_media_path,
                    allowed_media_domains=self.target_model_config.allowed_media_domains,
                    dtype=self.target_model_config.dtype,
                    seed=self.target_model_config.seed,
                    revision=self.revision,
                    code_revision=self.code_revision,
                    tokenizer_revision=self.target_model_config.tokenizer_revision,
                    spec_target_max_model_len=self.target_model_config.max_model_len,
                    quantization=self.quantization,
                    enforce_eager=self.target_model_config.enforce_eager,
                    max_logprobs=self.target_model_config.max_logprobs,
                    hf_overrides=SpeculativeConfig.hf_config_override,
                )

                # Automatically detect the method
                if self.method in ("eagle", "eagle3"):
                    pass
                # examples:
                # yuhuili/EAGLE-LLaMA3-Instruct-8B
                # yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
                # AngelSlim/Qwen3-8B_eagle3
                elif "eagle-" in self.draft_model_config.model.lower():
                    self.method = "eagle"
                elif "eagle3" in self.draft_model_config.model.lower():
                    self.method = "eagle3"
                elif self.draft_model_config.hf_config.model_type == "medusa":
                    self.method = "medusa"
                elif self.draft_model_config.hf_config.model_type == "mlp_speculator":
                    self.method = "mlp_speculator"
                elif self.draft_model_config.hf_config.model_type in get_args(
                    MTPModelTypes
                ):
                    self.method = "mtp"
                    if self.num_speculative_tokens > 1:
                        logger.warning(
                            "Enabling num_speculative_tokens > 1 will run"
                            "multiple times of forward on same MTP layer"
                            ",which may result in lower acceptance rate"
                        )
                elif self.draft_model_config.hf_config.model_type in (
                    "longcat_flash_mtp"
                ):
                    self.method = "longcat_flash_mtp"
                    if self.num_speculative_tokens > 1:
                        logger.warning(
                            "LongCat MTP models only have "
                            "one layer. Might need some code changes "
                            "to support multiple layers."
                        )
                else:
                    self.method = "draft_model"
                    raise NotImplementedError(
                        "Speculative decoding with draft model is not "
                        "supported yet. Please consider using other "
                        "speculative decoding methods such as ngram, medusa, "
                        "eagle, or mtp."
                    )

                # Replace hf_config for EAGLE draft_model
                if self.method in ("eagle", "eagle3"):
                    from vllm.transformers_utils.configs import SpeculatorsConfig
                    from vllm.transformers_utils.configs.eagle import EAGLEConfig

                    if isinstance(
                        self.draft_model_config.hf_config,
                        (EAGLEConfig, SpeculatorsConfig),
                    ):
                        pass
                    else:
                        eagle_config = EAGLEConfig(
                            self.draft_model_config.hf_config,
                            method=self.method,
                            model_type="eagle",
                        )
                        self.draft_model_config.hf_config = eagle_config

                if self.num_speculative_tokens is not None and hasattr(
                    self.draft_model_config.hf_config, "num_lookahead_tokens"
                ):
                    self.draft_model_config.hf_config.num_lookahead_tokens = (
                        self.num_speculative_tokens
                    )

                n_predict = getattr(
                    self.draft_model_config.hf_config, "n_predict", None
                )
                if n_predict is not None:
                    if self.num_speculative_tokens is None:
                        # Default to max value defined in draft model config.
                        self.num_speculative_tokens = n_predict
                    elif (
                        self.num_speculative_tokens > n_predict
                        and self.num_speculative_tokens % n_predict != 0
                    ):
                        # Ensure divisibility for MTP module reuse.
                        raise ValueError(
                            f"num_speculative_tokens:{self.num_speculative_tokens}"
                            f" must be divisible by {n_predict=}"
                        )

                if self.speculative_token_tree is None:
                    # Generate chain of tokens.
                    self.speculative_token_tree = str(
                        [(i + 1) * (0,) for i in range(self.num_speculative_tokens)]
                    )
                else:
                    # Sort the token tree breadth-first.
                    tree_choices = ast.literal_eval(self.speculative_token_tree)
                    self.speculative_token_tree = str(
                        sorted(tree_choices, key=lambda t: (len(t), t))
                    )

                self.draft_tensor_parallel_size = (
                    SpeculativeConfig._verify_and_get_draft_tp(
                        self.target_parallel_config,
                        self.draft_tensor_parallel_size,
                        self.draft_model_config.hf_config,
                    )
                )

                self.draft_model_config.max_model_len = (
                    SpeculativeConfig._maybe_override_draft_max_model_len(
                        self.max_model_len,
                        self.draft_model_config.max_model_len,
                        self.target_model_config.max_model_len,
                    )
                )

                self.draft_parallel_config = (
                    SpeculativeConfig.create_draft_parallel_config(
                        self.target_parallel_config, self.draft_tensor_parallel_size
                    )
                )
        return self

    def _validate_suffix_decoding(self):
        if not has_arctic_inference():
            raise ImportError(
                "Arctic Inference is required for suffix decoding. "
                "Install via `pip install arctic-inference==0.1.1`."
            )
        if self.num_speculative_tokens is None:
            # Suffix decoding decides the actual number of speculative tokens
            # dynamically and treats num_speculative_tokens as a maximum limit.
            self.num_speculative_tokens = self.suffix_decoding_max_tree_depth
            logger.warning(
                "Defaulted num_speculative_tokens to %s for suffix decoding.",
                self.num_speculative_tokens,
            )
        # Validate values
        if self.suffix_decoding_max_tree_depth < 1:
            raise ValueError(
                f"suffix_decoding_max_tree_depth="
                f"{self.suffix_decoding_max_tree_depth} must be >= 1"
            )
        if self.suffix_decoding_max_cached_requests < 0:
            raise ValueError(
                f"suffix_decoding_max_cached_requests="
                f"{self.suffix_decoding_max_cached_requests} must be >= 0"
            )
        if self.suffix_decoding_max_spec_factor < 0:
            raise ValueError(
                f"suffix_decoding_max_spec_factor="
                f"{self.suffix_decoding_max_spec_factor} must be >= 0"
            )
        if not 0 <= self.suffix_decoding_min_token_prob <= 1:
            raise ValueError(
                f"suffix_decoding_min_token_prob="
                f"{self.suffix_decoding_min_token_prob} must be in [0, 1]"
            )

    @staticmethod
    def _maybe_override_draft_max_model_len(
        speculative_max_model_len: int | None,
        draft_max_model_len: int,
        target_max_model_len: int,
    ) -> int:
        """Determine the max sequence len for the draft model. This is usually
        the draft_max_model_len, but may be the target_max_model_len if it is
        less than the draft_max_model_len, or may be speculative_max_model_len
        if it is specified.

        This is necessary so that sequences do not exceed the capacity of the
        draft model or the target model.

        speculative_max_model_len is mainly used for testing that sequences can
        skip speculation.
        """

        if speculative_max_model_len is not None:
            if speculative_max_model_len > draft_max_model_len:
                raise ValueError(
                    f"{speculative_max_model_len=} cannot be "
                    f"larger than {draft_max_model_len=}"
                )

            if speculative_max_model_len > target_max_model_len:
                raise ValueError(
                    f"{speculative_max_model_len=} cannot be "
                    f"larger than {target_max_model_len=}"
                )

            return speculative_max_model_len

        return min(
            draft_max_model_len,
            target_max_model_len,
        )

    @staticmethod
    def _verify_and_get_draft_tp(
        target_parallel_config: ParallelConfig,
        speculative_draft_tensor_parallel_size: int | None,
        draft_hf_config: PretrainedConfig,
    ) -> int:
        """
        Verifies and adjusts the tensor parallel size for a draft model
        specified using speculative_draft_tensor_parallel_size.
        """
        # If speculative_draft_tensor_parallel_size is unset then set it
        # appropriately else verify that it is set correctly.
        if speculative_draft_tensor_parallel_size is None:
            if draft_hf_config.model_type == "mlp_speculator":
                speculative_draft_tensor_parallel_size = 1
                if target_parallel_config.tensor_parallel_size > 1:
                    logger.warning(
                        "%s cannot currently be run with tp>1; "
                        "setting speculative_draft_tensor_parallel_size=1",
                        draft_hf_config.model_type,
                    )
            else:
                speculative_draft_tensor_parallel_size = (
                    target_parallel_config.tensor_parallel_size
                )
        elif speculative_draft_tensor_parallel_size not in (
            1,
            target_parallel_config.tensor_parallel_size,
        ):
            raise ValueError(
                f"{speculative_draft_tensor_parallel_size=} cannot be "
                f"other value than 1 or target model tensor_parallel_size"
            )
        return speculative_draft_tensor_parallel_size

    @staticmethod
    def create_draft_parallel_config(
        target_parallel_config: ParallelConfig,
        speculative_draft_tensor_parallel_size: int,
    ) -> ParallelConfig:
        """Create a parallel config for use by the draft worker.

        This is mostly a copy of the target parallel config, except the tp_size.
        """
        draft_parallel_config = ParallelConfig(
            pipeline_parallel_size=target_parallel_config.pipeline_parallel_size,
            tensor_parallel_size=speculative_draft_tensor_parallel_size,
            distributed_executor_backend=target_parallel_config.distributed_executor_backend,
            max_parallel_loading_workers=target_parallel_config.max_parallel_loading_workers,
            disable_custom_all_reduce=target_parallel_config.disable_custom_all_reduce,
            ray_workers_use_nsight=target_parallel_config.ray_workers_use_nsight,
            placement_group=target_parallel_config.placement_group,
        )

        return draft_parallel_config

    @model_validator(mode="after")
    def _verify_args(self) -> Self:
        if self.num_speculative_tokens is None:
            raise ValueError(
                "num_speculative_tokens must be provided with "
                "speculative model unless the draft model config contains an "
                "n_predict parameter."
            )

        if self.num_speculative_tokens <= 0:
            raise ValueError(
                "Expected num_speculative_tokens to be greater "
                f"than zero ({self.num_speculative_tokens})."
            )

        if self.draft_model_config:
            self.draft_model_config.verify_with_parallel_config(
                self.draft_parallel_config
            )

        if self.disable_by_batch_size is not None and self.disable_by_batch_size < 2:
            raise ValueError(
                "Expect the batch size threshold of disabling "
                "speculative decoding is > 1, but got "
                f"{self.disable_by_batch_size=}"
            )

        # Adaptive-γ ladder invariants:
        # (1) ladder[0] must equal num_speculative_tokens (starting γ) so that
        #     engine init buffers, dispatcher keys, and scheduler state all
        #     agree on the starting γ.
        # (2) ladder must be strictly ascending, unique, positive.
        # (3) ε ∈ [0, 1].
        if self.gamma_ladder is not None:
            _ladder = list(self.gamma_ladder)
            if len(_ladder) < 2:
                raise ValueError(
                    f"gamma_ladder must have >= 2 values for elevation, "
                    f"got {_ladder}"
                )
            if _ladder != sorted(_ladder) or len(set(_ladder)) != len(_ladder):
                raise ValueError(
                    f"gamma_ladder must be strictly ascending with unique "
                    f"values, got {_ladder}"
                )
            if any(g < 1 for g in _ladder):
                raise ValueError(
                    f"gamma_ladder values must be >= 1, got {_ladder}"
                )
            if _ladder[0] != self.num_speculative_tokens:
                raise ValueError(
                    f"Adaptive-γ invariant violated: "
                    f"gamma_ladder[0]={_ladder[0]} must equal "
                    f"num_speculative_tokens={self.num_speculative_tokens}. "
                    f"Set num_speculative_tokens={_ladder[0]} or adjust "
                    f"the ladder."
                )
            if self.method not in (None, "quant_self"):
                raise ValueError(
                    f"gamma_ladder only supported for method='quant_self', "
                    f"got '{self.method}'"
                )
            # Uniform step Δγ=2 required as a ladder convention: CUDA graphs
            # are pre-captured for each γ in the ladder at init, and Δγ=2
            # keeps the ladder well-spaced (e.g., [5,7,9,11]) without
            # over-populating the dispatch cache.
            _deltas = [
                _ladder[i + 1] - _ladder[i] for i in range(len(_ladder) - 1)
            ]
            if any(d != 2 for d in _deltas):
                raise ValueError(
                    f"gamma_ladder must use uniform step Δγ=2 "
                    f"(e.g., [5,7,9,11]); got deltas={_deltas}"
                )

        eagle3_target_supported = ["llama", "qwen", "minicpm", "gpt_oss"]
        if (
            self.method == "eagle3"
            and self.target_model_config
            and not any(
                supported_model in self.target_model_config.hf_text_config.model_type
                for supported_model in eagle3_target_supported
            )
        ):
            raise ValueError(
                f"Eagle3 is only supported for {eagle3_target_supported} models. "  # noqa: E501
                f"Got {self.target_model_config.hf_text_config.model_type=}"
            )

        return self

    @property
    def num_lookahead_slots(self) -> int:
        """The number of additional slots the scheduler should allocate per
        step, in addition to the slots allocated for each known token.

        This is equal to the number of speculative tokens, as each speculative
        token must be scored. When adaptive-γ (gamma_ladder) is enabled, the
        scheduler must reserve slots for the MAX γ on the ladder so KV cache
        and buffer allocations never underflow during elevation.
        """
        if self.gamma_ladder:
            return max(self.gamma_ladder)
        return self.num_speculative_tokens

    def use_eagle(self) -> bool:
        return self.method in ("eagle", "eagle3", "mtp")

    def __repr__(self) -> str:
        method = self.method
        model = None if method in ("ngram", "suffix") else self.draft_model_config.model
        num_spec_tokens = self.num_speculative_tokens
        return f"SpeculativeConfig({method=}, {model=}, {num_spec_tokens=})"
