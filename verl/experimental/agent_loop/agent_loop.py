# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import heapq
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Optional
from uuid import uuid4

import hydra
import numpy as np
import ray
import torch
from cachetools import LRUCache
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDict
from transformers import AutoProcessor, AutoTokenizer

from verl.experimental.agent_loop.prometheus_utils import update_prometheus_config
from verl.experimental.agent_loop.utils import resolve_config_path
from verl.experimental.reward_loop import RewardLoopWorker
from verl.protocol import DataProto
from verl.single_controller.ray.base import RayResourcePool, RayWorkerGroup
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.chat_template import initialize_system_prompt
from verl.utils.dataset.rl_dataset import RLHFDataset, get_dataset_class
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils.ray_utils import get_event_loop
from verl.utils.rollout_trace import (
    RolloutTraceConfig,
    rollout_trace_attr,
    rollout_trace_op,
)
from verl.utils.transferqueue_utils import tqbridge
from verl.workers.rollout.replica import TokenOutput, get_rollout_replica_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AsyncLLMServerManager:
    """
    A class to manage multiple OpenAI compatible LLM servers. This class provides
    - Load balance: least requests load balancing
    - Sticky session: send multi-turn chat completions to same server for automatic prefix caching
    """

    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], max_cache_size: int = 10000):
        """Initialize the AsyncLLMServerManager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            max_cache_size (int, optional): max cache size for request_id to server mapping. Defaults to 10000.
        """
        self.config = config
        self._canonical_handles = list(server_handles)
        self.server_handles = server_handles
        random.shuffle(self.server_handles)

        # Least requests load balancing
        self.weighted_serveres = [[0, idx, server] for idx, server in enumerate(self.server_handles)]
        heapq.heapify(self.weighted_serveres)

        # LRU cache to map request_id to server
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)

        # server handle → canonical GPU index (for per-GPU profiling)
        self._handle_to_gpu_idx: dict[ray.actor.ActorHandle, int] = {
            h: i for i, h in enumerate(self._canonical_handles)
        }
        self._gpu_response_lengths: dict[int, list[int]] = defaultdict(list)
        # Per-GPU timing: gpu_idx → list of (start_ts, end_ts) per request
        self._gpu_request_times: dict[int, list[tuple[float, float]]] = defaultdict(list)

    def _choose_server(self, request_id: str, target_gpu: int = -1) -> ray.actor.ActorHandle:
        if request_id in self.request_id_to_server:
            return self.request_id_to_server[request_id]

        if target_gpu >= 0:
            server = self._canonical_handles[target_gpu]
        else:
            _, _, server = self.weighted_serveres[0]
            self.weighted_serveres[0][0] += 1
            heapq.heapreplace(self.weighted_serveres, self.weighted_serveres[0])

        self.request_id_to_server[request_id] = server
        return server

    def _record_gpu_response(self, request_id: str, response_len: int, start_ts: float, end_ts: float) -> None:
        """Record that a request completed with a given response length on its assigned GPU."""
        server = self.request_id_to_server.get(request_id)
        if server is not None:
            gpu_idx = self._handle_to_gpu_idx[server]
            self._gpu_response_lengths[gpu_idx].append(response_len)
            self._gpu_request_times[gpu_idx].append((start_ts, end_ts))

    def pop_per_gpu_stats(self) -> dict[int, dict]:
        """Return per-GPU stats {gpu_idx: {"lengths": [...], "makespan_sec": float}} and reset."""
        result: dict[int, dict] = {}
        all_gpu_ids = set(self._gpu_response_lengths.keys()) | set(self._gpu_request_times.keys())
        for gpu_idx in all_gpu_ids:
            lengths = self._gpu_response_lengths.get(gpu_idx, [])
            times = self._gpu_request_times.get(gpu_idx, [])
            if times:
                makespan = max(t[1] for t in times) - min(t[0] for t in times)
            else:
                makespan = 0.0
            result[gpu_idx] = {"lengths": lengths, "makespan_sec": makespan}
        self._gpu_response_lengths.clear()
        self._gpu_request_times.clear()

        # NOTE: SD stats (get_sd_stats/reset_sd_stats) are NOT collected here.
        # Multiple workers share the same server handles, so per-worker collection
        # causes a race condition. SD stats are collected once at the manager level
        # in AgentLoopManager.generate_sequences() after all workers complete.

        return result

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        target_gpu: int = -1,
    ) -> TokenOutput:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.
            target_gpu (int): pre-assigned GPU index from AllocationPolicy. -1 = use default LB.

        Returns:
            TokenOutput: token output
        """
        server = self._choose_server(request_id, target_gpu=target_gpu)
        start_ts = time.monotonic()
        output = await server.generate.remote(
            request_id=uuid4().hex,  # use new request_id for each turn
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            video_data=video_data,
        )
        end_ts = time.monotonic()
        self._record_gpu_response(request_id, len(output.token_ids), start_ts, end_ts)
        return output


class AgentLoopMetrics(BaseModel):
    """Agent loop performance metrics."""

    generate_sequences: float = 0.0
    tool_calls: float = 0.0
    oot_fallback_count: int = 0
    """Out-of-top-N logprob fallback events count per request (SD recovery
    tokens sampled outside vLLM's top-N logprobs dict). 0 in normal sampling."""
    first_entry_mismatch_count: int = 0
    """Positions where the logprobs dict's first entry token_id mismatches
    the response token_id. Sanity check for the SD invariant
    (first entry = final accepted token + target logprob). 0 if invariant holds."""


class AgentLoopOutput(BaseModel):
    """Agent loop output."""

    prompt_ids: list[int]
    """Prompt token ids."""
    response_ids: list[int]
    """Response token ids including LLM generated token, tool response token."""
    response_mask: list[int]
    """Response mask, 1 for LLM generated token, 0 for tool response token."""
    response_logprobs: Optional[list[float]] = None
    """Log probabilities for the response tokens."""
    routed_experts: Optional[Any] = None
    """Routed experts for the total tokens."""
    multi_modal_data: Optional[dict[str, Any]] = None
    """Multi-modal data for multi-modal tools."""
    reward_score: Optional[float] = None
    """Reward score for the trajectory."""
    num_turns: int = 0
    """Number of chat turns, including user, assistant, tool."""
    metrics: AgentLoopMetrics
    """Auxiliary performance metrics"""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class _InternalAgentLoopOutput(AgentLoopOutput):
    """Internal agent loop output with padded sequences."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: torch.Tensor
    """Padded prompt token ids."""
    response_ids: torch.Tensor
    """Padded response token ids."""
    input_ids: torch.Tensor
    """Padded input ids(prompt_ids + response_ids)."""
    position_ids: torch.Tensor
    """Padded position ids."""
    response_mask: torch.Tensor
    """Padded response mask."""
    attention_mask: torch.Tensor
    """Padded attention mask."""
    response_logprobs: Optional[torch.Tensor] = None
    """Padded log probabilities for the response tokens."""
    routed_experts: Optional[torch.Tensor] = None
    """Padded routed experts for the total tokens."""
    multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None
    """Multi-modal inputs for processors (e.g., pixel_values, image_grid_thw)."""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class DictConfigWrap:
    """Wrapper for DictConfig to avoid hydra.utils.instantiate recursive resolve."""

    def __init__(self, config: DictConfig):
        self.config = config


class AgentLoopBase(ABC):
    """An agent loop takes an input message, chat with OpenAI compatible LLM server and interact with various
    environments."""

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        dataset_cls: type[RLHFDataset],
        dataset_config: DictConfig,
        **kwargs,
    ):
        """Initialize agent loop, each sample will have its own loop instance.

        Args:
            trainer_config (DictConfigWrap): trainer config.
            server_manager (AsyncLLMServerManager): OpenAI compatible LLM server manager.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
            processor (AutoProcessor): Processor for process messages.
            dataset_cls (type[Dataset]): Dataset class for creating dataset, Defaults to RLHFDataset.
            dataset_config (DictConfig): Dataset config.
        """
        self.config = trainer_config.config
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.processor = processor
        self.dataset_cls = dataset_cls
        self.dataset_config = dataset_config
        self.apply_chat_template_kwargs = dataset_config.get("apply_chat_template_kwargs", {})
        self.system_prompt = initialize_system_prompt(self.tokenizer, **self.apply_chat_template_kwargs)
        self.loop = get_event_loop()

    async def process_vision_info(self, messages: list[dict]) -> dict:
        """Extract images and videos from messages.

        Args:
            messages (list[dict]): Input messages.

        Returns:
            dict: Multi-modal data with keys "images" and "videos".
        """
        multi_modal_data = {}
        if self.processor is not None:
            images, videos = await self.dataset_cls.process_vision_info(
                messages, image_patch_size=self.processor.image_processor.patch_size, config=self.dataset_config
            )
            if images is not None:
                multi_modal_data["images"] = images
            if videos is not None:
                multi_modal_data["videos"] = videos

        return multi_modal_data

    async def apply_chat_template(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        images: list[Image.Image] = None,
        videos: list[tuple[torch.Tensor, dict]] = None,
        remove_system_prompt: bool = False,
    ):
        """Apply chat template to messages with optional tools, images, and videos.

        Args:
            messages (list[dict]): Input messages.
            tools (list[dict], optional): Tools schemas. Defaults to None.
            images (list[Image.Image], optional): Input images. Defaults to None.
            videos (list[tuple[torch.Tensor, dict]], optional): Input videos. Defaults to None.
            remove_system_prompt (bool, optional): Whether to remove system prompt. Defaults to False.

        Returns:
            list[int]: Prompt token ids.
        """
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )

            # split the videos and according metadatas
            if videos is not None:
                videos, video_metadatas = zip(*videos, strict=False)
                videos, video_metadatas = list(videos), list(video_metadatas)
            else:
                video_metadatas = None

            model_inputs = self.processor(
                text=[raw_prompt],
                images=images,
                videos=videos,
                video_metadatas=video_metadatas,
                return_tensors="pt",
                do_sample_frames=False,
            )
            prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            if self.tokenizer.chat_template is not None:
                prompt_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(
                        messages,
                        tools=tools,
                        add_generation_prompt=True,
                        tokenize=True,
                        **self.apply_chat_template_kwargs,
                    ),
                )
            else:
                # Base model without chat template: tokenize raw content
                text = "".join(
                    msg["content"] for msg in messages
                    if isinstance(msg.get("content"), str)
                )
                prompt_ids = self.tokenizer.encode(text)

        if remove_system_prompt:
            prompt_ids = prompt_ids[len(self.system_prompt) :]

        return prompt_ids

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run agent loop to interact with LLM server and environment.

        Args:
            sampling_params (Dict[str, Any]): LLM sampling params.
            **kwargs: dataset fields from `verl.utils.dataset.RLHFDataset`.

        Returns:
            AgentLoopOutput: Agent loop output.
        """
        raise NotImplementedError


"""Agent loop registry: key is agent_name, value is a dict of agent loop config
used by hydra.utils.instantiate to initialize agent loop instance.

https://hydra.cc/docs/advanced/instantiate_objects/overview/
"""
_agent_loop_registry: dict[str, dict] = {}


def register(agent_name: str):
    """Register agent loop class."""

    def decorator(subclass: type[AgentLoopBase]) -> type[AgentLoopBase]:
        fqdn = f"{subclass.__module__}.{subclass.__qualname__}"
        _agent_loop_registry[agent_name] = {"_target_": fqdn}
        return subclass

    return decorator


class AgentLoopWorkerBase:
    """Agent loop worker takes a batch of messages and run each message in an agent loop."""

    def __init__(
        self,
        config: DictConfig,
        server_handles: list[ray.actor.ActorHandle],
        reward_router_address: str = None,
    ):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
        """
        self.config = config

        # for recipe to change
        if not hasattr(self, "server_manager"):
            self.server_manager = AsyncLLMServerManager(config, server_handles)

        self.dataset_cls = get_dataset_class(config.data)
        self.reward_router_address = reward_router_address

        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.processor = hf_processor(local_path, trust_remote_code=True)

        agent_loop_config_path = config.actor_rollout_ref.rollout.agent.agent_loop_config_path
        if agent_loop_config_path:
            resolved_path = resolve_config_path(agent_loop_config_path)
            agent_loop_configs = OmegaConf.load(resolved_path)
            for agent_loop_config in agent_loop_configs:
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config
        if self.config.actor_rollout_ref.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.actor_rollout_ref.model.custom_chat_template
            self.tokenizer.chat_template = self.config.actor_rollout_ref.model.custom_chat_template

        use_reward_loop = True if self.config.reward_model.use_reward_loop else None
        self.use_reward_loop = use_reward_loop
        if use_reward_loop and not hasattr(self, "reward_loop_worker"):
            self.reward_loop_worker = RewardLoopWorker.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                ),
            ).remote(self.config, self.reward_router_address)

        trace_config = self.config.actor_rollout_ref.rollout.get("trace", {})
        RolloutTraceConfig.init(
            self.config.trainer.project_name,
            self.config.trainer.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
            trace_config.get("max_samples_per_step_per_worker", None),
        )

    @tqbridge()
    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
            frequency_penalty=config.get("frequency_penalty", 0.0),
            presence_penalty=config.get("presence_penalty", 0.0),
            logprobs=config.calculate_log_probs,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        max_samples_per_worker = RolloutTraceConfig.get_instance().max_samples_per_step_per_worker

        # For n rollouts per sample, we trace all n rollouts for selected samples
        # Note: This sampling happens per-worker, so total traces = max_samples_per_worker * num_workers * n
        if max_samples_per_worker is not None:
            unique_sample_indices = np.unique(index)
            if max_samples_per_worker < len(unique_sample_indices):
                selected_samples = set(
                    np.random.choice(unique_sample_indices, max_samples_per_worker, replace=False).tolist()
                )
                traced_indices = set(i for i in range(len(batch)) if index[i] in selected_samples)
            else:
                traced_indices = set(range(len(batch)))
        else:
            traced_indices = set(range(len(batch)))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )

        tasks = []
        for i in range(len(batch)):
            trace_this_sample = i in traced_indices
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            tasks.append(
                asyncio.create_task(
                    self._run_agent_loop(sampling_params, trajectory_info[i], trace=trace_this_sample, **kwargs)
                )
            )
        outputs = await asyncio.gather(*tasks)

        output = self._postprocess(outputs)

        output.meta_info["per_gpu_response_lengths"] = self.server_manager.pop_per_gpu_stats()

        return output

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs,
    ) -> _InternalAgentLoopOutput:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
            trace=trace,
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )

            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=DictConfigWrap(config=self.config),
                server_manager=self.server_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
                dataset_cls=self.dataset_cls,
                dataset_config=self.config.data,
            )
            output: AgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)
            return await self._agent_loop_postprocess(output, **kwargs)

    async def _agent_loop_postprocess(self, output, **kwargs) -> _InternalAgentLoopOutput:
        """Perform post-processing operations on the output of each individual agent loop."""
        output.extra_fields["raw_prompt"] = kwargs["raw_prompt"]

        # Some AgentLoop may have already computed the reward score, e.g SWE-agent.

        # NOTE: consistent with the legacy batch version of generate_sequences that existed in the
        # deprecated vLLM SPMD rollout implementation.
        # prompt_ids: left padded with zeros (e.g., [0,0,0,0,1,2,3,4])
        # response_ids: right padded with zeros (e.g., [5,6,7,8,0,0,0,0])
        # input_ids: concatenation of prompt + response
        # Mask:
        # For example, if the prompt is [1,2,3,4] and the response is [5,6,7,(tool start)8,9(tool end),10,11,12]
        # - prompt_attention_mask: 0s for padding, 1s for tokens
        #   e.g., [0,0,0,0,1,1,1,1]
        # - response_attention_mask: 0s for padding, 1s for tokens
        #   e.g., [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]
        # attention_mask: concatenation of prompt_attention_mask and response_attention_mask
        #   e.g., [0,0,0,0,1,1,1,1(prompt),1,1,1,1,1,1,1,1,1,1,1,0,0,0,0(response)]
        # - response_mask: 1s for LLM generated tokens, 0 for tool response/padding tokens
        #   e.g., [1,1,1,1,1,1,1,(tool start),0,0(tool end),1,1,0,0,0,0]
        # - position_ids: sequential positions for tokens, starting at 0
        #   e.g., [0,0,0,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,0,0,0,0]

        # TODO: remove padding and use tensordict.
        self.tokenizer.padding_side = "left"
        prompt_output = self.tokenizer.pad(
            {"input_ids": output.prompt_ids},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if prompt_output["input_ids"].dim() == 1:
            prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
            prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

        self.tokenizer.padding_side = "right"
        response_output = self.tokenizer.pad(
            {"input_ids": output.response_ids},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if response_output["input_ids"].dim() == 1:
            response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
            response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

        response_mask_output = self.tokenizer.pad(
            {"input_ids": output.response_mask},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=False,
        )
        if response_mask_output["input_ids"].dim() == 1:
            response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

        response_logprobs = None
        if output.response_logprobs is not None:
            pad_size = self.config.actor_rollout_ref.rollout.response_length - len(output.response_logprobs)
            response_logprobs = torch.tensor(output.response_logprobs + [0.0] * pad_size).unsqueeze(0)

        response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
        attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
        input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

        routed_experts = None
        if output.routed_experts is not None:
            total_length = input_ids.shape[1]
            length, layer_num, topk_num = output.routed_experts.shape
            experts_tensor = torch.from_numpy(output.routed_experts)
            routed_experts = torch.zeros(1, total_length, layer_num, topk_num, dtype=experts_tensor.dtype)

            # Calculate start position: left padding means original prompt starts at the end
            start_pos = prompt_output["input_ids"].shape[1] - len(output.prompt_ids)
            end_pos = min(start_pos + length, total_length)

            # Add boundary checks for robustness
            if start_pos < 0 or end_pos > total_length:
                raise ValueError(
                    f"Invalid position range: start_pos={start_pos}, end_pos={end_pos}, total_length={total_length}"
                )

            routed_experts[:, start_pos:end_pos] = experts_tensor.unsqueeze(0)

        multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
        position_ids = self._compute_position_ids(input_ids, attention_mask, multi_modal_inputs)
        await self._compute_score(
            output,
            prompts=prompt_output["input_ids"],
            responses=response_output["input_ids"],
            attention_mask=attention_mask,
            input_ids=input_ids,
            position_ids=position_ids,
            kwargs=kwargs,
        )

        return _InternalAgentLoopOutput(
            prompt_ids=prompt_output["input_ids"],
            response_ids=response_output["input_ids"],
            input_ids=input_ids,
            position_ids=position_ids,
            response_mask=response_mask,
            attention_mask=attention_mask,
            response_logprobs=response_logprobs,
            routed_experts=routed_experts,
            multi_modal_inputs=multi_modal_inputs,
            multi_modal_data=output.multi_modal_data,
            reward_score=output.reward_score,
            num_turns=output.num_turns,
            metrics=output.metrics,
            extra_fields=output.extra_fields,
        )

    def _compute_multi_modal_inputs(self, output, input_ids) -> dict[str, torch.Tensor]:
        """Compute multi-modal inputs with image and video."""
        multi_modal_inputs = {}
        if self.processor is None:
            return multi_modal_inputs

        images = output.multi_modal_data.get("images")
        videos = output.multi_modal_data.get("videos")
        # split the videos and according metadatas
        if videos is not None:
            videos, video_metadatas = zip(*videos, strict=False)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None
        current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
        multi_modal_inputs = self.processor(
            text=[current_text],
            images=images,
            videos=videos,
            video_metadatas=video_metadatas,
            return_tensors="pt",
            do_sample_frames=False,
        )
        multi_modal_inputs.pop("input_ids", None)
        multi_modal_inputs.pop("attention_mask", None)

        # We must use dict(multi_modal_inputs) to convert BatchFeature values to a new dict
        # because np.array() only keeps the keys for BatchFeature.
        multi_modal_inputs = dict(multi_modal_inputs.convert_to_tensors("pt"))
        return multi_modal_inputs

    def _compute_position_ids(self, input_ids, attention_mask, multi_modal_inputs) -> torch.Tensor:
        """Compute position ids for multi-modal inputs."""
        if self.processor is None:
            return compute_position_id_with_mask(attention_mask)  # (1, seq_len)

        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        video_grid_thw = multi_modal_inputs.get("video_grid_thw")

        # Model's get_rope_index has been dynamically bind to the processor.
        vision_position_ids, _ = self.processor.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        vision_position_ids = vision_position_ids.transpose(0, 1)  # (3, 1, seq_len) => (1, 3, seq_len)

        valid_mask = attention_mask[0].bool()
        text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
        text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
        text_position_ids = text_position_ids.unsqueeze(0)
        position_ids = torch.cat((text_position_ids, vision_position_ids), dim=1)  # (1, 4, seq_length)
        return position_ids

    async def _compute_score(self, output, prompts, responses, attention_mask, input_ids, position_ids, kwargs):
        """Compute reward score for single sample."""
        enable_async_reward = (
            self.reward_router_address is not None and self.config.reward_model.enable_resource_pool
        ) or not self.config.reward_model.enable

        if output.reward_score is None and enable_async_reward and self.use_reward_loop:
            batch = TensorDict(
                {
                    "prompts": prompts,  # [1, prompt_length]
                    "responses": responses,  # [1, response_length]
                    "attention_mask": attention_mask,  # [1, prompt_length + response_length]
                    "input_ids": input_ids,  # [1, prompt_length + response_length]
                    "position_ids": position_ids,
                },
                batch_size=1,
            )
            non_tensor_batch = {
                **{k: np.array([v]) for k, v in kwargs.items()},
                "__num_turns__": np.array([output.num_turns]),
                "tool_extra_fields": np.array([output.extra_fields], dtype=object),
            }

            data = DataProto(
                batch=batch,
                non_tensor_batch=non_tensor_batch,
            )
            result = await self.reward_loop_worker.compute_score.remote(data)
            output.reward_score = result["reward_score"]
            output.extra_fields["reward_extra_info"] = result["reward_extra_info"]

    def _postprocess(self, inputs: list[_InternalAgentLoopOutput]) -> DataProto:
        """Process the padded outputs from _run_agent_loop and combine them into a batch."""
        # Convert lists back to tensors and stack them to create a batch.
        prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
        response_ids = torch.cat([input.response_ids for input in inputs], dim=0)
        response_mask = torch.cat([input.response_mask for input in inputs], dim=0)
        attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
        input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
        position_ids = torch.cat([input.position_ids for input in inputs], dim=0)
        optional_outputs = {}
        if inputs[0].response_logprobs is not None:
            optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)
        if inputs[0].routed_experts is not None:
            optional_outputs["routed_experts"] = torch.cat([input.routed_experts for input in inputs], dim=0)

        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_ids,  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                # position_ids: [bsz, 3, prompt_length + response_length] or [bsz, prompt_length + response_length]
                "position_ids": position_ids,
                **optional_outputs,
            },
            batch_size=len(inputs),
        )

        scores = [input.reward_score for input in inputs]
        if all(score is not None for score in scores):
            prompt_length = prompt_ids.size(1)
            response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(scores, dtype=torch.float32)
            batch["rm_scores"] = rm_scores

        non_tensor_batch = {
            "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
        }

        # add reward_extra_info to non_tensor_batch
        reward_extra_infos = [input.extra_fields.get("reward_extra_info", {}) for input in inputs]
        reward_extra_keys = list(reward_extra_infos[0].keys())
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

        # Add multi_modal_inputs to non_tensor_batch if any samples have them
        multi_modal_inputs_list = [input.multi_modal_inputs for input in inputs]
        if any(mmi is not None for mmi in multi_modal_inputs_list):
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        metrics = [input.metrics.model_dump() for input in inputs]
        # Collect extra fields from all inputs and convert them to np.ndarray
        extra_fields = {}
        all_keys = set(key for input_item in inputs for key in input_item.extra_fields)
        for key in all_keys:
            temp_arr = np.empty(len(inputs), dtype=object)
            temp_arr[:] = [input.extra_fields.get(key) for input in inputs]
            extra_fields[key] = temp_arr

        non_tensor_batch.update(extra_fields)
        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info={"metrics": metrics, "reward_extra_keys": reward_extra_keys},
        )

    def create_transferqueue_client(
        self,
    ):
        """Create a client for data system (TransferQueue)."""
        from verl.single_controller.ray.base import get_random_string
        from verl.utils.transferqueue_utils import create_transferqueue_client

        client_name = get_random_string(length=6)

        self.tq_client = create_transferqueue_client(
            client_id=f"AgentLoopWorker_{client_name}",
            config=self.config.transfer_queue,
        )


@ray.remote
class AgentLoopWorker(AgentLoopWorkerBase):
    """Agent loop worker takes a batch of messages and run each message in an agent loop."""

    def __init__(
        self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], reward_router_address: str = None
    ):
        """Initialize agent loop manager.
        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            reward_router_address (str): reward router address.
        """
        super().__init__(config, server_handles, reward_router_address)


async def get_trajectory_info(step, index, validate):
    """Get trajectory info.

    Args:
        step (int): global steps in the trainer.
        index (list): form datastore extra_info.index column.
        validate (bool): whether is a validate step.

    Returns:
        list: trajectory.
    """
    trajectory_info = []
    rollout_n = 0
    for i in range(len(index)):
        if i > 0 and index[i - 1] == index[i]:
            rollout_n += 1
        else:
            rollout_n = 0
        trajectory_info.append({"step": step, "sample_index": index[i], "rollout_n": rollout_n, "validate": validate})
    return trajectory_info


class AgentLoopManager:
    """Agent loop manager that manages a group of agent loop workers."""

    def __init__(
        self, config: DictConfig, worker_group: RayWorkerGroup = None, rm_resource_pool: RayResourcePool = None
    ):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): trainer config.
            worker_group (RayWorkerGroup): ActorRolloutRef worker group for hybrid mode; None for standalone mode.
            rm_resource_pool (RayResourcePool): Resource pool for reward model (Standalone mode).
        """
        self.config = config
        self.worker_group = worker_group
        self.reward_model_manager = None
        self.reward_router_address = None
        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            from verl.experimental.reward_loop import RewardModelManager

            self.reward_model_manager = RewardModelManager(config.reward_model, rm_resource_pool)
            self.reward_router_address = self.reward_model_manager.get_router_address()

        # for recipe to change
        if not hasattr(self, "rollout_replica_class"):
            self.rollout_replica_class = get_rollout_replica_class(self.config.actor_rollout_ref.rollout.name)
        if not hasattr(self, "agent_loop_workers_class"):
            self.agent_loop_workers_class = AgentLoopWorker

        # --- length tracker + allocation policy ---
        from verl.utils.allocation_policy import create_allocation_policy
        from verl.utils.rollout_length_tracker import RolloutLengthTracker

        rollout_config = config.actor_rollout_ref.rollout
        tail_amp = OmegaConf.select(rollout_config, "tail_amplification", default=2.0)
        self.length_tracker = RolloutLengthTracker(
            l_max=rollout_config.response_length,
            ema_alpha=OmegaConf.select(rollout_config, "ema_alpha", default=0.3),
            tail_amplification=tail_amp,
        )
        policy_name = OmegaConf.select(rollout_config, "allocation_policy", default="default")
        self.allocation_policy = create_allocation_policy(
            policy_name=policy_name,
            G=rollout_config.n,
            l_max=rollout_config.response_length,
            tail_amplification=OmegaConf.select(rollout_config, "tail_amplification", default=2.0),
        )
        logger.info("allocation_policy=%s, length_tracker l_max=%d", policy_name, rollout_config.response_length)

        # ── Adaptive-γ cluster-consensus elevation/lowering state ────────
        # Trainer-side single-source controller replaces per-engine wake_up
        # elevation + MAX-sync propagation. Cluster-aggregate AR (token-
        # weighted: Σ accepted / Σ drafted) is the decision metric. Two-sided
        # (paper Alg. 1): elevate when AR ≥ VLLM_GAMMA_AR_THRESHOLD (α_up,
        # default 0.94), lower when AR ≤ VLLM_GAMMA_AR_THRESHOLD_LOWER (α_down,
        # default 0.85), each sustained for VLLM_GAMMA_PERSISTENCE consecutive
        # training rollouts (default 2); force_set_gamma is broadcast
        # uniformly. Ladder + thresholds come from RolloutConfig/env.
        _ladder_str = OmegaConf.select(rollout_config, "spec_gamma_ladder", default=None)
        if _ladder_str:
            try:
                self._gamma_ladder: list[int] | None = [
                    int(x.strip()) for x in _ladder_str.split(",") if x.strip()
                ]
            except (ValueError, AttributeError):
                logger.warning(
                    "Adaptive-γ: failed to parse spec_gamma_ladder=%r; "
                    "trainer-side elevation disabled.", _ladder_str,
                )
                self._gamma_ladder = None
        else:
            self._gamma_ladder = None
        self._ar_threshold: float = float(
            os.environ.get("VLLM_GAMMA_AR_THRESHOLD", "0.94")
        )
        self._ar_threshold_lower: float = float(
            os.environ.get("VLLM_GAMMA_AR_THRESHOLD_LOWER", "0.85")
        )
        self._persistence_n: int = max(
            1, int(os.environ.get("VLLM_GAMMA_PERSISTENCE", "2"))
        )
        self._consecutive_crossings: int = 0
        self._consecutive_low_crossings: int = 0
        if self._gamma_ladder:
            logger.info(
                "Adaptive-γ (trainer-side cluster consensus): ladder=%s, "
                "AR_THRESHOLD=%.3f, AR_THRESHOLD_LOWER=%.3f, PERSISTENCE_N=%d",
                self._gamma_ladder, self._ar_threshold,
                self._ar_threshold_lower, self._persistence_n,
            )

        self._initialize_llm_servers()
        self._init_agent_loop_workers()

        # Initially we're in sleep mode.
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

    def _initialize_llm_servers(self):
        rollout_world_size = (
            self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
            * self.config.actor_rollout_ref.rollout.data_parallel_size
            * self.config.actor_rollout_ref.rollout.pipeline_model_parallel_size
        )
        world_size = (
            self.worker_group.world_size
            if self.worker_group
            else self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        )
        num_replicas = world_size // rollout_world_size

        rollout_config = self.config.actor_rollout_ref.rollout
        model_config = self.config.actor_rollout_ref.model
        self.rollout_replicas = [
            self.rollout_replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=self.config.trainer.n_gpus_per_node,
            )
            for replica_rank in range(num_replicas)
        ]
        if self.worker_group:
            self._run_all([server.init_hybrid(self.worker_group) for server in self.rollout_replicas])
        else:
            self._run_all([server.init_standalone() for server in self.rollout_replicas])
        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]

        print(f"AgentLoopManager: {self.server_addresses}")

        # Update Prometheus configuration with server addresses
        if rollout_config.prometheus.enable:
            if rollout_config.disable_log_stats:
                raise ValueError("PROMETHEUS needs disable_log_stats==False, but it is currently True.")
            update_prometheus_config(rollout_config.prometheus, self.server_addresses)

    def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            # Round-robin scheduling over the all nodes
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(
                self.agent_loop_workers_class.options(
                    name=f"agent_loop_worker_{i}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(self.config, self.server_handles, self.reward_router_address)
            )

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Thin wrapper around _generate_sequences_impl that brackets the
        rollout with adaptive-γ validation-mode flag toggles.

        When the input batch carries meta_info["validate"]=True (i.e. the
        call originates from _validate()), we set validate-mode True on
        every rollout engine before wake_up() and clear it in a finally
        block afterwards. This keeps val rollouts from contaminating the
        engine-side adaptive-γ state (elevation + _rollout_* accumulator).
        Training rollout N's data is preserved through the val rollout
        and consumed at training rollout N+1's wake_up.
        """
        validate = prompts.meta_info.get("validate", False)
        try:
            if validate:
                # Enable inside try so that a partial-success failure mid-
                # fanout still triggers the clear path in finally — without
                # this, some replicas could be left in validate-mode on a
                # raise, leaking into the next training rollout.
                self.set_validate_mode(True)
            return self._generate_sequences_impl(prompts)
        finally:
            if validate:
                # Always clear, even on exception, to avoid leaking val
                # mode into subsequent training rollouts.
                self.set_validate_mode(False)

    def _broadcast_set_gamma(self, target_gamma: int, current_gamma: int) -> None:
        """Broadcast an adaptive-γ transition (elevation or lowering) to all
        rollout servers with fail-loud semantics and best-effort rollback.

        force_set_gamma is exception-propagating at every layer (async_llm +
        server) and must return ``target_gamma`` on every server. Any
        deviation → attempt rollback to ``current_gamma`` on the servers that
        already mutated, then re-raise to abort training (a split-γ cluster
        silently compounds across subsequent rollouts).
        """
        try:
            results = ray.get([
                h.force_set_gamma.remote(target_gamma)
                for h in self.server_handles
            ])
            bad = [(i, r) for i, r in enumerate(results) if r != target_gamma]
            if bad:
                raise RuntimeError(
                    f"force_set_gamma partial failure: expected "
                    f"{target_gamma}, got {results}"
                )
        except Exception as broadcast_err:
            logger.error(
                "Adaptive-γ broadcast failed: %s. Attempting rollback to "
                "γ=%d on all servers (some may have already advanced).",
                broadcast_err, current_gamma,
            )
            try:
                rb_results = ray.get([
                    h.force_set_gamma.remote(current_gamma)
                    for h in self.server_handles
                ])
                rb_bad = [
                    (i, r) for i, r in enumerate(rb_results)
                    if r != current_gamma
                ]
                if rb_bad:
                    logger.error(
                        "Adaptive-γ ROLLBACK PARTIAL: expected γ=%d on all "
                        "servers, got %s. Cluster state is now INCONSISTENT "
                        "— do NOT resume from current checkpoint without "
                        "manual verification of γ per engine.",
                        current_gamma, rb_results,
                    )
                else:
                    logger.error(
                        "Adaptive-γ rollback to γ=%d completed (all %d "
                        "servers confirmed); aborting training.",
                        current_gamma, len(self.server_handles),
                    )
            except Exception as rb_err:
                logger.error(
                    "Adaptive-γ ROLLBACK ALSO FAILED: %s. Cluster state is "
                    "now INCONSISTENT — do NOT resume from current "
                    "checkpoint without manual verification of γ per "
                    "engine.", rb_err,
                )
            raise

    def _generate_sequences_impl(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

        The input batch is already expanded (prompts × rollout.n, interleaved)
        so that consecutive rollout.n entries belong to the same prompt.

        AllocationPolicy determines prompt→GPU mapping. target_gpu is attached
        to each request via non_tensor_batch so that _choose_server() respects it.

        Args:
            prompts (DataProto): Input batch (expanded by rollout.n).

        Returns:
            DataProto: Output batch.
        """

        # Fix for Issue #4147: Always call wake_up() to ensure weight sync
        # The wake_up()/sleep() methods internally check free_cache_engine
        self.wake_up()
        if self.reward_model_manager:
            self.reward_model_manager.wake_up()

        # --- prompt→GPU allocation ---
        from verl.utils.allocation_policy import PromptInfo

        rollout_n = self.config.actor_rollout_ref.rollout.n
        num_gpus = len(self.server_handles)
        total_requests = len(prompts)
        num_unique_prompts = total_requests // rollout_n

        prompt_keys = prompts.non_tensor_batch.get("index", np.arange(total_requests))
        unique_prompt_indices = list(range(0, total_requests, rollout_n))
        prompt_infos = [
            PromptInfo(key=int(prompt_keys[i]), prompt_length=0)
            for i in unique_prompt_indices
        ]

        assignment = self.allocation_policy.allocate(prompt_infos, num_gpus)

        if assignment:
            prompt_idx_to_gpu: dict[int, int] = {}
            for gpu_id, indices in assignment.items():
                for idx in indices:
                    prompt_idx_to_gpu[idx] = gpu_id

            target_gpus = np.empty(total_requests, dtype=np.int64)
            for req_idx in range(total_requests):
                prompt_idx = req_idx // rollout_n
                target_gpus[req_idx] = prompt_idx_to_gpu[prompt_idx]
        else:
            # default to use default LB
            target_gpus = np.full(total_requests, -1, dtype=np.int64)

        prompts.non_tensor_batch["target_gpu"] = target_gpus

        chunkes = prompts.chunk(len(self.agent_loop_workers))
        outputs = ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )
        # --- merge per-GPU stats from all workers ---
        from verl.utils.rollout_profiler import GpuRolloutSummary

        merged_gpu_lengths: dict[int, list[int]] = defaultdict(list)
        merged_gpu_makespans: dict[int, float] = defaultdict(float)
        for out in outputs:
            worker_gpu_stats = out.meta_info.pop("per_gpu_response_lengths", {})
            for gpu_idx, stats in worker_gpu_stats.items():
                merged_gpu_lengths[gpu_idx].extend(stats["lengths"])
                merged_gpu_makespans[gpu_idx] = max(merged_gpu_makespans[gpu_idx], stats["makespan_sec"])

        # Collect SD stats ONCE at manager level (not per-worker) to avoid race.
        # All workers share the same server_handles, so per-worker get+reset
        # causes stats to be cleared before other workers can read them.
        merged_sd_stats: dict[int, dict] = {}
        if self.server_handles:
            sd_futures = {}
            for gpu_idx, handle in enumerate(self.server_handles):
                sd_futures[gpu_idx] = handle.get_sd_stats.remote()
            # Fail-loud on decision-input collection: if any server's
            # get_sd_stats fails, the cluster-level evidence for this
            # rollout is incomplete. Silencing this to a warning would
            # let the persistence gate drift out of invariant ("N
            # consecutive crossings") by carrying a counter across a
            # rollout we never actually observed. Consistent with the
            # fail-loud policy applied to force_set_gamma and
            # reset_sd_stats below.
            sd_results = ray.get(list(sd_futures.values()))
            for gpu_idx, sd_stats in zip(sd_futures.keys(), sd_results):
                if sd_stats:
                    merged_sd_stats[gpu_idx] = sd_stats

            # ── Adaptive-γ cluster-consensus elevation ───────────────────
            # Single-source-of-truth elevation driven by cluster-aggregate
            # AR. Replaces the old per-engine wake_up elevation + MAX-sync
            # propagation, which allowed a single outlier engine (1/8) to
            # trigger a cluster-wide γ bump via max-propagation.
            #
            # cluster_ar = Σ accepted_tokens / Σ draft_tokens across all
            # engines — identical to the token-weighted `acceptance_rate`
            # already logged to wandb. γ elevates iff cluster_ar ≥ α_up and
            # lowers iff cluster_ar ≤ α_down, each sustained for N consecutive
            # training rollouts (persistence gate, N=2 default).
            #
            # Val rollouts are excluded: meta_info["validate"]=True
            # short-circuits the decision so val's greedy-decode AR does
            # not contaminate training-driven elevation.
            validate = prompts.meta_info.get("validate", False)
            if (
                not validate
                and merged_sd_stats
                and self._gamma_ladder
            ):
                total_accepted = sum(
                    sd.get("total_accepted_tokens", 0)
                    for sd in merged_sd_stats.values()
                )
                total_drafts = sum(
                    sd.get("total_draft_tokens", 0)
                    for sd in merged_sd_stats.values()
                )
                gammas_seen = [
                    sd.get("sd_current_gamma", -1)
                    for sd in merged_sd_stats.values()
                    if sd.get("sd_current_gamma", -1) >= 0
                ]
                if total_drafts > 0 and gammas_seen:
                    # Split-γ abort: after this refactor all engines MUST
                    # be at the same γ. Any drift indicates either a prior
                    # partial force_set_gamma failure that was silently
                    # rolled back, a leaked per-engine elevation path, or
                    # a Ray actor restart. Taking max() would paper over
                    # the inconsistency and let the next rollout compound
                    # the bug; abort instead so the operator sees it.
                    unique_gammas = set(gammas_seen)
                    if len(unique_gammas) > 1:
                        raise RuntimeError(
                            f"Adaptive-γ: cluster γ drift detected "
                            f"({sorted(unique_gammas)} across "
                            f"{len(merged_sd_stats)} engines). "
                            f"Trainer-side consensus guarantees uniform γ; "
                            f"drift here is a bug — refusing to elevate."
                        )
                    cluster_ar = total_accepted / total_drafts
                    current_gamma = gammas_seen[0]
                    try:
                        idx = self._gamma_ladder.index(current_gamma)
                        next_gamma = (
                            self._gamma_ladder[idx + 1]
                            if (idx + 1) < len(self._gamma_ladder)
                            else None
                        )
                        prev_gamma = (
                            self._gamma_ladder[idx - 1] if idx > 0 else None
                        )
                    except ValueError:
                        next_gamma = None
                        prev_gamma = None

                    # Two-sided adaptive-γ (paper Alg. 1). ar = (MAL-1)/γ, so
                    # the AR thresholds ARE the paper's α_up / α_down
                    # (τ ≥ 1+γ·α ⇔ ar ≥ α): elevate when cluster_ar ≥ α_up
                    # (VLLM_GAMMA_AR_THRESHOLD), lower when cluster_ar ≤ α_down
                    # (VLLM_GAMMA_AR_THRESHOLD_LOWER). α_down < α_up makes the
                    # two mutually exclusive; each direction has its own
                    # persistence counter and drives γ through the same
                    # fail-loud force_set_gamma broadcast (_broadcast_set_gamma).
                    if (
                        next_gamma is not None
                        and cluster_ar >= self._ar_threshold
                    ):
                        self._consecutive_crossings += 1
                        self._consecutive_low_crossings = 0
                        if self._consecutive_crossings >= self._persistence_n:
                            logger.warning(
                                "Adaptive-γ cluster ELEVATE: γ=%d→%d "
                                "cluster_ar=%.4f ≥ %.4f "
                                "(%d/%d consecutive crossings)",
                                current_gamma, next_gamma,
                                cluster_ar, self._ar_threshold,
                                self._consecutive_crossings,
                                self._persistence_n,
                            )
                            self._broadcast_set_gamma(next_gamma, current_gamma)
                            self._consecutive_crossings = 0
                    elif (
                        prev_gamma is not None
                        and cluster_ar <= self._ar_threshold_lower
                    ):
                        self._consecutive_low_crossings += 1
                        self._consecutive_crossings = 0
                        if self._consecutive_low_crossings >= self._persistence_n:
                            logger.warning(
                                "Adaptive-γ cluster LOWER: γ=%d→%d "
                                "cluster_ar=%.4f ≤ %.4f "
                                "(%d/%d consecutive crossings)",
                                current_gamma, prev_gamma,
                                cluster_ar, self._ar_threshold_lower,
                                self._consecutive_low_crossings,
                                self._persistence_n,
                            )
                            self._broadcast_set_gamma(prev_gamma, current_gamma)
                            self._consecutive_low_crossings = 0
                    else:
                        # Reset both persistence counters on any rollout that
                        # is neither an elevation nor a lowering candidate.
                        self._consecutive_crossings = 0
                        self._consecutive_low_crossings = 0

            # Reset accumulators for next step. Fail-loud: if any server
            # fails to reset, stale tokens from THIS rollout will leak
            # into NEXT rollout's cluster_ar computation and contaminate
            # the persistence-gate decision. Silencing this to a warning
            # lets the bug compound; abort instead so the operator sees
            # the failure and can investigate.
            ray.get([h.reset_sd_stats.remote() for h in self.server_handles])

        gpu_summaries = []
        for gpu_idx in sorted(merged_gpu_lengths.keys()):
            lens = merged_gpu_lengths[gpu_idx]
            sd = merged_sd_stats.get(gpu_idx, {})
            gpu_summaries.append(GpuRolloutSummary(
                gpu_id=gpu_idx,
                makespan_sec=round(merged_gpu_makespans[gpu_idx], 3),
                n_requests=len(lens),
                n_completed=sum(1 for l in lens if l > 0),
                mean_response_len=round(sum(lens) / len(lens), 1) if lens else 0.0,
                max_response_len=max(lens) if lens else 0,
                sd_activated_at_tick=sd.get("sd_toggle_tick", -1) if sd.get("sd_toggled", False) else -1,
                sd_activated_at_batch=sd.get("sd_toggle_batch", -1) if sd.get("sd_toggled", False) else -1,
                acceptance_rate=sd.get("acceptance_rate", 0.0),
                total_draft_tokens=sd.get("total_draft_tokens", 0),
                total_accepted_tokens=sd.get("total_accepted_tokens", 0),
                num_drafts=sd.get("num_drafts", 0),
                accepted_per_pos=sd.get("accepted_per_pos", []),
                toggle_L_accept_used=sd.get("toggle_L_accept_used", 0.0),
                sd_current_gamma=sd.get("sd_current_gamma", -1),
                sd_elevation_to_gamma=sd.get("sd_elevation_to_gamma", -1),
                sd_ar_observed=sd.get("sd_ar_observed", -1.0),
                sd_elevated_at_step=sd.get("sd_elevated_at_step", -1),
            ))

        output = DataProto.concat(outputs)
        # IMPORTANT: sleep() must happen AFTER SD stats collection above,
        # because sleep() transitions the engine to trainer mode.
        self.sleep()
        if self.reward_model_manager:
            self.reward_model_manager.sleep()

        # calculate performance metrics
        metrics = [output.meta_info.pop("metrics") for output in outputs]  # List[List[Dict[str, str]]]
        timing = self._performance_metrics(metrics, output)

        # --- record response lengths for length tracker ---
        epoch = prompts.meta_info.get("epoch", 0)
        prompt_keys = output.non_tensor_batch.get("index", np.arange(len(output)))
        prompt_length = output.batch["prompts"].shape[1]
        response_lengths_all = output.batch["attention_mask"][:, prompt_length:].sum(dim=1).numpy()
        self.length_tracker.record_step(prompt_keys, response_lengths_all, epoch=epoch)

        # Inject requantize overhead from wake_up() into timing
        requantize_sec = getattr(self, '_last_requantize_sec', 0.0)
        if requantize_sec > 0:
            timing["requantize"] = requantize_sec
            self._last_requantize_sec = 0.0

        output.meta_info = {
            "timing": timing,
            "gpu_rollout_summaries": gpu_summaries,
            **outputs[0].meta_info,
        }
        return output

    def _performance_metrics(self, metrics: list[list[dict[str, str]]], output: DataProto) -> dict[str, float]:
        timing = {}
        t_generate_sequences = np.array([metric["generate_sequences"] for chunk in metrics for metric in chunk])
        t_tool_calls = np.array([metric["tool_calls"] for chunk in metrics for metric in chunk])
        timing["agent_loop/generate_sequences/min"] = t_generate_sequences.min()
        timing["agent_loop/generate_sequences/max"] = t_generate_sequences.max()
        timing["agent_loop/generate_sequences/mean"] = t_generate_sequences.mean()
        timing["agent_loop/tool_calls/min"] = t_tool_calls.min()
        timing["agent_loop/tool_calls/max"] = t_tool_calls.max()
        timing["agent_loop/tool_calls/mean"] = t_tool_calls.mean()

        # SD out-of-top-N logprob fallback telemetry: structurally always 0
        # post-rejection_sampler.py:154 fix (the -20.0 silent fallback path was
        # replaced with fail-loud RuntimeError in vllm_async_server.py). Field
        # is retained on TokenOutput for schema compatibility with checkpoints
        # from older runs, but emission to wandb is suppressed to avoid
        # always-zero panels polluting historical run comparisons.

        # SD first-entry invariant sanity check (per-request count of
        # positions where dict's first entry token_id != response token_id).
        # Should be ~0 if the SD invariant holds.
        first_entry_mismatch_counts = np.array(
            [metric.get("first_entry_mismatch_count", 0) for chunk in metrics for metric in chunk]
        )
        timing["agent_loop/first_entry_mismatch_count/sum"] = float(first_entry_mismatch_counts.sum())
        timing["agent_loop/first_entry_mismatch_count/mean"] = float(first_entry_mismatch_counts.mean())
        timing["agent_loop/first_entry_mismatch_count/max"] = float(first_entry_mismatch_counts.max())
        timing["agent_loop/first_entry_mismatch_count/nonzero_request_frac"] = float(
            (first_entry_mismatch_counts > 0).mean()
        )

        # batch sequence generation is bounded by the slowest sample
        slowest = np.argmax(t_generate_sequences + t_tool_calls)
        attention_mask = output.batch["attention_mask"][slowest]
        prompt_length = output.batch["prompts"].shape[1]
        timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
        timing["agent_loop/slowest/tool_calls"] = t_tool_calls[slowest]
        timing["agent_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
        timing["agent_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()

        return timing

    def wake_up(self):
        """Wake up all rollout replica instances. Returns max requantize_sec across replicas."""
        async def _wake_all():
            return await asyncio.gather(*[replica.wake_up() for replica in self.rollout_replicas])
        results = asyncio.run(_wake_all())
        self._last_requantize_sec = max(
            (r if isinstance(r, (int, float)) and not isinstance(r, bool) else 0.0) for r in results
        )

    def sleep(self):
        """Sleep all rollout replica instances."""
        self._run_all([replica.sleep() for replica in self.rollout_replicas])

    def clear_kv_cache(self):
        """Clear all rollout kv cache, but don`t sleep."""
        self._run_all([replica.clear_kv_cache() for replica in self.rollout_replicas])

    def set_validate_mode(self, mode: bool):
        """Propagate adaptive-γ validation-mode flag to all replicas.

        When True (set before a val rollout), engine-side wake_up skips
        elevation + per-rollout accumulator reset, and step() skips per-
        rollout SD accumulation. Must be cleared (False) in a finally
        block after the val rollout so the next training rollout's
        wake_up can elevate from training rollout N's preserved data.
        """
        self._run_all([
            replica.set_validate_mode(mode) for replica in self.rollout_replicas
        ])

    def _run_all(self, tasks: list[asyncio.Task]):
        async def run_all():
            await asyncio.gather(*tasks)

        asyncio.run(run_all())
