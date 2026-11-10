# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from transformers.models.nemotron3_5_asr.generation_nemotron3_5_asr import (
    Nemotron3_5AsrRNNTDecoderCache,
)
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup

from .model import Nemotron3_5AsrDecodeState


class Nemotron3_5AsrStreamingState:
    """Caches and pending acoustic frames for one realtime session."""

    def __init__(
        self,
        model: torch.nn.Module,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        config = model.config
        lookahead = int(
            getattr(config.encoder_config, "default_num_lookahead_tokens", 13)
        )
        self.frame_capacity = max(2 * (lookahead + 2), 4)
        self.frame_buffer = torch.empty(
            (self.frame_capacity, config.decoder_hidden_size),
            dtype=dtype,
            device=device,
        )
        self.decode_state = Nemotron3_5AsrDecodeState(
            encoder_frames=self.frame_buffer[:0],
            decoder_cache=Nemotron3_5AsrRNNTDecoderCache(config),
            last_token_id=config.blank_token_id,
        )
        self.past_key_values: object | None = None
        self.padding_cache: object | None = None
        self.encoded_features = 0
        self.num_features = 0


def _get_mm_scalar(feature: object, key: str) -> object | None:
    data = getattr(feature, "data", None)
    if data is None:
        return None
    elem = data.get(key)
    if elem is None:
        return None
    value = elem.data
    if isinstance(value, torch.Tensor):
        return value.reshape(-1)[0].item()
    return value


def _feature_is_streaming(feature: object) -> bool:
    return bool(_get_mm_scalar(feature, "is_streaming") or False)


class Nemotron3_5AsrModelState(ModelState):
    """Keep RNNT acoustic and LSTM state isolated per request."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: torch.nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        assert encoder_cache is not None
        super().__init__(vllm_config, model, encoder_cache, device)
        budget = MultiModalBudget(vllm_config, MULTIMODAL_REGISTRY, enable_cache=False)
        # The shared encoder cache is released after prefill. RNNT needs the
        # acoustic frames until completion, so reserve accounted per-request slots.
        self.acoustic_states = torch.empty(
            (
                self.max_num_reqs,
                budget.mm_max_toks_per_item["audio"],
                self.model.config.decoder_hidden_size,
            ),
            dtype=self.dtype,
            device=device,
        )
        self.audio_hashes: dict[str, str] = {}
        self.request_indices: dict[str, int] = {}
        self.decode_states: dict[str, Nemotron3_5AsrDecodeState] = {}
        self.num_tokens_to_replay: dict[str, int] = {}
        self.streaming_states: dict[str, Nemotron3_5AsrStreamingState] = {}
        self.pending_streaming_states: dict[str, Nemotron3_5AsrStreamingState] = {}

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        audio_features = [
            feature
            for feature in new_req_data.mm_features
            if feature.modality == "audio"
        ]
        if not audio_features and new_req_data.req_id.startswith("_warmup_"):
            return

        if audio_features and _feature_is_streaming(audio_features[-1]):
            req_id = new_req_data.req_id
            state = self.pending_streaming_states.pop(req_id, None)
            if state is None:
                state = self.streaming_states.get(req_id)
            if state is None:
                state = Nemotron3_5AsrStreamingState(
                    self.model,
                    self.dtype,
                    self.device,
                )
            state.num_features = len(audio_features)
            self.streaming_states[req_id] = state
            self.request_indices[req_id] = req_index
            self.audio_hashes.pop(req_id, None)
            self.decode_states.pop(req_id, None)
            self.num_tokens_to_replay.pop(req_id, None)
            return

        if len(audio_features) != 1:
            raise ValueError("Nemotron 3.5 ASR expects one audio per request.")
        self.audio_hashes[new_req_data.req_id] = audio_features[0].identifier
        self.request_indices[new_req_data.req_id] = req_index
        assert new_req_data.prefill_token_ids is not None
        self.num_tokens_to_replay[new_req_data.req_id] = (
            len(new_req_data.prefill_token_ids) - new_req_data.prompt_len
        )

    def remove_request(self, req_id: str) -> None:
        if state := self.streaming_states.pop(req_id, None):
            self.pending_streaming_states[req_id] = state
        self.audio_hashes.pop(req_id, None)
        self.request_indices.pop(req_id, None)
        self.decode_states.pop(req_id, None)
        self.num_tokens_to_replay.pop(req_id, None)

    def apply_staged_writes(self) -> None:
        self.pending_streaming_states.clear()

    def prepare_inputs_embeds(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> None:
        non_streaming_encoder_inputs = {
            req_id: input_ids
            for req_id, input_ids in scheduled_encoder_inputs.items()
            if req_id not in self.streaming_states
        }
        if non_streaming_encoder_inputs:
            self.execute_mm_encoder(non_streaming_encoder_inputs)

        for req_id in input_batch.req_ids:
            state = self.streaming_states.get(req_id)
            while state is not None and state.encoded_features < state.num_features:
                self._encode_next_streaming_feature(req_id, state)
        return None

    def _encode_next_streaming_feature(
        self,
        req_id: str,
        state: Nemotron3_5AsrStreamingState,
    ) -> None:
        features = self.encoder_cache.mm_features.get(req_id, [])
        if state.encoded_features >= len(features):
            raise RuntimeError(
                f"Missing streaming audio features for request {req_id}."
            )

        feature = features[state.encoded_features]
        if feature.data is None:
            raise RuntimeError(
                f"Streaming audio feature {state.encoded_features} for "
                f"request {req_id} is no longer available."
            )
        input_features = feature.data["input_features"].data
        prompt_ids = feature.data["prompt_ids"].data
        lookahead = feature.data["num_lookahead_tokens"].data
        assert isinstance(input_features, torch.Tensor)
        if input_features.ndim == 2:
            input_features = input_features.unsqueeze(0)
        if isinstance(prompt_ids, torch.Tensor):
            prompt_ids = prompt_ids.reshape(-1)[:1]
        if isinstance(lookahead, torch.Tensor):
            lookahead = int(lookahead.reshape(-1)[0].item())

        frames, past_key_values, padding_cache = (
            self.model.audio_encoder.forward_streaming_chunk(
                input_features.to(device=self.device, dtype=self.dtype),
                prompt_ids,
                int(lookahead),
                past_key_values=state.past_key_values,
                padding_cache=state.padding_cache,
            )
        )
        self._append_streaming_frames(state, frames[0])
        state.past_key_values = past_key_values
        state.padding_cache = padding_cache
        state.encoded_features += 1

    @staticmethod
    def _append_streaming_frames(
        state: Nemotron3_5AsrStreamingState,
        frames: torch.Tensor,
    ) -> None:
        decode_state = state.decode_state
        pending = decode_state.encoder_frames.shape[0] - decode_state.frame_idx
        pending_frames = (
            decode_state.encoder_frames[decode_state.frame_idx :].clone()
            if pending
            else None
        )
        end = pending + frames.shape[0]
        if end > state.frame_capacity:
            state.frame_capacity = max(end, state.frame_capacity * 2)
            state.frame_buffer = torch.empty(
                (state.frame_capacity, frames.shape[1]),
                dtype=frames.dtype,
                device=frames.device,
            )
        if pending_frames is not None:
            state.frame_buffer[:pending].copy_(pending_frames)
        state.frame_buffer[pending:end].copy_(frames)
        decode_state.encoder_frames = state.frame_buffer[:end]
        decode_state.frame_idx = 0

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, Any]:
        self.pending_streaming_states.clear()
        # Eager dummy runs use synthetic request IDs and mark every token as
        # padding. Profile the RNNT path without touching live request state.
        if not any(
            req_id in self.request_indices for req_id in input_batch.req_ids
        ) and bool(input_batch.is_padding[0].item()):
            dummy_inputs = self.prepare_dummy_inputs(
                input_batch.num_reqs, input_batch.num_tokens
            )
            dummy_inputs["query_end_positions"] = input_batch.query_start_loc_np[
                1 : input_batch.num_reqs + 1
            ].tolist()
            return dummy_inputs

        states: list[Nemotron3_5AsrDecodeState | None] = []
        for req_id in input_batch.req_ids:
            streaming_state = self.streaming_states.get(req_id)
            if streaming_state is not None:
                states.append(streaming_state.decode_state)
                continue

            state = self.decode_states.get(req_id)
            if state is None and req_id in self.audio_hashes:
                audio_hash = self.audio_hashes[req_id]
                encoder_frames = self.encoder_cache.encoder_outputs[audio_hash]
                frames = self.acoustic_states[
                    self.request_indices[req_id], : encoder_frames.shape[0]
                ]
                frames.copy_(encoder_frames)
                state = Nemotron3_5AsrDecodeState(
                    encoder_frames=frames,
                    decoder_cache=Nemotron3_5AsrRNNTDecoderCache(self.model.config),
                    last_token_id=self.model.config.blank_token_id,
                    num_tokens_to_replay=self.num_tokens_to_replay.pop(req_id),
                )
                self.decode_states[req_id] = state
            states.append(state)

        ready = (
            input_batch.num_computed_tokens_np + input_batch.num_scheduled_tokens
            >= input_batch.prefill_len_np
        )
        for index, req_id in enumerate(input_batch.req_ids):
            streaming_state = self.streaming_states.get(req_id)
            if (
                streaming_state is not None
                and streaming_state.encoded_features < streaming_state.num_features
            ):
                ready[index] = False
        return {
            "decode_states": states,
            "query_end_positions": input_batch.query_start_loc_np[
                1 : input_batch.num_reqs + 1
            ].tolist(),
            "decode_ready": ready.tolist(),
        }

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        frames = torch.zeros(
            (1, self.model.config.decoder_hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        return {
            "decode_states": [
                Nemotron3_5AsrDecodeState(
                    encoder_frames=frames,
                    decoder_cache=Nemotron3_5AsrRNNTDecoderCache(self.model.config),
                    last_token_id=self.model.config.blank_token_id,
                )
                for _ in range(num_reqs)
            ],
            "query_end_positions": [
                (i + 1) * num_tokens // num_reqs for i in range(num_reqs)
            ],
            "decode_ready": [True] * num_reqs,
        }

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
    ) -> dict[str, Any]:
        return {}
