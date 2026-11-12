# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import os
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup

from .model import Nemotron3_5AsrDecodeState, decoder_cache_cls

logger = init_logger(__name__)
_ENCODER_GRAPHS = os.environ.get("NEMOTRON_ENCODER_GRAPHS", "1") != "0"


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
            decoder_cache=decoder_cache_cls(config)(config),
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
        self._encoder_graphs: dict[tuple, _StreamingEncoderGraph | None] = {}
        self._graph_pool = torch.cuda.graph_pool_handle()

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
            raise ValueError("Nemotron ASR expects one audio per request.")
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

        while True:
            pending = [
                (req_id, state)
                for req_id in input_batch.req_ids
                if (state := self.streaming_states.get(req_id)) is not None
                and state.encoded_features < state.num_features
            ]
            if not pending:
                break
            self._encode_streaming_features(pending)
        return None

    def _streaming_feature(
        self,
        req_id: str,
        state: Nemotron3_5AsrStreamingState,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int]:
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
        prompt_ids_elem = feature.data.get("prompt_ids")
        prompt_ids = None if prompt_ids_elem is None else prompt_ids_elem.data
        lookahead = feature.data["num_lookahead_tokens"].data
        assert isinstance(input_features, torch.Tensor)
        if input_features.ndim == 2:
            input_features = input_features.unsqueeze(0)
        if isinstance(prompt_ids, torch.Tensor):
            prompt_ids = prompt_ids.reshape(-1)[:1]
        if isinstance(lookahead, torch.Tensor):
            lookahead = int(lookahead.reshape(-1)[0].item())
        return input_features, prompt_ids, int(lookahead)

    def _encode_streaming_features(
        self,
        batch: list[tuple[str, Nemotron3_5AsrStreamingState]],
    ) -> None:
        groups: dict[tuple, list] = {}
        for req_id, state in batch:
            feature = self._streaming_feature(req_id, state)
            key = _streaming_batch_key(state, feature)
            groups.setdefault(key, []).append((req_id, state, *feature))
        for entries in groups.values():
            self._run_streaming_batch(entries)

    def _run_streaming_batch(self, entries: list[tuple]) -> None:
        inputs = [
            input_features.to(device=self.device, dtype=self.dtype)
            for _, _, input_features, _, _ in entries
        ]
        input_features = inputs[0] if len(inputs) == 1 else torch.cat(inputs, dim=0)
        prompt_ids = entries[0][3]
        lookahead = entries[0][4]
        if prompt_ids is not None and len(entries) > 1:
            prompt_ids = torch.cat(
                [entry[3].to(device=self.device) for entry in entries], dim=0
            )
        elif prompt_ids is not None:
            prompt_ids = prompt_ids.to(device=self.device)

        past_key_values = None
        padding_cache = None
        if entries[0][1].past_key_values is not None:
            past_key_values = _cat_attention_caches(
                [state.past_key_values for _, state, _, _, _ in entries]
            )
            padding_cache = _cat_padding_caches(
                [state.padding_cache for _, state, _, _, _ in entries]
            )

        frames, past_out, padding_out = self._encode_streaming(
            input_features,
            prompt_ids,
            lookahead,
            past_key_values,
            padding_cache,
        )
        frame_rows = list(frames.split(1, dim=0)) if len(entries) > 1 else [frames]
        past_per_request = _split_attention_cache(past_out, len(entries))
        padding_per_request = _split_padding_cache(padding_out, len(entries))
        for (_, state, _, _, _), frame_row, state_past, state_padding in zip(
            entries,
            frame_rows,
            past_per_request,
            padding_per_request,
            strict=True,
        ):
            self._append_streaming_frames(state, frame_row[0])
            state.past_key_values = state_past
            state.padding_cache = state_padding
            state.encoded_features += 1

    def _encode_streaming(
        self,
        input_features: torch.Tensor,
        prompt_ids: torch.Tensor | None,
        lookahead: int,
        past_key_values: object | None,
        padding_cache: object | None,
    ) -> tuple[torch.Tensor, object, object]:
        if not _ENCODER_GRAPHS or past_key_values is None:
            return self.model.audio_encoder.forward_streaming_chunk(
                input_features,
                prompt_ids,
                lookahead,
                past_key_values=past_key_values,
                padding_cache=padding_cache,
            )
        key = _graph_key(
            input_features, prompt_ids, lookahead, past_key_values, padding_cache
        )
        if key not in self._encoder_graphs:
            try:
                self._encoder_graphs[key] = _StreamingEncoderGraph(
                    self.model.audio_encoder,
                    self._graph_pool,
                    input_features,
                    prompt_ids,
                    lookahead,
                    past_key_values,
                    padding_cache,
                )
            except Exception:
                logger.exception("Falling back to eager streaming encoder.")
                self._encoder_graphs[key] = None
        graph = self._encoder_graphs[key]
        if graph is not None:
            return graph.run(input_features, prompt_ids, past_key_values, padding_cache)
        return self.model.audio_encoder.forward_streaming_chunk(
            input_features,
            prompt_ids,
            lookahead,
            past_key_values=past_key_values,
            padding_cache=padding_cache,
        )

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
                    decoder_cache=decoder_cache_cls(self.model.config)(
                        self.model.config
                    ),
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
                    decoder_cache=decoder_cache_cls(self.model.config)(
                        self.model.config
                    ),
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


def _streaming_batch_key(
    state: Nemotron3_5AsrStreamingState,
    feature: tuple[torch.Tensor, torch.Tensor | None, int],
) -> tuple:
    """Identify requests whose encoder step can share one batched forward."""
    input_features, prompt_ids, lookahead = feature
    past_key_values = state.past_key_values
    if past_key_values is None:
        cache_key = None
        padding_key = None
    else:
        cache_key = tuple(
            (
                layer.keys.shape,
                layer.values.shape,
                _cache_length(layer),
            )
            for layer in past_key_values.layers
        )
        padding_key = tuple(
            (name, layer.cache.shape)
            for name, layer in state.padding_cache.layers.items()
        )
    return (
        tuple(input_features.shape),
        lookahead,
        prompt_ids is not None,
        cache_key,
        padding_key,
    )


def _clone_cache(cache: object, layers: object) -> object:
    clone = copy.copy(cache)
    clone.layers = layers
    return clone


def _cat_attention_caches(caches: list[object]) -> object:
    if len(caches) == 1:
        return caches[0]
    base = caches[0]
    layers = []
    for index, layer in enumerate(base.layers):
        layer_copy = copy.copy(layer)
        layer_copy.keys = torch.cat(
            [cache.layers[index].keys for cache in caches], dim=0
        )
        layer_copy.values = torch.cat(
            [cache.layers[index].values for cache in caches], dim=0
        )
        layers.append(layer_copy)
    return _clone_cache(base, layers)


def _split_attention_cache(cache: object, count: int) -> list[object]:
    splits = []
    for row in range(count):
        layers = []
        for layer in cache.layers:
            layer_copy = copy.copy(layer)
            layer_copy.keys = layer.keys[row : row + 1].clone()
            layer_copy.values = layer.values[row : row + 1].clone()
            layers.append(layer_copy)
        splits.append(_clone_cache(cache, layers))
    return splits


def _cat_padding_caches(caches: list[object]) -> object:
    if len(caches) == 1:
        return caches[0]
    base = caches[0]
    clone = copy.copy(base)
    clone.layers = {}
    for name, layer in base.layers.items():
        layer_copy = copy.copy(layer)
        layer_copy.cache = torch.cat(
            [cache.layers[name].cache for cache in caches], dim=0
        )
        clone.layers[name] = layer_copy
    return clone


def _split_padding_cache(cache: object, count: int) -> list[object]:
    splits = []
    for row in range(count):
        clone = copy.copy(cache)
        clone.layers = {}
        for name, layer in cache.layers.items():
            layer_copy = copy.copy(layer)
            layer_copy.cache = layer.cache[row : row + 1].clone()
            clone.layers[name] = layer_copy
        splits.append(clone)
    return splits


def _graph_key(
    input_features: torch.Tensor,
    prompt_ids: torch.Tensor | None,
    lookahead: int,
    past_key_values: object | None,
    padding_cache: object | None,
) -> tuple:
    if past_key_values is None:
        cache_key = None
    else:
        cache_key = tuple(
            (
                layer.keys.shape,
                layer.values.shape,
                _cache_length(layer),
            )
            for layer in past_key_values.layers
        )
    if padding_cache is None:
        padding_key = None
    else:
        padding_key = tuple(
            (name, layer.cache.shape) for name, layer in padding_cache.layers.items()
        )
    return (
        tuple(input_features.shape),
        lookahead,
        None if prompt_ids is None else tuple(prompt_ids.shape),
        cache_key,
        padding_key,
    )


def _cache_length(layer: object) -> int | None:
    """Bucket the absolute cache length once the sliding window saturates."""
    length = getattr(layer, "cumulative_length", None)
    window = getattr(layer, "sliding_window", None)
    if length is None:
        return None
    return length if window is None else min(length, window)


def _static_attention_cache(cache: object) -> object:
    if cache is None:
        return None
    layers = []
    for layer in cache.layers:
        layer_copy = copy.copy(layer)
        layer_copy.keys = torch.zeros_like(layer.keys)
        layer_copy.values = torch.zeros_like(layer.values)
        layers.append(layer_copy)
    return _clone_cache(cache, layers)


def _static_padding_cache(cache: object) -> object:
    if cache is None:
        return None
    clone = copy.copy(cache)
    clone.layers = {}
    for name, layer in cache.layers.items():
        layer_copy = copy.copy(layer)
        layer_copy.cache = torch.zeros_like(layer.cache)
        clone.layers[name] = layer_copy
    return clone


class _StreamingEncoderGraph:
    """Replay one fixed-shape cache-aware encoder step as a CUDA graph.

    The attention cache is replaced in place by the captured kernels, so the
    graph keeps its own static cache buffers and copies live state in and out
    around every replay. Padding caches are already updated in place.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        pool: object,
        input_features: torch.Tensor,
        prompt_ids: torch.Tensor | None,
        lookahead: int,
        past_key_values: object | None,
        padding_cache: object | None,
    ) -> None:
        assert past_key_values is not None
        assert padding_cache is not None
        self.input_features = input_features.clone()
        self.prompt_ids = None if prompt_ids is None else prompt_ids.clone()
        self.cache = _static_attention_cache(past_key_values)
        self.padding_cache = _static_padding_cache(padding_cache)
        self.cache_inputs = [(layer.keys, layer.values) for layer in self.cache.layers]
        self.padding_inputs = {
            name: layer.cache for name, layer in self.padding_cache.layers.items()
        }

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(2):
                encoder.forward_streaming_chunk(
                    self.input_features,
                    self.prompt_ids,
                    lookahead,
                    past_key_values=_static_attention_cache(self.cache),
                    padding_cache=_static_padding_cache(self.padding_cache),
                )
        torch.cuda.current_stream().wait_stream(stream)

        self.graph = torch.cuda.CUDAGraph()
        with (
            torch.inference_mode(),
            torch.cuda.graph(self.graph, pool=pool, capture_error_mode="thread_local"),
        ):
            output = encoder.forward_streaming_chunk(
                self.input_features,
                self.prompt_ids,
                lookahead,
                past_key_values=self.cache,
                padding_cache=self.padding_cache,
            )
        torch.cuda.synchronize()
        self.frames, self.past_key_values, self.padding_key_values = output

    def run(
        self,
        input_features: torch.Tensor,
        prompt_ids: torch.Tensor | None,
        past_key_values: object | None,
        padding_cache: object | None,
    ) -> tuple[torch.Tensor, object, object]:
        self.input_features.copy_(input_features)
        if self.prompt_ids is not None and prompt_ids is not None:
            self.prompt_ids.copy_(prompt_ids)
        for (key_buffer, value_buffer), layer in zip(
            self.cache_inputs, past_key_values.layers, strict=True
        ):
            key_buffer.copy_(layer.keys)
            value_buffer.copy_(layer.values)
        for name, buffer in self.padding_inputs.items():
            buffer.copy_(padding_cache.layers[name].cache)
        self.graph.replay()
        return self.frames, self.past_key_values, self.padding_key_values
