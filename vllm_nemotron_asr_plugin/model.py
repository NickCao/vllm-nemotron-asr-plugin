# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA Nemotron ASR transcription.

Supports the multilingual ``nvidia/nemotron-3.5-asr-streaming-0.6b`` checkpoint
(language-prompt conditioning) and the English
``nvidia/nemotron-speech-streaming-en-0.6b`` checkpoint (no prompt
conditioning). Both share the cache-aware FastConformer encoder and the LSTM
RNN-T decoder/joint implementation from Transformers.
"""

import asyncio
from collections.abc import AsyncGenerator, Iterable
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import (
    Nemotron3_5AsrConfig,
    NemotronAsrStreamingConfig,
    NemotronAsrStreamingEncoder,
)
from transformers.models.nemotron3_5_asr.generation_nemotron3_5_asr import (
    Nemotron3_5AsrRNNTDecoderCache,
)
from transformers.models.nemotron3_5_asr.modeling_nemotron3_5_asr import (
    Nemotron3_5AsrPromptProjector,
    Nemotron3_5AsrRNNTDecoder,
    Nemotron3_5AsrRNNTJointNetwork,
)
from transformers.models.nemotron3_5_asr.processing_nemotron3_5_asr import (
    DEFAULT_PROMPT_DICTIONARY,
)
from transformers.models.nemotron_asr_streaming.generation_nemotron_asr_streaming import (  # noqa: E501
    NemotronAsrStreamingRNNTDecoderCache,
)
from transformers.models.nemotron_asr_streaming.modeling_nemotron_asr_streaming import (
    NemotronAsrStreamingRNNTDecoder,
    NemotronAsrStreamingRNNTJointNetwork,
)
from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.config.speech_to_text import SpeechToTextParams
from vllm.inputs import (
    ExplicitEncoderDecoderPrompt,
    PromptType,
    TextPrompt,
    TokensPrompt,
)
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsRealtime,
    SupportsTranscription,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.model_executor.models.whisper_utils import ISO639_1_SUPPORTED_LANGS
from vllm.multimodal import MULTIMODAL_REGISTRY

from .processing import (
    Nemotron3_5AsrDummyInputsBuilder,
    Nemotron3_5AsrMultiModalProcessor,
    Nemotron3_5AsrProcessingInfo,
)

if TYPE_CHECKING:
    from vllm_nemotron_asr_plugin.model_state import (
        Nemotron3_5AsrModelState,
    )


NemotronAsrConfig = Nemotron3_5AsrConfig | NemotronAsrStreamingConfig
NemotronAsrRNNTDecoder = Nemotron3_5AsrRNNTDecoder | NemotronAsrStreamingRNNTDecoder
NemotronAsrRNNTJointNetwork = (
    Nemotron3_5AsrRNNTJointNetwork | NemotronAsrStreamingRNNTJointNetwork
)
NemotronAsrRNNTDecoderCache = (
    Nemotron3_5AsrRNNTDecoderCache | NemotronAsrStreamingRNNTDecoderCache
)


def supports_prompt_conditioning(config: object) -> bool:
    """Whether the checkpoint fuses a language-prompt one-hot into the encoder."""
    return getattr(config, "num_prompts", None) is not None


def decoder_cache_cls(config: object) -> type:
    if isinstance(config, Nemotron3_5AsrConfig):
        return Nemotron3_5AsrRNNTDecoderCache
    if isinstance(config, NemotronAsrStreamingConfig):
        return NemotronAsrStreamingRNNTDecoderCache
    raise TypeError(f"Unsupported Nemotron ASR config: {type(config).__name__}")


def rnnt_decoder_cls(config: object) -> type:
    if isinstance(config, Nemotron3_5AsrConfig):
        return Nemotron3_5AsrRNNTDecoder
    if isinstance(config, NemotronAsrStreamingConfig):
        return NemotronAsrStreamingRNNTDecoder
    raise TypeError(f"Unsupported Nemotron ASR config: {type(config).__name__}")


def rnnt_joint_cls(config: object) -> type:
    if isinstance(config, Nemotron3_5AsrConfig):
        return Nemotron3_5AsrRNNTJointNetwork
    if isinstance(config, NemotronAsrStreamingConfig):
        return NemotronAsrStreamingRNNTJointNetwork
    raise TypeError(f"Unsupported Nemotron ASR config: {type(config).__name__}")


class Nemotron3_5AsrAudioEncoder(nn.Module):
    """Encode mel features, applying language-prompt conditioning when present."""

    def __init__(self, config: NemotronAsrConfig):
        super().__init__()
        self.config = config
        self.encoder = NemotronAsrStreamingEncoder(config.encoder_config)
        self.encoder_projector = nn.Linear(
            config.encoder_config.hidden_size,
            config.decoder_hidden_size,
        )
        self.prompt_projector = (
            Nemotron3_5AsrPromptProjector(config)
            if supports_prompt_conditioning(config)
            else None
        )

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        prompt_ids: torch.LongTensor | None = None,
        num_lookahead_tokens: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        encoder_outputs = self.encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            num_lookahead_tokens=num_lookahead_tokens,
            use_cache=False,
        )
        hidden_states = encoder_outputs.last_hidden_state
        output_mask = encoder_outputs.attention_mask
        if output_mask is not None:
            output_mask = output_mask.bool()

        return self._project(hidden_states, prompt_ids), output_mask

    def forward_streaming_chunk(
        self,
        input_features: torch.Tensor,
        prompt_ids: torch.LongTensor | None,
        num_lookahead_tokens: int,
        past_key_values: object | None = None,
        padding_cache: object | None = None,
    ) -> tuple[torch.Tensor, object, object]:
        """Encode one exact-size cache-aware chunk and return updated caches."""
        encoder_outputs = self.encoder(
            input_features=input_features,
            attention_mask=None,
            num_lookahead_tokens=num_lookahead_tokens,
            past_key_values=past_key_values,
            padding_cache=padding_cache,
            use_cache=True,
            output_attention_mask=False,
        )
        hidden_states = self._project(encoder_outputs.last_hidden_state, prompt_ids)
        return (
            hidden_states,
            encoder_outputs.past_key_values,
            encoder_outputs.padding_cache,
        )

    def _project(
        self,
        hidden_states: torch.Tensor,
        prompt_ids: torch.LongTensor | None,
    ) -> torch.Tensor:
        if self.prompt_projector is None:
            return self.encoder_projector(hidden_states)
        if prompt_ids is None:
            prompt_ids = torch.full(
                (hidden_states.shape[0],),
                self.config.default_prompt_id,
                dtype=torch.long,
                device=hidden_states.device,
            )
        prompt_ids = prompt_ids.to(hidden_states.device)
        prompt = F.one_hot(
            prompt_ids,
            num_classes=self.config.num_prompts,
        ).to(hidden_states.dtype)
        prompt = prompt[:, None, :].expand(-1, hidden_states.shape[1], -1)
        hidden_states = self.prompt_projector(
            torch.cat((hidden_states, prompt), dim=-1)
        )
        return self.encoder_projector(hidden_states)


class Nemotron3_5AsrRealtimeBuffer:
    """Accumulate arbitrary PCM writes into exact cache-aware chunk sizes."""

    def __init__(self, initial_capacity: int = 32_000) -> None:
        self._buffer = np.empty(initial_capacity, dtype=np.float32)
        self._filled = 0

    def write(self, audio: np.ndarray) -> None:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        end = self._filled + audio.size
        if end > self._buffer.size:
            new_buffer = np.empty(max(end, self._buffer.size * 2), dtype=np.float32)
            new_buffer[: self._filled] = self._buffer[: self._filled]
            self._buffer = new_buffer
        self._buffer[self._filled : end] = audio
        self._filled = end

    def read(self, num_samples: int) -> np.ndarray | None:
        if self._filled < num_samples:
            return None
        chunk = self._buffer[:num_samples].copy()
        remaining = self._filled - num_samples
        if remaining:
            self._buffer[:remaining] = self._buffer[num_samples : self._filled]
        self._filled = remaining
        return chunk

    def flush(self) -> np.ndarray | None:
        if not self._filled:
            return None
        audio = self._buffer[: self._filled].copy()
        self._filled = 0
        return audio


async def iter_realtime_chunks(
    audio_stream: AsyncGenerator[np.ndarray, None],
    *,
    first_chunk_samples: int,
    subsequent_chunk_samples: int,
) -> AsyncGenerator[tuple[np.ndarray, bool, bool], None]:
    """Split a live audio stream into cache-aware encoder chunks.

    Yields ``(audio, is_first_audio_chunk, is_last_audio_chunk)``. A chunk is
    emitted as soon as it is complete so the engine can start on it without
    waiting for the next chunk; only the trailing partial chunk is padded and
    marked as last.
    """
    buffer = Nemotron3_5AsrRealtimeBuffer()
    audio_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()

    async def fill_audio_buffer() -> None:
        try:
            async for audio in audio_stream:
                audio_queue.put_nowait(audio)
        finally:
            audio_queue.put_nowait(None)

    async def read_chunk(num_samples: int) -> np.ndarray | None:
        while True:
            chunk = buffer.read(num_samples)
            if chunk is not None:
                return chunk
            audio = await audio_queue.get()
            if audio is None:
                return None
            buffer.write(audio)

    fill_task = asyncio.create_task(fill_audio_buffer())
    try:
        is_first_audio_chunk = True
        while True:
            required_samples = (
                first_chunk_samples
                if is_first_audio_chunk
                else subsequent_chunk_samples
            )
            chunk = await read_chunk(required_samples)
            if chunk is None:
                break
            yield chunk, is_first_audio_chunk, False
            is_first_audio_chunk = False

        remaining = buffer.flush()
        if remaining is not None:
            required_samples = (
                first_chunk_samples
                if is_first_audio_chunk
                else subsequent_chunk_samples
            )
            if remaining.size < required_samples:
                remaining = np.pad(
                    remaining,
                    (0, required_samples - remaining.size),
                )
            yield remaining, is_first_audio_chunk, True
    finally:
        fill_task.cancel()
        with suppress(asyncio.CancelledError):
            await fill_task


@dataclass
class Nemotron3_5AsrDecodeState:
    encoder_frames: torch.Tensor
    decoder_cache: NemotronAsrRNNTDecoderCache
    last_token_id: int
    frame_idx: int = 0
    symbols_at_frame: int = 0
    num_tokens_to_replay: int = 0


def _decode_next_tokens(
    states: list[Nemotron3_5AsrDecodeState],
    decoder: NemotronAsrRNNTDecoder,
    joint: NemotronAsrRNNTJointNetwork,
    *,
    blank_token_id: int,
    max_symbols_per_step: int,
) -> list[int | None]:
    """Advance active requests to their next transcript token or exhaustion."""
    selected: list[int | None] = [None] * len(states)
    pending = list(enumerate(states))
    while pending:
        active = []
        for index, state in pending:
            if state.frame_idx < state.encoder_frames.shape[0]:
                active.append((index, state))
            elif state.num_tokens_to_replay:
                raise RuntimeError("RNNT exhausted before replaying the prefix.")
        if not active:
            break

        # HF initializes every fresh row, including its blank seed. Once the
        # cache is initialized, blank rows instead preserve the previous state.
        active.sort(key=lambda item: item[1].decoder_cache.is_initialized)
        groups = [
            [s for _, s in active if s.decoder_cache.is_initialized == initialized]
            for initialized in (False, True)
        ]
        decoder_outputs = []
        for initialized, group in zip((False, True), groups):
            if not group:
                continue
            cache = type(group[0].decoder_cache)(group[0].decoder_cache.config)
            if initialized:
                cache.cache = torch.cat([s.decoder_cache.cache for s in group])
                cache.hidden_state = torch.cat(
                    [s.decoder_cache.hidden_state for s in group], dim=1
                )
                cache.cell_state = torch.cat(
                    [s.decoder_cache.cell_state for s in group], dim=1
                )
                cache.is_initialized = True
            input_ids = torch.tensor(
                [[s.last_token_id] for s in group],
                dtype=torch.long,
                device=group[0].encoder_frames.device,
            )
            decoder_outputs.append(decoder(input_ids, cache=cache)[:, -1, :])
            for row, state in enumerate(group):
                state.decoder_cache.update(
                    cache.cache[row : row + 1],
                    cache.hidden_state[:, row : row + 1],
                    cache.cell_state[:, row : row + 1],
                )

        encoder_frames = torch.stack([s.encoder_frames[s.frame_idx] for _, s in active])
        logits = joint(torch.cat(decoder_outputs), encoder_frames)
        token_ids = logits.argmax(dim=-1).tolist()
        pending = []
        for (index, state), token_id in zip(active, token_ids):
            state.last_token_id = token_id
            if token_id == blank_token_id:
                state.frame_idx += 1
                state.symbols_at_frame = 0
                pending.append((index, state))
                continue

            state.symbols_at_frame += 1
            if state.symbols_at_frame >= max_symbols_per_step:
                state.frame_idx += 1
                state.symbols_at_frame = 0
            if state.num_tokens_to_replay:
                state.num_tokens_to_replay -= 1
                pending.append((index, state))
            else:
                selected[index] = token_id
    return selected


@MULTIMODAL_REGISTRY.register_processor(
    Nemotron3_5AsrMultiModalProcessor,
    info=Nemotron3_5AsrProcessingInfo,
    dummy_inputs=Nemotron3_5AsrDummyInputsBuilder,
)
class Nemotron3_5AsrForRNNT(
    nn.Module, SupportsTranscription, SupportsMultiModal, SupportsRealtime
):
    supports_transcription_only = False
    realtime_max_tokens = 256
    supported_languages = {
        code: name
        for code, name in ISO639_1_SUPPORTED_LANGS.items()
        if code in DEFAULT_PROMPT_DICTIONARY
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "encoder.": "audio_encoder.encoder.",
            "encoder_projector.": "audio_encoder.encoder_projector.",
            "prompt_projector.": "audio_encoder.prompt_projector.",
        }
    )

    @staticmethod
    def get_model_state_cls() -> type["Nemotron3_5AsrModelState"]:
        from vllm_nemotron_asr_plugin.model_state import (
            Nemotron3_5AsrModelState,
        )

        return Nemotron3_5AsrModelState

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig:
        del model_config, task_type
        return SpeechToTextConfig(
            sample_rate=16_000,
            max_audio_clip_s=None,
            min_energy_split_window_size=None,
        )

    @classmethod
    def get_generation_prompt(cls, stt_params: SpeechToTextParams) -> PromptType:
        if stt_params.task_type != "transcribe":
            raise ValueError("Nemotron ASR supports transcription only.")
        if stt_params.request_prompt or stt_params.hotwords or stt_params.to_language:
            raise ValueError("Text prompts, hotwords and translation are unsupported.")
        language = stt_params.language or "auto"
        if supports_prompt_conditioning(stt_params.model_config.hf_config):
            mm_processor_kwargs: dict[str, object] = {"language": language}
        else:
            if language != "auto" and not language.lower().startswith("en"):
                raise ValueError(
                    "Nemotron ASR Streaming supports English transcription only."
                )
            mm_processor_kwargs = {}
        return ExplicitEncoderDecoderPrompt(
            encoder_prompt=TextPrompt(
                prompt="",
                multi_modal_data={
                    "audio": (stt_params.audio, stt_params.stt_config.sample_rate)
                },
                mm_processor_kwargs=mm_processor_kwargs,
            ),
            decoder_prompt=TokensPrompt(
                prompt_token_ids=[stt_params.model_config.hf_config.blank_token_id]
            ),
        )

    @classmethod
    async def buffer_realtime_audio(
        cls,
        audio_stream: AsyncGenerator[np.ndarray, None],
        input_stream: asyncio.Queue[list[int]],
        model_config: ModelConfig,
    ) -> AsyncGenerator[PromptType, None]:
        processor = MULTIMODAL_REGISTRY.create_processor(
            model_config
        ).info.get_hf_processor()
        feature_extractor = processor.feature_extractor
        sampling_rate = int(feature_extractor.sampling_rate)
        first_chunk_samples = int(processor.num_samples_first_audio_chunk)
        subsequent_chunk_samples = int(processor.num_samples_per_audio_chunk)
        blank_token_id = model_config.hf_config.blank_token_id
        prompt_conditioning = supports_prompt_conditioning(model_config.hf_config)

        def make_prompt(
            audio: np.ndarray,
            *,
            is_first_audio_chunk: bool,
            is_last_audio_chunk: bool = False,
        ) -> PromptType:
            mm_processor_kwargs: dict[str, object] = {
                "sampling_rate": sampling_rate,
                "is_streaming": True,
                "is_first_audio_chunk": is_first_audio_chunk,
                "audio_kwargs": {
                    "is_last_audio_chunk": is_last_audio_chunk,
                },
            }
            if prompt_conditioning:
                mm_processor_kwargs["language"] = "auto"
            return ExplicitEncoderDecoderPrompt(
                encoder_prompt=TextPrompt(
                    prompt="",
                    multi_modal_data={"audio": audio},
                    mm_processor_kwargs=mm_processor_kwargs,
                ),
                decoder_prompt=TokensPrompt(
                    prompt_token_ids=[blank_token_id],
                ),
            )

        async def drain_output_tokens() -> None:
            while True:
                await input_stream.get()

        drain_task = asyncio.create_task(drain_output_tokens())
        try:
            async for audio, is_first, is_last in iter_realtime_chunks(
                audio_stream,
                first_chunk_samples=first_chunk_samples,
                subsequent_chunk_samples=subsequent_chunk_samples,
            ):
                yield make_prompt(
                    audio,
                    is_first_audio_chunk=is_first,
                    is_last_audio_chunk=is_last,
                )
        finally:
            drain_task.cancel()
            with suppress(asyncio.CancelledError):
                await drain_task

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config: NemotronAsrConfig = vllm_config.model_config.hf_config
        self.audio_encoder = Nemotron3_5AsrAudioEncoder(self.config)
        self.decoder = rnnt_decoder_cls(self.config)(self.config)
        self.joint = rnnt_joint_cls(self.config)(self.config)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        input_features = cast(
            torch.Tensor | list[torch.Tensor], kwargs["input_features"]
        )
        attention_mask = cast(
            torch.Tensor | list[torch.Tensor] | None, kwargs.get("attention_mask")
        )
        prompt_ids = cast(torch.Tensor | None, kwargs.get("prompt_ids"))
        lookahead = cast(int | torch.Tensor | None, kwargs.get("num_lookahead_tokens"))
        dtype = self.audio_encoder.encoder_projector.weight.dtype
        if isinstance(input_features, torch.Tensor):
            hidden_states, output_mask = self.audio_encoder(
                input_features.to(dtype),
                attention_mask,
                prompt_ids,
                int(lookahead) if lookahead is not None else None,
            )
            if output_mask is None:
                return list(hidden_states.unbind(0))
            return [states[mask] for states, mask in zip(hidden_states, output_mask)]

        output: list[torch.Tensor] = []
        for i, features in enumerate(input_features):
            mask = attention_mask[i] if attention_mask is not None else None
            prompt = prompt_ids[i : i + 1] if prompt_ids is not None else None
            states, valid = self.audio_encoder(
                features.unsqueeze(0).to(dtype),
                mask.unsqueeze(0) if mask is not None else None,
                prompt,
                int(lookahead) if lookahead is not None else None,
            )
            output.append(states[0] if valid is None else states[0][valid[0]])
        return output

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        decode_states: list[Nemotron3_5AsrDecodeState | None] | None = None,
        query_end_positions: list[int] | None = None,
        decode_ready: list[bool] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        del positions, kwargs
        selected_ids = torch.full(
            (input_ids.shape[0], 1),
            self.config.blank_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        if decode_states is not None:
            assert query_end_positions is not None and decode_ready is not None
            active = [
                (state, end)
                for state, end, ready in zip(
                    decode_states, query_end_positions, decode_ready
                )
                if state is not None and ready
            ]
            token_ids = _decode_next_tokens(
                [state for state, _ in active],
                self.decoder,
                self.joint,
                blank_token_id=self.config.blank_token_id,
                max_symbols_per_step=self.config.max_symbols_per_step,
            )
            for (_, end), token_id in zip(active, token_ids):
                if token_id is not None:
                    selected_ids[end - 1, 0] = token_id
        return selected_ids

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # RNNT already consumed blanks and selected the next transcript token.
        # Preserve that choice in the sampler; these are not RNNT probabilities.
        logits = torch.full(
            (hidden_states.shape[0], self.config.vocab_size),
            float("-inf"),
            dtype=self.audio_encoder.encoder_projector.weight.dtype,
            device=hidden_states.device,
        )
        return logits.scatter_(1, hidden_states.long(), 0.0)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(
            weights, mapper=self.hf_to_vllm_mapper
        )


class NemotronAsrStreamingForRNNT(Nemotron3_5AsrForRNNT):
    """English Nemotron ASR Streaming checkpoint (no prompt conditioning)."""

    supported_languages = {"en": "English"}
