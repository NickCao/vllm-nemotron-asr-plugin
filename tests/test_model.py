# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from transformers import (
    Nemotron3_5AsrConfig,
    NemotronAsrStreamingEncoderConfig,
)
from transformers import (
    Nemotron3_5AsrForRNNT as HFNemotron3_5AsrForRNNT,
)
from transformers.models.nemotron3_5_asr.generation_nemotron3_5_asr import (
    Nemotron3_5AsrRNNTDecoderCache,
)
from transformers.models.nemotron3_5_asr.modeling_nemotron3_5_asr import (
    Nemotron3_5AsrRNNTDecoder,
)
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers

from vllm_nemotron_asr_plugin.config import Nemotron3_5AsrForRNNTConfig
from vllm_nemotron_asr_plugin.model import (
    Nemotron3_5AsrDecodeState,
    Nemotron3_5AsrForRNNT,
    Nemotron3_5AsrRealtimeBuffer,
    _decode_next_tokens,
    iter_realtime_chunks,
)
from vllm_nemotron_asr_plugin.model_state import (
    Nemotron3_5AsrModelState,
    Nemotron3_5AsrStreamingState,
)
from vllm_nemotron_asr_plugin.processing import (
    _get_streaming_subsampling_output_lengths,
)


def test_installed_plugin_discovery_without_in_tree_nemotron() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys

class NoInTreeNemotron(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("vllm.") and fullname.endswith(".nemotron3_5_asr"):
            raise AssertionError(f"Unexpected in-tree import: {fullname}")

sys.meta_path.insert(0, NoInTreeNemotron())
from vllm.plugins import load_general_plugins
load_general_plugins()
from vllm import ModelRegistry
from vllm.config import ModelConfig
from vllm.model_executor.models.config import MODELS_CONFIG_MAP
from vllm.renderers.registry import RENDERER_REGISTRY
from vllm.tokenizers.registry import TokenizerRegistry
from vllm_nemotron_asr_plugin import register
from vllm_nemotron_asr_plugin.config import Nemotron3_5AsrForRNNTConfig
from vllm_nemotron_asr_plugin.model import Nemotron3_5AsrForRNNT
from vllm_nemotron_asr_plugin.processing import Nemotron3_5AsrRenderer

arch = "Nemotron3_5AsrForRNNT"
assert MODELS_CONFIG_MAP[arch] is Nemotron3_5AsrForRNNTConfig
register()  # Repeated registration must preserve the same bindings.
config = ModelConfig(
    model="nvidia/nemotron-3.5-asr-streaming-0.6b", max_model_len=512,
    enforce_eager=True,
)
assert ModelRegistry.resolve_model_cls(arch, config)[0] is Nemotron3_5AsrForRNNT
assert config.tokenizer_mode == "nemotron3_5_asr"
renderer = RENDERER_REGISTRY.load_renderer_cls(config.tokenizer_mode)
tokenizer = TokenizerRegistry.load_tokenizer_cls(config.tokenizer_mode)
state = Nemotron3_5AsrForRNNT.get_model_state_cls()
assert renderer is Nemotron3_5AsrRenderer
assert tokenizer.__name__ == "CachedHfTokenizer"
assert state.__module__ == "vllm_nemotron_asr_plugin.model_state"
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("tokenizer_mode", ["auto", "nemotron3_5_asr"])
def test_nemotron_asr_model_config(tokenizer_mode: str) -> None:
    config = SimpleNamespace(
        hf_config=SimpleNamespace(decoder_hidden_size=640),
        model_arch_config=SimpleNamespace(hidden_size=1024),
        tokenizer_mode=tokenizer_mode,
        max_logprobs=20,
    )

    Nemotron3_5AsrForRNNTConfig.verify_and_update_model_config(config)

    assert config.model_arch_config.hidden_size == 640
    assert config.tokenizer_mode == "nemotron3_5_asr"
    assert config.max_logprobs == 0


def test_nemotron_asr_rejects_unsupported_tokenizer_mode() -> None:
    config = SimpleNamespace(
        hf_config=SimpleNamespace(decoder_hidden_size=640),
        model_arch_config=SimpleNamespace(hidden_size=1024),
        tokenizer_mode="hf",
        max_logprobs=20,
    )

    with pytest.raises(ValueError, match="tokenizer mode"):
        Nemotron3_5AsrForRNNTConfig.verify_and_update_model_config(config)


@pytest.mark.parametrize(
    ("section", "field", "value", "error"),
    [
        (None, "use_v2_model_runner", False, "Model Runner V2"),
        ("model_config", "enforce_eager", False, "enforce-eager"),
        ("parallel_config", "tensor_parallel_size", 2, "TP=1"),
        ("parallel_config", "pipeline_parallel_size", 2, "PP=1"),
        ("model_config", "quantization", "fp8", "quantization"),
        (None, "speculative_config", object(), "speculative decoding"),
    ],
)
def test_nemotron_asr_rejects_unsupported_execution_config(
    section: str | None,
    field: str,
    value: object,
    error: str,
) -> None:
    config = SimpleNamespace(
        use_v2_model_runner=True,
        model_config=SimpleNamespace(enforce_eager=True, quantization=None),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1
        ),
        speculative_config=None,
    )
    target = getattr(config, section) if section else config
    setattr(target, field, value)

    with pytest.raises(ValueError, match=error):
        Nemotron3_5AsrForRNNTConfig.verify_and_update_config(config)


def _get_tiny_config() -> Nemotron3_5AsrConfig:
    encoder_config = NemotronAsrStreamingEncoderConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        attention_bias=False,
        convolution_bias=False,
        conv_kernel_size=3,
        subsampling_factor=8,
        subsampling_conv_channels=4,
        num_mel_bins=8,
        subsampling_conv_kernel_size=3,
        subsampling_conv_stride=2,
        dropout=0.0,
        dropout_positions=0.0,
        layerdrop=0.0,
        activation_dropout=0.0,
        attention_dropout=0.0,
        max_position_embeddings=32,
        scale_input=False,
        sliding_window=9,
        default_num_lookahead_tokens=3,
    )
    return Nemotron3_5AsrConfig(
        encoder_config=encoder_config,
        decoder_hidden_size=8,
        num_prompts=8,
        prompt_intermediate_size=16,
        default_prompt_id=3,
    )


def _get_vllm_config(config: Nemotron3_5AsrConfig) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config),
    )


@pytest.mark.parametrize(
    ("num_mel_frames", "physical_frames", "valid_frames"),
    [(26, 5, 4), (128, 17, 17)],
)
def test_nemotron_audio_encoder_preserves_batch_and_valid_lengths(
    num_mel_frames: int,
    physical_frames: int,
    valid_frames: int,
) -> None:
    torch.manual_seed(0)
    config = _get_tiny_config()
    config.encoder_config.num_hidden_layers = 2
    model = Nemotron3_5AsrForRNNT(vllm_config=_get_vllm_config(config)).eval()
    input_features = torch.randn(2, num_mel_frames, 8)
    attention_mask = torch.zeros(2, num_mel_frames, dtype=torch.bool)
    attention_mask[0, : num_mel_frames - 1] = True
    attention_mask[1, :17] = True

    with torch.inference_mode():
        output, output_mask = model.audio_encoder(
            input_features,
            attention_mask,
            prompt_ids=torch.tensor([2, 3]),
        )
        batched = model.embed_multimodal(
            input_features=input_features,
            attention_mask=attention_mask,
            prompt_ids=torch.tensor([2, 3]),
        )
        unpadded = model.embed_multimodal(
            input_features=[input_features[0, :-1], input_features[1, :17]],
            attention_mask=[attention_mask[0, :-1], attention_mask[1, :17]],
            prompt_ids=torch.tensor([2, 3]),
        )

    assert output.shape == (2, physical_frames, 8)
    assert output_mask is not None
    assert output_mask.sum(-1).tolist() == [valid_frames, 3]
    assert torch.isfinite(output).all()
    assert len(batched) == len(unpadded) == 2
    for row in range(2):
        torch.testing.assert_close(batched[row], output[row, output_mask[row]])
        torch.testing.assert_close(unpadded[row], batched[row])


@pytest.mark.parametrize("num_mel_frames", [26, 32, 106, 112])
def test_streaming_subsampling_lengths_match_encoder(
    num_mel_frames: int,
) -> None:
    config = _get_tiny_config()
    model = Nemotron3_5AsrForRNNT(vllm_config=_get_vllm_config(config)).eval()
    encoder_config = config.encoder_config

    with torch.inference_mode():
        frames, _, _ = model.audio_encoder.forward_streaming_chunk(
            torch.randn(1, num_mel_frames, encoder_config.num_mel_bins),
            torch.tensor([2]),
            encoder_config.default_num_lookahead_tokens,
        )

    expected = _get_streaming_subsampling_output_lengths(
        torch.tensor([num_mel_frames]),
        subsampling_factor=encoder_config.subsampling_factor,
        subsampling_conv_kernel_size=encoder_config.subsampling_conv_kernel_size,
        subsampling_conv_stride=encoder_config.subsampling_conv_stride,
    )
    assert frames.shape[1] == expected.item()


def test_realtime_buffer_emits_exact_chunks_and_flushes_remainder() -> None:
    buffer = Nemotron3_5AsrRealtimeBuffer(initial_capacity=4)
    buffer.write(np.arange(10, dtype=np.float32))

    np.testing.assert_array_equal(
        buffer.read(4),
        np.arange(4, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        buffer.read(4),
        np.arange(4, 8, dtype=np.float32),
    )
    assert buffer.read(4) is None
    np.testing.assert_array_equal(
        buffer.flush(),
        np.arange(8, 10, dtype=np.float32),
    )
    assert buffer.flush() is None


def _collect_realtime_chunks(
    chunks: list[list[float]],
    *,
    first_chunk_samples: int,
    subsequent_chunk_samples: int,
) -> list[tuple[np.ndarray, bool, bool]]:
    async def audio_stream():
        for chunk in chunks:
            yield np.asarray(chunk, dtype=np.float32)

    async def collect():
        return [
            item
            async for item in iter_realtime_chunks(
                audio_stream(),
                first_chunk_samples=first_chunk_samples,
                subsequent_chunk_samples=subsequent_chunk_samples,
            )
        ]

    return asyncio.run(collect())


def test_iter_realtime_chunks_consumes_stream_and_marks_final_chunk() -> None:
    output = _collect_realtime_chunks(
        [[1, 2], [3, 4, 5], [6, 7], [8, 9, 10, 11]],
        first_chunk_samples=4,
        subsequent_chunk_samples=4,
    )

    assert len(output) == 3
    np.testing.assert_array_equal(output[0][0], [1, 2, 3, 4])
    assert output[0][1:] == (True, False)
    np.testing.assert_array_equal(output[1][0], [5, 6, 7, 8])
    assert output[1][1:] == (False, False)
    np.testing.assert_array_equal(output[2][0], [9, 10, 11, 0])
    assert output[2][1:] == (False, True)


def test_iter_realtime_chunks_pads_short_single_chunk() -> None:
    output = _collect_realtime_chunks(
        [[1, 2]],
        first_chunk_samples=4,
        subsequent_chunk_samples=4,
    )

    assert len(output) == 1
    np.testing.assert_array_equal(output[0][0], [1, 2, 0, 0])
    assert output[0][1:] == (True, True)


def test_iter_realtime_chunks_marks_exact_multiple_last_without_padding() -> None:
    output = _collect_realtime_chunks(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]],
        first_chunk_samples=4,
        subsequent_chunk_samples=4,
    )

    assert len(output) == 3
    np.testing.assert_array_equal(output[0][0], [1, 2, 3, 4])
    np.testing.assert_array_equal(output[1][0], [5, 6, 7, 8])
    np.testing.assert_array_equal(output[2][0], [9, 10, 11, 12])
    assert [item[1:] for item in output] == [
        (True, False),
        (False, False),
        (False, True),
    ]


def test_streaming_frame_buffer_grows_and_compacts_pending_frames() -> None:
    config = _get_tiny_config()
    model = Nemotron3_5AsrForRNNT(vllm_config=_get_vllm_config(config)).eval()
    state = Nemotron3_5AsrStreamingState(
        model,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    initial_capacity = state.frame_capacity
    first = torch.arange(
        3 * initial_capacity,
        dtype=torch.float32,
    ).reshape(3 * initial_capacity, 1)

    Nemotron3_5AsrModelState._append_streaming_frames(state, first)

    assert state.frame_capacity >= first.shape[0]
    torch.testing.assert_close(state.decode_state.encoder_frames, first)

    state.decode_state.frame_idx = 2
    second = torch.full((5, 1), -1.0)
    Nemotron3_5AsrModelState._append_streaming_frames(state, second)

    expected = torch.cat((first[2:], second))
    torch.testing.assert_close(state.decode_state.encoder_frames, expected)
    assert state.decode_state.frame_idx == 0


def _streaming_feature(is_streaming: bool) -> SimpleNamespace:
    return SimpleNamespace(
        modality="audio",
        identifier="streaming-audio",
        data={
            "is_streaming": SimpleNamespace(
                data=torch.tensor([is_streaming], dtype=torch.bool)
            )
        },
    )


def test_streaming_state_survives_worker_remove_and_readd() -> None:
    config = _get_tiny_config()
    model = Nemotron3_5AsrForRNNT(vllm_config=_get_vllm_config(config)).eval()
    state = Nemotron3_5AsrModelState.__new__(Nemotron3_5AsrModelState)
    state.model = model
    state.dtype = torch.float32
    state.device = torch.device("cpu")
    state.audio_hashes = {}
    state.request_indices = {}
    state.decode_states = {}
    state.num_tokens_to_replay = {}
    state.streaming_states = {}
    state.pending_streaming_states = {}
    feature = _streaming_feature(True)

    state.add_request(
        0,
        SimpleNamespace(req_id="stream", mm_features=[feature]),
    )
    session_state = state.streaming_states["stream"]
    session_state.encoded_features = 1

    state.remove_request("stream")
    assert "stream" not in state.streaming_states
    assert state.pending_streaming_states["stream"] is session_state

    state.add_request(
        1,
        SimpleNamespace(req_id="stream", mm_features=[feature, feature]),
    )

    assert state.streaming_states["stream"] is session_state
    assert session_state.num_features == 2
    assert session_state.encoded_features == 1
    assert state.request_indices["stream"] == 1

    state.apply_staged_writes()
    assert not state.pending_streaming_states


class _ScriptedJoint(torch.nn.Module):
    def __init__(self, token_ids: list[list[int]], vocab_size: int):
        super().__init__()
        self.tokens = [iter(tokens) for tokens in token_ids]
        self.vocab_size = vocab_size
        self.batch_sizes: list[int] = []

    def forward(self, decoder_hidden_states, encoder_hidden_states):
        batch_size = encoder_hidden_states.shape[0]
        self.batch_sizes.append(batch_size)
        logits = torch.full((batch_size, self.vocab_size), -1.0)
        for row, request in enumerate(encoder_hidden_states[:, 0].long().tolist()):
            logits[row, next(self.tokens[request])] = 1.0
        return logits


def test_nemotron_rnnt_advances_on_blank_and_symbol_limit() -> None:
    config = _get_tiny_config()
    config.max_symbols_per_step = 2
    blank = config.blank_token_id
    states = [
        Nemotron3_5AsrDecodeState(
            encoder_frames=torch.full((frames, config.decoder_hidden_size), float(i)),
            decoder_cache=Nemotron3_5AsrRNNTDecoderCache(config),
            last_token_id=blank,
        )
        for i, frames in enumerate((3, 1, 2))
    ]
    state = states[0]
    decoder = Nemotron3_5AsrRNNTDecoder(config).eval()
    joint = _ScriptedJoint(
        [[2, blank, blank, 3, 0], [blank], [blank, 4, blank]], config.vocab_size
    )

    with torch.inference_mode():
        assert _decode_next_tokens(
            states[:2],
            decoder,
            joint,
            blank_token_id=blank,
            max_symbols_per_step=2,
        ) == [2, None]
        assert (state.frame_idx, state.symbols_at_frame) == (0, 1)
        assert _decode_next_tokens(
            [state, states[2]],
            decoder,
            joint,
            blank_token_id=blank,
            max_symbols_per_step=2,
        ) == [3, 4]
        assert (state.frame_idx, state.symbols_at_frame) == (2, 1)
        # Blanks preserve initialized state; a fresh blank seed initializes it.
        for row, inputs in ((0, [blank, 2]), (2, [blank])):
            reference_cache = Nemotron3_5AsrRNNTDecoderCache(config)
            for token in inputs:
                decoder(torch.tensor([[token]]), cache=reference_cache)
            for field in ("cache", "hidden_state", "cell_state"):
                torch.testing.assert_close(
                    getattr(states[row].decoder_cache, field),
                    getattr(reference_cache, field),
                )
        assert _decode_next_tokens(
            states,
            decoder,
            joint,
            blank_token_id=blank,
            max_symbols_per_step=2,
        ) == [0, None, None]
        assert (state.frame_idx, state.symbols_at_frame) == (3, 0)
        assert _decode_next_tokens(
            states,
            decoder,
            joint,
            blank_token_id=blank,
            max_symbols_per_step=2,
        ) == [None, None, None]
        state.num_tokens_to_replay = 1
        with pytest.raises(RuntimeError, match="exhausted before replaying"):
            _decode_next_tokens(
                [state],
                decoder,
                joint,
                blank_token_id=blank,
                max_symbols_per_step=2,
            )
    assert joint.batch_sizes == [2, 2, 2, 1, 2]


def test_nemotron_eager_dummy_prepare_profiles_decoder_without_live_mutation() -> None:
    config = _get_tiny_config()
    config.max_symbols_per_step = 1
    model = Nemotron3_5AsrForRNNT(vllm_config=_get_vllm_config(config)).eval()
    state = Nemotron3_5AsrModelState.__new__(Nemotron3_5AsrModelState)
    state.model = model
    state.device = torch.device("cpu")
    state.dtype = torch.float32
    state.request_indices = {"live": 0}
    state.pending_streaming_states = {}
    live_state = Nemotron3_5AsrDecodeState(
        encoder_frames=torch.zeros(1, config.decoder_hidden_size),
        decoder_cache=Nemotron3_5AsrRNNTDecoderCache(config),
        last_token_id=config.blank_token_id,
    )
    state.decode_states = {"live": live_state}
    batch = InputBatch.make_dummy(
        4,
        10,
        InputBuffers(max_num_reqs=4, max_num_tokens=10, device=torch.device("cpu")),
    )

    inputs = state.prepare_inputs(batch, req_states=None)
    assert inputs["query_end_positions"] == batch.query_start_loc_np[1:].tolist()
    assert all(
        decode_state is not live_state for decode_state in inputs["decode_states"]
    )

    with (
        torch.inference_mode(),
        patch.object(model.decoder, "forward", wraps=model.decoder.forward) as decode,
        patch.object(model.joint, "forward", wraps=model.joint.forward) as join,
    ):
        model.forward(batch.input_ids, batch.positions, **inputs)

    assert decode.call_count == 1
    assert decode.call_args.args[0].shape == (batch.num_reqs, 1)
    assert join.call_count == 1
    assert join.call_args.args[0].shape[0] == batch.num_reqs
    assert state.decode_states == {"live": live_state}
    assert live_state.frame_idx == 0
    assert not live_state.decoder_cache.is_initialized


@pytest.mark.parametrize("num_tokens_to_replay", [0, 3])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_nemotron_greedy_tokens_match_transformers(
    num_tokens_to_replay: int, batch_size: int
) -> None:
    torch.manual_seed(0)
    config = _get_tiny_config()
    config.max_symbols_per_step = 2
    hf_model = HFNemotron3_5AsrForRNNT(config).eval()
    model = Nemotron3_5AsrForRNNT(vllm_config=_get_vllm_config(config)).eval()
    model.load_weights(hf_model.state_dict().items())
    features = torch.randn(batch_size, 26, config.encoder_config.num_mel_bins)
    mask = torch.zeros(batch_size, 26, dtype=torch.bool)
    for row, length in enumerate((25, 17, 9)[:batch_size]):
        mask[row, :length] = True
    prompt_ids = torch.arange(2, 2 + batch_size)
    replay_counts = [
        max(0, num_tokens_to_replay - row * 2) for row in range(batch_size)
    ]

    with torch.inference_mode():
        encoded, output_mask = model.audio_encoder(features, mask, prompt_ids)
        assert output_mask is not None
        states = [
            Nemotron3_5AsrDecodeState(
                encoder_frames=encoded[row][output_mask[row]],
                decoder_cache=Nemotron3_5AsrRNNTDecoderCache(config),
                last_token_id=config.blank_token_id,
                num_tokens_to_replay=replay_counts[row],
            )
            for row in range(batch_size)
        ]
        token_ids: list[list[int]] = [[] for _ in states]
        for step in range(64):
            # Admit later rows after the first request already has an LSTM cache.
            ready = [row == 0 or step > 0 for row in range(batch_size)]
            hidden_states = model.forward(
                input_ids=torch.full((batch_size,), config.blank_token_id),
                positions=torch.zeros(batch_size, dtype=torch.long),
                decode_states=states,
                query_end_positions=list(range(1, batch_size + 1)),
                decode_ready=ready,
            )
            selected = model.compute_logits(hidden_states).argmax(dim=-1).tolist()
            for row, token_id in enumerate(selected):
                if ready[row] and token_id != config.blank_token_id:
                    token_ids[row].append(token_id)
            if all(ready) and all(token == config.blank_token_id for token in selected):
                break
        for row, state in enumerate(states):
            assert state.frame_idx == state.encoder_frames.shape[0]
            reference = hf_model.generate(
                input_features=features[row : row + 1],
                attention_mask=mask[row : row + 1],
                prompt_ids=prompt_ids[row : row + 1],
                decoder_start_token_id=config.blank_token_id,
                max_new_tokens=64,
            )
            assert (
                token_ids[row]
                == [
                    token_id
                    for token_id in reference.sequences[0].tolist()
                    if token_id != config.blank_token_id
                ][replay_counts[row] :]
            )


def test_nemotron_loads_all_transformers_weights() -> None:
    config = _get_tiny_config()
    reference = HFNemotron3_5AsrForRNNT(config).eval()
    model = Nemotron3_5AsrForRNNT(vllm_config=_get_vllm_config(config)).eval()

    loaded = model.load_weights(reference.state_dict().items())

    assert loaded == set(model.state_dict())
    for name, tensor in model.state_dict().items():
        source_name = name.removeprefix("audio_encoder.")
        torch.testing.assert_close(tensor, reference.state_dict()[source_name])
