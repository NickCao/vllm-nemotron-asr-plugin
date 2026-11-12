# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import (
    NemotronAsrStreamingConfig,
    NemotronAsrStreamingEncoderConfig,
    NemotronAsrStreamingProcessor,
    PreTrainedTokenizerFast,
)
from transformers import (
    NemotronAsrStreamingForRNNT as HFNemotronAsrStreamingForRNNT,
)
from transformers.models.nemotron_asr_streaming.generation_nemotron_asr_streaming import (  # noqa: E501
    NemotronAsrStreamingRNNTDecoderCache,
)
from transformers.models.nemotron_asr_streaming.modeling_nemotron_asr_streaming import (
    NemotronAsrStreamingRNNTDecoder,
    NemotronAsrStreamingRNNTJointNetwork,
)

from vllm_nemotron_asr_plugin import register
from vllm_nemotron_asr_plugin.config import NemotronAsrStreamingForRNNTConfig
from vllm_nemotron_asr_plugin.model import (
    Nemotron3_5AsrDecodeState,
    NemotronAsrStreamingForRNNT,
    _decode_next_tokens,
    decoder_cache_cls,
)
from vllm_nemotron_asr_plugin.processing import Nemotron3_5AsrProcessingInfo


def _tiny_english_config() -> NemotronAsrStreamingConfig:
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
    return NemotronAsrStreamingConfig(
        encoder_config=encoder_config,
        decoder_hidden_size=8,
        vocab_size=16,
        blank_token_id=15,
        num_decoder_layers=1,
        max_symbols_per_step=2,
    )


def _build_model(
    config: NemotronAsrStreamingConfig,
) -> NemotronAsrStreamingForRNNT:
    return NemotronAsrStreamingForRNNT(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(hf_config=config),
        )
    )


def _stt_params(
    config: NemotronAsrStreamingConfig, language: str | None
) -> SimpleNamespace:
    return SimpleNamespace(
        task_type="transcribe",
        request_prompt=None,
        hotwords=None,
        to_language=None,
        language=language,
        audio=np.zeros(1600, dtype=np.float32),
        stt_config=SimpleNamespace(sample_rate=16_000),
        model_config=SimpleNamespace(hf_config=config),
    )


def test_english_config_hook() -> None:
    config = SimpleNamespace(
        hf_config=SimpleNamespace(decoder_hidden_size=640),
        model_arch_config=SimpleNamespace(hidden_size=1024),
        tokenizer_mode="auto",
        max_logprobs=20,
    )

    NemotronAsrStreamingForRNNTConfig.verify_and_update_model_config(config)

    assert config.model_arch_config.hidden_size == 640
    assert config.tokenizer_mode == "nemotron_asr_streaming"
    assert config.max_logprobs == 0


def test_english_checkpoint_registered() -> None:
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP
    from vllm.renderers.registry import RENDERER_REGISTRY
    from vllm.tokenizers.registry import TokenizerRegistry

    from vllm_nemotron_asr_plugin.processing import (
        NemotronAsrStreamingRenderer,
    )

    register()

    assert (
        MODELS_CONFIG_MAP["NemotronAsrStreamingForRNNT"]
        is NemotronAsrStreamingForRNNTConfig
    )
    assert (
        RENDERER_REGISTRY.load_renderer_cls("nemotron_asr_streaming")
        is NemotronAsrStreamingRenderer
    )
    assert (
        TokenizerRegistry.load_tokenizer_cls("nemotron_asr_streaming").__name__
        == "CachedHfTokenizer"
    )


def test_english_audio_encoder_has_no_prompt_projector() -> None:
    torch.manual_seed(0)
    config = _tiny_english_config()
    model = _build_model(config).eval()

    assert model.audio_encoder.prompt_projector is None
    assert isinstance(model.decoder, NemotronAsrStreamingRNNTDecoder)
    assert isinstance(model.joint, NemotronAsrStreamingRNNTJointNetwork)

    input_features = torch.randn(2, 12, 8)
    attention_mask = torch.ones(2, 12, dtype=torch.bool)
    with torch.inference_mode():
        output, output_mask = model.audio_encoder(
            input_features,
            attention_mask,
            prompt_ids=None,
        )
        embeddings = model.embed_multimodal(
            input_features=input_features,
            attention_mask=attention_mask,
        )

    assert output.shape[0] == 2
    assert output.shape[-1] == config.decoder_hidden_size
    assert output_mask is not None
    assert len(embeddings) == 2


def test_english_decoder_cache_class() -> None:
    assert (
        decoder_cache_cls(_tiny_english_config())
        is NemotronAsrStreamingRNNTDecoderCache
    )


@pytest.mark.parametrize("language", [None, "auto", "en", "en-US", "EN"])
def test_english_generation_prompt_omits_language(language: str | None) -> None:
    config = _tiny_english_config()
    prompt = NemotronAsrStreamingForRNNT.get_generation_prompt(
        _stt_params(config, language)
    )

    assert prompt["encoder_prompt"]["mm_processor_kwargs"] == {}
    assert prompt["decoder_prompt"]["prompt_token_ids"] == [config.blank_token_id]


def test_english_rejects_non_english_language() -> None:
    config = _tiny_english_config()
    with pytest.raises(ValueError, match="English"):
        NemotronAsrStreamingForRNNT.get_generation_prompt(_stt_params(config, "de"))


def test_english_processor_has_no_prompt_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    from vllm_nemotron_asr_plugin import processing

    backend = Tokenizer(
        WordLevel(
            {"<pad>": 0, "<unk>": 1, "hello": 2, "<blank>": 3},
            unk_token="<unk>",
        )
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
    )
    config = _tiny_english_config()
    ctx = SimpleNamespace(
        model_config=SimpleNamespace(
            model="test-model",
            revision=None,
            multimodal_config=SimpleNamespace(mm_processor_kwargs={}),
        ),
        get_hf_config=lambda typ: config,
    )
    info = Nemotron3_5AsrProcessingInfo(ctx)
    info.get_tokenizer = lambda: tokenizer
    monkeypatch.setattr(
        processing,
        "get_hf_file_to_dict",
        lambda *args, **kwargs: {
            "blank_token": "<blank>",
            "default_num_lookahead_tokens": 3,
            "supported_num_lookahead_tokens": [3, 1],
            "feature_extractor": {
                "feature_size": 8,
                "sampling_rate": 16_000,
                "hop_length": 160,
                "n_fft": 512,
                "win_length": 400,
            },
        },
    )

    processor = info.get_hf_processor()
    processed = processor(
        np.zeros(16_000, dtype=np.float32),
        sampling_rate=16_000,
    )

    assert isinstance(processor, NemotronAsrStreamingProcessor)
    assert processor.blank_token_id == 3
    assert processed["num_lookahead_tokens"] == 3
    assert "prompt_ids" not in processed


def test_english_greedy_decode_matches_transformers() -> None:
    torch.manual_seed(0)
    config = _tiny_english_config()
    reference = HFNemotronAsrStreamingForRNNT(config).eval()
    model = _build_model(config).eval()

    loaded = model.load_weights(list(reference.state_dict().items()))
    assert loaded == {name for name, _ in model.named_parameters()}

    input_features = torch.randn(1, 40, config.encoder_config.num_mel_bins)
    attention_mask = torch.ones(1, 40, dtype=torch.bool)
    with torch.inference_mode():
        reference_encoder = reference.get_audio_features(
            input_features=input_features,
            attention_mask=attention_mask,
            num_lookahead_tokens=3,
        )
        frames, _ = model.audio_encoder(
            input_features,
            attention_mask,
            prompt_ids=None,
            num_lookahead_tokens=3,
        )
        expected = reference.generate(
            input_features,
            attention_mask=attention_mask,
            num_lookahead_tokens=3,
            decoder_start_token_id=config.blank_token_id,
            max_new_tokens=64,
        )

    torch.testing.assert_close(reference_encoder.pooler_output, frames)

    state = Nemotron3_5AsrDecodeState(
        encoder_frames=frames[0],
        decoder_cache=decoder_cache_cls(config)(config),
        last_token_id=config.blank_token_id,
    )
    tokens: list[int] = []
    with torch.inference_mode():
        for _ in range(config.max_symbols_per_step * frames.shape[1] + 1):
            token_id = _decode_next_tokens(
                [state],
                model.decoder,
                model.joint,
                blank_token_id=config.blank_token_id,
                max_symbols_per_step=config.max_symbols_per_step,
            )[0]
            if token_id is None:
                break
            tokens.append(token_id)

    assert [config.blank_token_id, *tokens] == expected.sequences[0].tolist()
