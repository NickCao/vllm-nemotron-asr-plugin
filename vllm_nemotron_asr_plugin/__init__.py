# SPDX-License-Identifier: Apache-2.0
"""Nemotron ASR model plugin for vLLM."""


def register() -> None:
    from vllm import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP
    from vllm.renderers.registry import RENDERER_REGISTRY
    from vllm.tokenizers.registry import TokenizerRegistry

    from .config import (
        Nemotron3_5AsrForRNNTConfig,
        NemotronAsrStreamingForRNNTConfig,
    )

    MODELS_CONFIG_MAP["Nemotron3_5AsrForRNNT"] = Nemotron3_5AsrForRNNTConfig
    MODELS_CONFIG_MAP["NemotronAsrStreamingForRNNT"] = NemotronAsrStreamingForRNNTConfig
    for tokenizer_mode, renderer in (
        ("nemotron3_5_asr", "Nemotron3_5AsrRenderer"),
        ("nemotron_asr_streaming", "NemotronAsrStreamingRenderer"),
    ):
        TokenizerRegistry.register(
            tokenizer_mode, "vllm.tokenizers.hf", "CachedHfTokenizer"
        )
        RENDERER_REGISTRY.register(
            tokenizer_mode,
            "vllm_nemotron_asr_plugin.processing",
            renderer,
        )
    ModelRegistry.register_model(
        "Nemotron3_5AsrForRNNT",
        "vllm_nemotron_asr_plugin.model:Nemotron3_5AsrForRNNT",
    )
    ModelRegistry.register_model(
        "NemotronAsrStreamingForRNNT",
        "vllm_nemotron_asr_plugin.model:NemotronAsrStreamingForRNNT",
    )
