# SPDX-License-Identifier: Apache-2.0
"""Transcribe a complete audio clip through the installed vLLM plugin."""

import argparse
import os

from transformers import AutoConfig
from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.multimodal.media.audio import load_audio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "audio", nargs="?", help="Audio file; defaults to a speech sample"
    )
    parser.add_argument(
        "--model",
        default="nvidia/nemotron-3.5-asr-streaming-0.6b",
        help="Nemotron ASR checkpoint to load",
    )
    parser.add_argument("--language", default="en")
    args = parser.parse_args()
    audio = (
        load_audio(args.audio, sr=16_000)
        if args.audio
        else AudioAsset("mary_had_lamb").audio_and_sample_rate
    )
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
    llm = LLM(
        model=args.model,
        enforce_eager=True,
        dtype="float16",
        max_model_len=512,
        max_num_seqs=4,
        gpu_memory_utilization=0.5,
    )
    prompt = {
        "prompt": "",
        "multi_modal_data": {"audio": audio},
    }
    if AutoConfig.from_pretrained(args.model).model_type == "nemotron3_5_asr":
        prompt["mm_processor_kwargs"] = {"language": args.language}
    output = llm.generate(
        prompt,
        SamplingParams(temperature=0, max_tokens=511),
        use_tqdm=False,
    )
    if output[0].outputs[0].finish_reason == "length":
        raise RuntimeError(
            "Transcript reached the token limit; increase max_model_len."
        )
    print(output[0].outputs[0].text)


if __name__ == "__main__":
    main()
