# Evaluation Scripts

## FLEURS evaluation

Install evaluation dependencies and run:

```bash
uv pip install -e '.[eval]'
python scripts/eval_fleurs.py --language en_us --num-samples 10
```

Example for another language:

```bash
python scripts/eval_fleurs.py --language de_de --num-samples 10
```

The script evaluates complete-audio transcription on the FLEURS test split using the installed vLLM plugin and reports WER and CER.

## Realtime transcription

Start a vLLM server with Model Runner V2, then stream the default microphone
through the OpenAI-compatible WebSocket endpoint:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve \
    nvidia/nemotron-3.5-asr-streaming-0.6b \
    --served-model-name nemotron \
    --enforce-eager \
    --dtype float16
```

```bash
uv pip install -e '.[mic]'
python scripts/realtime_transcribe.py \
    --url ws://127.0.0.1:8000/v1/realtime \
    --model nemotron
```

Transcript deltas are printed as they arrive. Press Ctrl+C to stop the
microphone and finish the session. Use `--list-devices` to find a microphone
and `--device` to select one.

A 16 kHz mono WAV file can still be streamed by passing it as the positional
argument. WAV input is played at real-time speed by default so the run behaves
like a live microphone; pass `--no-pace` to send it as fast as possible. Use
`--duration` to stop the microphone after a fixed number of seconds.

## Package version

Preview or update the package version:

```bash
python scripts/bump_version.py --dry-run
python scripts/bump_version.py patch
python scripts/bump_version.py minor
python scripts/bump_version.py major
python scripts/bump_version.py --set 0.2.0
```

The script updates `[project].version` in `pyproject.toml`. It does not create commits, tags, or releases.
