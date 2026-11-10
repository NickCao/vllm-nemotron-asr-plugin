#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stream microphone or WAV audio through vLLM's /v1/realtime endpoint."""

import argparse
import asyncio
import base64
import json
import signal
import sys
import time
from collections.abc import AsyncIterator

import numpy as np
import soundfile as sf
import websockets


def load_pcm16(path: str) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != 16_000:
        raise ValueError(f"Expected 16 kHz audio, got {sample_rate} Hz.")
    if audio.shape[1] != 1:
        raise ValueError(f"Expected mono audio, got {audio.shape[1]} channels.")
    return np.clip(audio[:, 0] * 32768.0, -32768, 32767).astype(np.int16)


def encode_audio(audio: np.ndarray) -> str:
    return base64.b64encode(audio.tobytes()).decode()


def iter_file_chunks(
    audio: np.ndarray,
    chunk_samples: int,
    *,
    realtime: bool,
) -> AsyncIterator[np.ndarray]:
    async def chunks() -> AsyncIterator[np.ndarray]:
        started_at = time.monotonic()
        streamed_samples = 0
        for offset in range(0, audio.size, chunk_samples):
            chunk = audio[offset : offset + chunk_samples]
            if realtime:
                streamed_samples += chunk.size
                delay = started_at + streamed_samples / 16_000 - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
            yield chunk

    return chunks()


def iter_microphone_chunks(
    chunk_samples: int,
    device: int | str | None,
    stop_event: asyncio.Event,
) -> AsyncIterator[np.ndarray]:
    async def chunks() -> AsyncIterator[np.ndarray]:
        import sounddevice as sd

        loop = asyncio.get_running_loop()
        audio_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=16)

        def enqueue(audio: np.ndarray) -> None:
            if audio_queue.full():
                audio_queue.get_nowait()
            audio_queue.put_nowait(audio)

        def callback(indata, frames, time_info, status) -> None:
            del frames, time_info
            if status:
                print(f"Microphone: {status}", file=sys.stderr)
            loop.call_soon_threadsafe(enqueue, indata[:, 0].copy())

        with sd.InputStream(
            samplerate=16_000,
            channels=1,
            dtype="int16",
            blocksize=chunk_samples,
            device=device,
            callback=callback,
        ):
            while not stop_event.is_set():
                try:
                    chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                except TimeoutError:
                    continue
                yield chunk

    return chunks()


async def receive_deltas(websocket) -> None:
    while True:
        event = json.loads(await websocket.recv())
        if event["type"] == "transcription.delta":
            print(event["delta"], end="", flush=True)
        elif event["type"] == "transcription.done":
            print()
            return
        elif event["type"] == "error":
            raise RuntimeError(event["error"])


async def stream_audio(
    websocket,
    chunks: AsyncIterator[np.ndarray],
    start_after_samples: int,
    stop_event: asyncio.Event,
) -> None:
    buffered_samples = 0
    started = False
    async for chunk in chunks:
        if stop_event.is_set():
            break
        await websocket.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": encode_audio(chunk),
                }
            )
        )
        buffered_samples += chunk.size
        if not started and buffered_samples >= start_after_samples:
            await websocket.send(
                json.dumps({"type": "input_audio_buffer.commit", "final": False})
            )
            started = True

    if not buffered_samples and not stop_event.is_set():
        raise RuntimeError("No audio was captured.")
    if not started and buffered_samples:
        await websocket.send(
            json.dumps({"type": "input_audio_buffer.commit", "final": False})
        )
    await websocket.send(
        json.dumps({"type": "input_audio_buffer.commit", "final": True})
    )


async def transcribe(
    url: str,
    model: str,
    chunks: AsyncIterator[np.ndarray],
    *,
    start_after_ms: int,
    stop_event: asyncio.Event,
) -> None:
    start_after_samples = 16_000 * start_after_ms // 1000
    async with websockets.connect(url, max_size=None) as websocket:
        created = json.loads(await websocket.recv())
        if created["type"] != "session.created":
            raise RuntimeError(f"Unexpected event: {created}")
        await websocket.send(json.dumps({"type": "session.update", "model": model}))

        async with asyncio.TaskGroup() as task_group:
            task_group.create_task(
                stream_audio(
                    websocket,
                    chunks,
                    start_after_samples,
                    stop_event,
                )
            )
            task_group.create_task(receive_deltas(websocket))


async def run(args: argparse.Namespace) -> None:
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop_event.set)
    if args.duration is not None:
        loop.call_later(args.duration, stop_event.set)

    chunk_samples = 16_000 * args.chunk_ms // 1000
    if args.audio:
        chunks = iter_file_chunks(
            load_pcm16(args.audio),
            chunk_samples,
            realtime=args.pace,
        )
    else:
        chunks = iter_microphone_chunks(chunk_samples, args.device, stop_event)
    await transcribe(
        args.url,
        args.model,
        chunks,
        start_after_ms=args.start_after_ms,
        stop_event=stop_event,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "audio",
        nargs="?",
        help="Optional 16 kHz mono WAV file; uses the microphone when omitted",
    )
    parser.add_argument("--url", default="ws://127.0.0.1:8000/v1/realtime")
    parser.add_argument("--model", default="nemotron")
    parser.add_argument("--chunk-ms", type=int, default=250)
    parser.add_argument("--start-after-ms", type=int, default=500)
    parser.add_argument(
        "--pace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Play WAV input at real-time speed (default: enabled)",
    )
    parser.add_argument("--duration", type=float, help="Stop after this many seconds")
    parser.add_argument("--device", help="Microphone device name or index")
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit",
    )
    args = parser.parse_args()
    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return
    if args.chunk_ms <= 0 or args.start_after_ms < 0:
        parser.error("--chunk-ms must be positive and --start-after-ms nonnegative.")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive.")
    if isinstance(args.device, str) and args.device.isdigit():
        args.device = int(args.device)

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
