# Required vLLM core patch

Streaming encoder-decoder sessions add a new encoder input for every audio
chunk, each positioned at the chunk's offset in the decoder sequence. Upstream
vLLM assumes the opposite: that every encoder input of an encoder-decoder
request lives at position 0 and has already been computed once
`num_computed_tokens > 0`. Two places in `_try_schedule_encoder_inputs` encode
that assumption:

1. The encoder loop is skipped once any token has been computed, and asserts
   `start_pos == 0` if it is not skipped.
2. The scan start is clamped to `lo = 0` for every encoder-decoder request, so
   a per-chunk layout re-walks the whole history on every pass.

`vllm-resumable-encoder-decoder.patch` relaxes both for `request.resumable`
only. Non-resumable encoder-decoder requests (e.g. Whisper) keep the existing
behaviour and validation: their encoder inputs really do live at position 0.

Fixing only the first hunk is not enough. The scheduler then schedules — and
charges the encoder cache for — every historical chunk on every step. Encoder
work becomes O(session length) per emitted token, and the per-pass requirement
`nfeat * 7` embeds reaches `encoder_cache_size` (8192 by default). When it
does, the request fails `can_allocate` on a historical feature whose
`start_pos < num_computed_tokens`, so `num_new_tokens` is clamped to 0, the
request is skipped, no model step runs, nothing is freed, and the engine spins
forever with `Running: 0` until restart.

Apply it to the vLLM checkout used to build the environment:

```bash
git -C /path/to/vllm apply /path/to/vllm-resumable-encoder-decoder.patch
```

The patch was validated against vLLM 0.33.0 with the Model Runner V2
resumable-request implementation. Remove it once equivalent support is
available upstream (vLLM RFC #65543 / PR #65544).

## Validation

RTX PRO 6000 Blackwell, `--enforce-eager --dtype float16 --max-model-len 512
--max-num-seqs 128`, two realtime streams replaying 300 s of audio in
wall-clock time. `--max-num-batched-tokens 1024` shrinks the encoder cache 8x
so the terminal state is reached in seconds instead of minutes.

| Patch | Result |
|---|---|
| first hunk only | stalled after ~20 s: `free_end=86`, `admits=0 skips=2 fails=2`, 14,604 empty scheduler passes, 0 % GPU, client hung |
| both hunks | `ok=2 err=0 stalls=0`, 300.1 s per stream, zero encoder-allocation failures |

With both hunks the free-slot counter stops constraining scheduling but
inflates past the cache size (observed >1.1 M against 8192), because of the
accounting limitation below.

## Limitations

- **Encoder-cache accounting is still upstream's single-input model.**
  `EncoderDecoderCacheManager` assumes one encoder input per request:
  `get_cached_input_ids()` returns every feature id, and
  `free_encoder_input()` credits slots for every id without checking what was
  actually allocated. With one feature per chunk and this patch, the counter
  over-frees without bound. Scheduling fails open instead of deadlocking, but
  the encoder budget no longer limits anything. A production patch needs
  per-request allocated-id tracking so alloc/free stay paired.
- **`Request.mm_features` grows without bound.** One entry is appended per
  audio chunk and never trimmed, so scheduler memory and per-pass bookkeeping
  grow with session length. Compacting the list needs care because the runner
  reads features by index.
- **Session context is not bounded.** A five-minute stream reached 745 decoder
  tokens with `--max-model-len 512` and still completed. This patch fixes
  scheduling, not context management; the resumable path does not enforce the
  model-length limit.
- **Single-target validation.** Only vLLM 0.33.0 plus Model Runner V2 was
  exercised. The resumable encoder-decoder combination does not exist
  upstream, so this stays a stop-gap.
