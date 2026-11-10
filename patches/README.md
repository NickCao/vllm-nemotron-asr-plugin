# Required vLLM core patch

Streaming encoder-decoder sessions add a new encoder input for every audio
chunk. The vLLM scheduler currently assumes that every encoder input of an
encoder-decoder request lives at position 0 and has already been computed once
`num_computed_tokens > 0`, so later chunks trip this assertion:

```text
AssertionError: Encoder input should be processed at the beginning of the
sequence when encoder-decoder models are used.
```

`vllm-resumable-encoder-decoder.patch` relaxes that invariant for resumable
requests only, allowing each streaming chunk to be scheduled normally.
Non-resumable encoder-decoder requests keep the existing validation.

Apply it to the vLLM checkout used to build the environment:

```bash
git -C /path/to/vllm apply /path/to/vllm-resumable-encoder-decoder.patch
```

The patch was validated against vLLM 0.33.0 with the Model Runner V2
resumable-request implementation. Remove it once equivalent support is
available upstream (vLLM RFC #65543 / PR #65544).
