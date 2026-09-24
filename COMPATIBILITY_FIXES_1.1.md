# Stage 1.1 fixes

- Target the installed TileLang 0.1.13.
- Remove `from __future__ import annotations` from TileLang kernel module; TileLang eager JIT type resolution has a known incompatibility with that pattern.
- Make the CLI forward subcommand arguments directly, so `cli.py smoke --M 64 ...` works without a `--` separator.
- Fix inference pipeline while preserving the mathematically correct lower-layer weight shape `[K,M]`; shared memory stores `[BK,BM]` tiles and `T.gemm(..., transpose_A=True)` computes `[M,K] @ [K,B]`.

## Stage 1.2 fixes

- Remove the custom `_copy_global_tile` helper; its `T.Parallel` use is incompatible with this eager lowering path. Native `T.copy` is used so TileLang can legalize boundary accesses.
- Prediction and recompute-activation weight-update kernels no longer overwrite a pipelined shared buffer after a `T.copy`. A raw shared tile and transformed shared tile are separate buffers.
- Preserve the correct `W_lower[K,M]` inference layout; no transpose is performed in global memory.

## Stage 1.2.1 correctness detail

- Tail activation tiles explicitly write zero for out-of-range elements before GEMM. This preserves zero-padding for activations such as sigmoid/tanh where `f(0) != 0`.
