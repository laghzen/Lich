# Stage 1.2 — TileLang 0.1.13 pipeline fix

This stage addresses the first real RTX 3060 compile report.

## Fixes

1. Removed the custom `_copy_global_tile` helper. It used `T.Parallel` from a Python helper path that fails in the TileLang 0.1.13 eager lowering path.
2. Prediction path uses distinct `a_raw` and `a_shared` buffers. `T.copy` writes the raw tile once; a `T.Parallel` transform writes the activated tile once; GEMM consumes the activated tile.
3. Recompute-activation weight update uses the same two-buffer pattern.
4. Inference preserves the mathematically correct `W_lower[K,M]` layout and transposes the shared tile in GEMM.
5. Tail activation writes explicit zero for out-of-range elements, so nonzero-at-zero activations do not contaminate padded GEMM regions.
6. TileLang is pinned to 0.1.13.

## Why the pipeline error occurred

TileLang's software pipeline planner rejects multiple writes to overlapping regions of the same pipelined buffer when those writes land in different pipeline stages. The original kernel copied into `a_shared` and then overwrote `a_shared` with the activation inside the same `T.Pipelined` loop. The new version separates the producer and transformed consumer buffers.

## Next performance stage

After smoke correctness passes on SM86, benchmark:

- `num_stages = 1,2,3`
- `BM/BN/BK`
- `threads`
- shared-memory swizzle
- CTA rasterization
- saved activation vs recomputation

Then implement small-B outer-product and layer-grouped/grid.z execution.
