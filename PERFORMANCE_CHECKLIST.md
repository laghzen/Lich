# iPC / RTX 3060 — performance master checklist

Legend: `[x]` implemented, `[~]` scaffolded/needs measurement on the real GPU, `[ ]` not implemented yet.

## 1. Correctness and paper parity
- [x] PyTorch/FP32 reference equations.
- [x] Fixed `x[0]` target and `x[L]` input convention.
- [x] Correct `W_lower[K,M]` inference layout.
- [x] Tail-safe prediction/error path.
- [x] Tail-safe recompute activation path, including `f(0) != 0` activations.
- [ ] Full SM86 numerical comparison for every activation.
- [ ] Reproduce MNIST MLP 784→64→64→10.
- [ ] Reproduce discriminative efficiency sweep with 64 hidden units, depths 3/4/6.
- [ ] 5-seed paper-style accuracy/convergence study where all implementation details are pinned.

## 2. Memory lifetime
- [x] Persistent W/X/E buffers.
- [x] In-place X update.
- [x] Do not materialize `mu`.
- [x] Do not materialize `W^T E` inference signal.
- [x] Do not materialize `f'(X)`.
- [x] Save-A vs recompute-A switch.
- [ ] Automatic save-vs-recompute cost model.
- [ ] Automatic register/shared/recompute placement.
- [ ] Full intermediate lifetime analysis.

## 3. Kernel fusion
- [x] Activation fused into prediction path.
- [x] Prediction and error fused.
- [x] Derivative fused into inference update.
- [x] X update fused with inference epilogue.
- [x] Activation recomputation fused into weight-gradient path.
- [ ] Fused optimizer epilogue.
- [ ] Layer-grouped fused execution.
- [ ] Experiment with safe iteration-level fusion.

## 4. Tensor Core / tiling
- [x] Target `sm_86`.
- [x] FP16 storage / FP32 accumulation baseline.
- [~] BM search.
- [~] BN search.
- [~] BK search.
- [~] thread-count search.
- [~] pipeline-stage search.
- [~] shared-memory swizzle search.
- [~] CTA rasterization search.
- [ ] Register-pressure pruning.
- [ ] Detect register spills from generated CUDA/PTX.
- [ ] Independent autotune tables for prediction / inference / weight update.

## 5. Weight update
- [x] GEMM path for `E @ A^T`.
- [ ] Small-B outer-product path.
- [ ] Empirical crossover B*.
- [ ] Fused SGD.
- [ ] Fused AdamW.
- [ ] Gradient never materialized globally.

## 6. Layer parallelism
- [ ] Group identical layer shapes.
- [ ] `grid.z = layer/group` kernels.
- [ ] Separate kernels for small/medium/large layers.
- [ ] L2-aware CTA scheduling.
- [ ] Layer-batched repeated blocks.

## 7. Runtime
- [x] Reusable GPU allocations.
- [x] CUDA Graph scaffold.
- [ ] Graph benchmark vs normal launches.
- [ ] Pinned host memory.
- [ ] Async H2D prefetch.
- [ ] CPU/GPU pipeline overlap.
- [ ] Remove all per-step Python overhead from the steady-state path.

## 8. Thermal / power stability
- [x] Rate-limited telemetry.
- [x] Temperature ceiling + hysteresis.
- [x] Throttle-reason monitoring.
- [~] 120 s sustained benchmark.
- [ ] Per-machine sustainable clock profile.
- [ ] Reject cold-start-only winners.
- [ ] Temperature-aware autotuning objective.
- [ ] Long-run (>10 min) stability verification.

## 9. Acceptance criteria
A configuration is accepted only when all are true:

- Correct against FP32 reference within the configured numerical tolerance.
- Warmed-up end-to-end iPC throughput is measured.
- No register spills that materially damage performance.
- No illegal memory access / race / intermittent correctness failure.
- VRAM remains within the 6 GiB device budget with a safety margin.
- After sustained load, temperature remains below the project ceiling.
- No thermal slowdown is observed in the accepted sustained run.
- The configuration wins on sustained end-to-end throughput, not on cold-start kernel latency.

## Stage 1.4 Windows/bootstrap
- [x] Fix nested cmd.exe quoting for vcvars64.bat.
- [x] Add separated-source bootstrap and `run_cli.py`.
- [ ] Toolchain probe passes with native MSVC cl.exe.

[x] fixed vcvars64 invocation through temporary .cmd
[x] explicit selected MSVC bin first in PATH
[x] toolchain diagnostic reports selected compiler
[ ] successful toolchain probe on user Windows host


## Stage 1.7
[x] TileLang JIT receives explicit `-ccbin=<MSVC cl.exe>`
[ ] TileLang prediction kernel compiles/execututes with native MSVC
