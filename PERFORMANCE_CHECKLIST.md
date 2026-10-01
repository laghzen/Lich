# iPC / RTX 3060 Laptop GPU — performance master checklist

Target platform: NVIDIA GeForce RTX 3060 Laptop GPU, SM86 `(8, 6)`, 6 GiB VRAM. The project intentionally targets one fixed GPU so tuning decisions can be specialized to this machine.

## 1. Correctness and paper parity

- [x] PyTorch/FP32 reference equations.
- [x] `x[0]` target and `x[L]` input convention fixed.
- [x] Correct `W_lower[K,M]` inference layout.
- [x] Tail-safe prediction/error path.
- [x] Tail-safe activation recomputation, including `f(0) != 0` activations.
- [x] Core SM86 smoke coverage for regular/tail/save-A paths.
- [ ] Complete numerical comparison for every supported activation.
- [ ] Reproduce MNIST MLP 784→64→64→10.
- [ ] Reproduce paper-style discriminative efficiency sweep (hidden=64, depths 3/4/6).
- [ ] 5-seed paper-style convergence/generalization study with all implementation details pinned.

## 2. Memory lifetime

- [x] Persistent W/X/E buffers.
- [x] In-place X update.
- [x] No global `mu` materialization.
- [x] No global `W^T E` inference-signal materialization.
- [x] No global `f'(X)` materialization.
- [x] Save-A vs recompute-A paths.
- [x] Recompute policy selected by Stage 2 execution search.
- [ ] Full automatic lifetime/resource cost model.
- [ ] Automatic register/shared/recompute placement model.

## 3. Kernel fusion

- [x] Activation fused into prediction.
- [x] Prediction + error fused.
- [x] Derivative fused into inference update.
- [x] X update fused with inference epilogue.
- [x] Activation recomputation fused into weight-gradient path.
- [ ] Fused optimizer epilogue for the selected optimizer.
- [x] Layer-grouped execution kernels.
- [ ] Safe iteration-level fusion beyond current CUDA Graph path.

## 4. Tensor Core / tiling / kernel autotuning

- [x] Target `sm_86`.
- [x] FP16 storage / FP32 accumulation baseline.
- [x] BM search through locked Stage 1.19 v12.
- [x] BN search through locked Stage 1.19 v12.
- [x] BK search through locked Stage 1.19 v12.
- [x] Thread-count search through locked Stage 1.19 v12.
- [x] Pipeline-stage search through locked Stage 1.19 v12.
- [x] Shared-memory swizzle search through locked Stage 1.19 v12.
- [x] CTA/grid policy search in Stage 2.
- [x] Independent P/I/W kernel tables.
- [ ] Register-pressure pruning.
- [ ] Generated-kernel register-spill detection integrated into acceptance.
- [ ] Resource/occupancy-aware scoring.

## 5. Adaptive autotuning

- [x] Deterministic static legality filtering.
- [x] Full legal finite-space representation.
- [x] Semantic-tree exploration.
- [x] Online EI/local racing.
- [x] Global scouts.
- [x] Factorized local surrogate.
- [x] Conservative region pruning.
- [x] Persistent failure memory.
- [x] Fresh-run discovery without requiring persistent measurement cache.
- [x] Locked v12 reference engine.
- [ ] Do not claim mathematical global optimum from a finite adaptive sample.

## 6. Level 2 execution policy

- [x] Group identical hidden-layer shapes.
- [x] `grid.z` layer batching.
- [x] Shape/depth-dependent Z policy.
- [x] CUDA Graph A/B.
- [x] Hierarchical search over P/I/W kernel alternatives plus execution parameters.
- [x] Recompute-policy search.
- [x] Fresh Level-1 top-K carried into Level 2.
- [x] Level-2 execution results keyed by exact kernel signature.
- [x] Two coordinate passes and pairwise rescue surface.
- [x] Final exhaustive topology matrix for the selected kernel table.

## 7. Runtime / steady state

- [x] Reusable GPU allocations.
- [x] CUDA Graph capture scaffold.
- [x] CUDA Graph benchmark.
- [x] Stage 2.3 runtime execution-plan caching: kernel handles prepared once and reused across `step_initialized()`.
- [x] Per-batch active-kernel-signature caching.
- [x] Per-batch recompute-policy caching.
- [x] One-time grid-z policy application per batch within a trainer lifetime.
- [ ] Complete steady-state graph integration in the public training CLI.
- [ ] Pinned host memory where host staging exists.
- [ ] Async H2D prefetch where input pipeline exists.
- [ ] CPU/GPU input pipeline overlap.

## 8. Weight update specialization

- [x] GEMM path for `E @ A^T`.
- [ ] Small-B outer-product alternative.
- [ ] Empirical crossover `B*` between weight-update implementations.
- [ ] Fused SGD epilogue.
- [ ] Fused AdamW epilogue.
- [ ] Eliminate any remaining unnecessary global gradient materialization.

## 9. Thermal / sustained acceptance

- [~] Sustained benchmark on the real laptop GPU.
- [ ] Per-machine sustainable clock profile.
- [ ] Reject cold-start-only winners.
- [ ] Temperature-aware autotuning objective.
- [ ] Long-run (>10 min) final stability verification for the accepted policy.
- [ ] Final acceptance must use sustained end-to-end throughput, not cold-start kernel latency.

## 10. Historical performance notes

- Historical runs have observed approximately `0.0152 ms` for one prediction workload, but this is not the current reproducible acceptance result and must not be treated as a guaranteed optimum.
- Current clean-run Level-1 examples have reached approximately `0.0174–0.0184 ms` on the 64×64×128 prediction workload, depending on run conditions.
- Stage 2 execution benchmarks have demonstrated substantial launch-overhead reductions with CUDA Graph, but those benchmark coefficients must not be arithmetically multiplied with independent Stage-1 speedups.

## 11. Final acceptance gate

An accepted configuration must satisfy all of:

- numerically correct against the configured reference tolerance;
- legal and free of intermittent races/access faults;
- warm end-to-end throughput measured on the target GPU;
- VRAM within the 6 GiB budget with safety margin;
- stable under sustained load;
- no thermal slowdown during the acceptance window;
- final policy corresponds to the exact kernel/runtime fingerprint that was measured.
