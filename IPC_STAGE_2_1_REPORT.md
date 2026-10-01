# iPC — Stage 2.1 Hierarchical Autotuning Report

**Target:** NVIDIA GeForce RTX 3060 Laptop GPU, SM 8.6, 6 GiB VRAM  
**Project:** Lich / iPC  
**Stage:** 2.1 — hierarchical execution + locked Stage 1.19 v12  
**Status:** ACCEPTED / COMPLETE

## 1. Goal

Stage 2.1 adds a second optimization level around the already working per-kernel autotuner.
The purpose is to optimize the real iPC execution topology rather than to expand or redesign the
kernel search itself.

The stage combines the measured P/I/W kernel table from the locked Stage 1.19 v12 engine with
execution parameters such as `grid.z`, recomputation policy and CUDA Graph, and verifies whether
kernel alternatives remain beneficial when embedded into a full iPC step.

## 2. Locked Level-1 foundation

Stage 1.19 v12 is treated as a frozen reference engine.
No search-budget, legality, acquisition or stopping logic is changed by Stage 2.1.

The v12 search represents the complete finite legal frontier and adaptively launches only selected
candidates. Fresh Level-1 discovery is supported and does not require a persistent measurement cache.

A representative clean-run result for prediction `64x64x128` reached approximately `0.0174 ms` with:

```text
block_m=16
block_n=32
block_k=16
threads=64
num_stages=3
swizzle=False
swizzle_panel=8
shared_swizzle=False
```

This historical measurement is recorded as a measured run, not as a mathematical global optimum.

## 3. Stage 2.1 architecture

```text
Locked Stage 1.19 v12
        |
        +--> independent P / I / W top-K tables
        |
        v
full iPC execution configuration
        |
        +--> depth
        +--> grid.z cap
        +--> recompute activation
        +--> CUDA Graph
        +--> kernel interaction candidates
        |
        v
whole-step measurement and verification
        |
        v
hierarchical execution policy
```

The execution policy is stored separately and keyed by the exact kernel signature so that a policy
cannot silently select a topology for a different kernel table.

## 4. Implemented mechanisms

### 4.1 Kernel/result reuse

Stage 2 can explicitly reuse an already measured Level-1 result. In reuse mode `autotune.py` is not
launched again. This permits Level 2 to operate on a completed Level-1 measurement without paying
for another kernel search.

### 4.2 Top-K preservation

For each exact P/I/W workload, multiple measured Level-1 candidates are retained rather than only
the single winner. Stage 2 can therefore test interactions between kernel choice and execution policy.

### 4.3 Execution search

The stage evaluates the small topology space of `grid.z`/Graph/recompute and then tests selected
kernel alternatives using full-step measurements. Coordinate passes and pairwise rescue allow a
candidate to survive only when it improves the complete execution rather than an isolated kernel.

### 4.4 Shape/depth-specific policy

The policy is not a global constant. Different depths and shapes can select different execution
settings because the cost of layer launches and grouped execution changes with topology.

### 4.5 Failure memory and exact signatures

Failed execution candidates are remembered, while successful measurements are associated with the
exact execution/kernel fingerprint. This prevents repeated compilation of known-bad configurations
and avoids applying a stale topology to a different measured kernel table.

## 5. Verification

The Stage 2.1 namespace/integration verification passed:

```text
Stage 2.1 hierarchical policy + locked-v12 namespace test: PASS
```

The locked-v12 source files were verified against their reference hashes in the Stage 2.1 package.
The package contains no separate `ipc_tilelang_adaptive` tree, so the reference engine is consolidated
inside the main `ipc_tilelang` namespace.

## 6. Execution measurements

A representative Stage 2 measurement on the target GPU showed a clear CUDA Graph benefit in the
launch-sensitive execution benchmark:

```text
Depth 4
  direct = 0.2246 ms
  graph  = 0.0612 ms
  ratio  = 3.671x

Depth 6
  direct = 0.3777 ms
  graph  = 0.0831 ms
  ratio  = 4.544x
```

These numbers are benchmark-level measurements and must not be arithmetically multiplied with
independent Level-1 kernel speedups.

Later full-fidelity Stage 2 runs continued to select CUDA Graph as the useful execution mode while
showing that kernel swaps do not automatically improve whole-step latency. This is expected: a kernel
that is faster in isolation can lose its advantage when its interaction with grouping, launch count,
recompute and graph capture is included.

## 7. Acceptance decision

Stage 2.1 is accepted as complete because all intended functions of the stage are present:

- Level 1 remains locked and unchanged as the reference engine.
- Level 2 consumes measured Level-1 top-K candidates.
- Execution parameters are searched at whole-step level.
- `grid.z` and recomputation are policy-controlled.
- CUDA Graph is benchmarked as an execution topology.
- Policies are tied to exact kernel fingerprints.
- The implementation passes the Stage 2.1 integration test.

The stage therefore reaches its intended endpoint: it converts independent optimized kernels into a
measured hierarchical execution policy without turning the project into an ever-growing autotuner.

## 8. What is deliberately not claimed

The Stage 2.1 finite adaptive search does not prove a mathematical global optimum.
Likewise, the separate Graph speedup coefficients cannot be multiplied with Stage-1 kernel speedups
because they measure different execution layers.

The acceptance criterion is an empirically measured, correct and reusable execution policy for the
target RTX 3060, not an assertion of theoretical optimality.

## 9. Current position in the optimization roadmap

Stage 2.1 itself is closed.
The subsequent runtime work extends it rather than reopening the hierarchical search:

```text
Stage 1.19 v12        DONE / LOCKED
Stage 2.1             DONE
Stage 2.3 runtime     DONE
Stage 2.4 Graph       VERIFIED on real GPU
Stage 2.5 small-B     VERIFIED; shape-dependent policy
```

The measured Stage 2.4 steady-state result for `784 -> 64 -> 64 -> 10`, `B=128`, was:

```text
direct = 0.2579 ms/step
graph  = 0.0944 ms/step
speedup = 2.732x
```

## 10. Conclusion

Stage 2.1 successfully completes the second-level execution optimization layer around the locked
v12 kernel autotuner. Further performance work should target remaining unchecked implementation
areas—especially optimizer fusion, unnecessary gradient materialization, resource-aware kernel
acceptance and sustained end-to-end training—rather than redesigning the Stage 2.1 search itself.
