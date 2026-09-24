# iPC × TileLang — RTX 3060 (SM86) implementation

This project is the first implementation stage of an aggressively specialized incremental predictive coding (iPC) trainer for a single NVIDIA RTX 3060 Laptop GPU.

## Scope of stage 1

1. Reproduce the iPC update equations from Salvatori et al. (ICLR 2024).
2. Provide a pure PyTorch reference implementation for numerical validation.
3. Provide TileLang CUDA kernels specialized for `sm_86`:
   - prediction + error with fused activation;
   - inference update with fused `W^T @ E` + derivative + state update;
   - weight update with `E @ A^T` and optional activation recomputation.
4. Provide a conservative Windows thermal/power guard based on `nvidia-smi`.
5. Provide benchmark/autotuning scaffolding and paper experiment profiles.

## Important

The paper's exact high-level iPC equations and several experiment settings are known, but some implementation details of the authors' reference classifier are not specified in the paper/supplement. Those details are exposed as explicit configuration instead of silently guessed.

## Windows environment

This project targets TileLang 0.1.13 on Windows x86-64 and Python >= 3.10. TileLang 0.1.13 is the installed/current target for this stage. Its current Windows packaging can use pip-provided CUDA toolchain packages when building from source; prebuilt PyPI installation is the preferred first attempt.

Recommended baseline (from the checkout):

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\install_windows.ps1
```

If `scripts` and `src/ipc_tilelang` are in different sibling folders, set `IPC_TILELANG_ROOT` (or use `scripts/run_cli.py`, which auto-discovers it).

Then verify:

```powershell
.\.venv\Scripts\python.exe .\scripts\system_report.py
```

## Numerical reference

The supervised PC/iPC convention in this repository is:

- `x[0]` = fixed one-hot target/output;
- `x[L]` = fixed input vector;
- `W[l]` maps `f(x[l+1]) -> x[l]`;
- `epsilon[l] = x[l] - W[l] @ f(x[l+1])`;
- hidden state update:
  `x[l] += gamma * (-epsilon[l] + f'(x[l]) * W[l-1]^T @ epsilon[l-1])`;
- weight update:
  `W[l] += alpha * epsilon[l] @ f(x[l+1])^T`.

The weights therefore use the same sign as Eq. (7) of the paper after writing the tensors in conventional matrix form.

## First paper validation target

The paper's discriminative efficiency supplement specifies fully-connected classifiers with 64 hidden neurons and depths `L in {3,4,6}`, with `alpha=1e-4` and `gamma=0.5`. This repository exposes exactly that profile.

For an end-to-end sanity check, the paper also reports an MLP with two hidden layers of 64 neurons on MNIST and reports iPC accuracy `98.54 +/- 0.86%`. Reproducing that exact number requires matching the paper's omitted implementation details, initialization, training schedule and data handling, so the repository does not claim exact reproduction until those are pinned down.

## Performance strategy already encoded

- hard-pin CUDA target to `sm_86`;
- FP16 storage / FP32 accumulation as the default fast path;
- tile sizes exposed to autotuning;
- 1/2/3-stage pipeline search;
- optional threadblock swizzle;
- activation fusion;
- derivative fusion;
- `A=f(X)` save-vs-recompute switch;
- separate weight-update kernel family;
- small-batch outer-product path is reserved as the next specialization;
- no per-step tensor allocation in the intended trainer;
- thermal guard pauses or aborts before the user's configured safety ceiling.

## Run

Reference only:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_cli.py reference --depth 3 --steps 10
```

TileLang correctness smoke test:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_cli.py smoke --M 64 --K 64 --B 128
```

Autotune a single prediction/error kernel:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_cli.py autotune --M 64 --K 64 --B 128
```

Train the MNIST paper profile:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_cli.py train-mnist --depth 2 --hidden 64 --batch 128 --epochs 5
```

Run the sustained thermal benchmark:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_cli.py thermal-bench --seconds 120 --batch 128 --steps 4 --graph
```

Run the 64-wide efficiency sweep:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_cli.py efficiency --depths 3 4 6 --hidden 64 --batch 128
```

The full benchmark intentionally has a temperature ceiling and hysteresis. The program never assumes that a nominal 120 W laptop GPU can sustain a given clock/power indefinitely; it observes actual telemetry and throttles the benchmark workload itself.

## Thermal policy

The default sustained-training ceiling is 76 C with 4 C hysteresis. The code does not change the driver's 120 W power limit or force clocks. NVIDIA's throttle-reason bitmask is sampled at a low rate; `SW_POWER_CAP` is reported as an informational normal-at-limit condition, while explicit thermal slowdown and hot hardware slowdown trigger a pause. This keeps `nvidia-smi` out of the per-batch hot path.

Before a long training run on the laptop, calibrate the actual cooling system with:

```powershell
.\.venv\Scripts\python.exe .\scripts\thermal_bench.py --seconds 120 --batch 128 --steps 4 --graph
```

A sustained run is intentionally required before accepting a kernel configuration. A short benchmark can pick a configuration that looks faster before the laptop reaches its steady-state temperature.

## Maximum-performance checklist

### Stage 1.2 current status
The first real SM86 compile report exposed TileLang 0.1.13 pipeline-planning constraints. The prediction and recompute-activation paths now keep raw and transformed shared tiles separate; this avoids multiple writes to one pipelined buffer at different pipeline stages. This costs one additional shared tile and is intentional until a faster single-write activation loader is benchmarked. Tail handling uses native `T.copy` legalization rather than a custom `T.Parallel` copy helper. The inference path preserves `W_lower[K,M]` and transposes that shared tile inside `T.gemm`.


Status legend: `[x]` implemented through the current stage; `[~]` scaffolded or requires real-GPU validation; `[ ]` next optimization.

### Correctness / benchmark parity
- [x] FP32/PyTorch mathematical reference.
- [x] iPC step ordering matched to the paper profile.
- [ ] Reproduce the paper discriminative efficiency sweep: 64 hidden units, L in {3,4,6}, alpha=1e-4, gamma=0.5.
- [ ] Reproduce 5-seed accuracy protocol where source details are sufficiently specified.

### Kernel / memory
- [x] SM86 specialization.
- [x] FP16 storage / FP32 accumulation baseline.
- [x] Fuse f(x) into prediction GEMM.
- [x] Fuse f'(x) into inference update.
- [x] Do not materialize mu, backward signal, or f'(x).
- [x] Save-A vs recompute-A switch.
- [x] Preallocated persistent X/E/A/W buffers.
- [x] Correct W^T handling in inference kernel (`W_lower[K,M]`, transpose A).
- [ ] Small-B rank-1/outer-product weight-update kernel.
- [ ] Layer-grouped/grid.z fused kernels for repeated shapes.
- [ ] Per-shape kernel specializations and padding strategy.

### Tensor Core / scheduling search
- [~] BM/BN/BK autotune.
- [~] threads autotune.
- [~] 1/2/3-stage pipeline search.
- [~] shared-memory swizzle search.
- [~] CTA rasterization/swizzle search.
- [ ] Register-pressure-aware pruning and spill detection.
- [ ] Separate autotune tables for prediction/inference/weight kernels.
- [ ] End-to-end rather than single-kernel objective.
- [ ] L2-locality-aware layer grouping.

### Training/runtime
- [x] No per-step allocation in the intended hot path.
- [x] CUDA Graph support for fixed batch/step shapes.
- [ ] Fused AdamW epilogue without global gradient tensor.
- [ ] Dynamic store-vs-recompute cost model.
- [ ] Dynamic register/shared/recompute placement policy.
- [ ] Pinned host input + async H2D batch pipeline.
- [ ] Double/triple-buffered prefetch where it actually overlaps.

### RTX 3060 thermal stability
- [x] Telemetry rate-limited off the hot path.
- [x] Temperature ceiling + hysteresis.
- [x] Throttle-reason reporting.
- [~] Sustained benchmark required before accepting a configuration.
- [ ] Build a per-machine sustainable-clock/power profile.
- [ ] Reject configurations that win only in a short cold-start benchmark.
- [ ] Tune workload below the thermal ceiling rather than trying to force clocks.

### Required evidence before declaring an optimization "better"
- [ ] Correctness against FP32 reference.
- [ ] Warmed-up latency.
- [ ] End-to-end iPC iterations/s.
- [ ] Samples/s.
- [ ] Registers/thread and spill report.
- [ ] Shared-memory bytes/CTA and active CTAs/SM.
- [ ] VRAM usage.
- [ ] Sustained temperature and clock after >= 2 minutes.
- [ ] No thermal slowdown in the accepted run.

## CUDA Graph mode

For fixed batch shape and fixed iPC step count, `--graph` captures the hot iPC loop after JIT compilation and replays it for later batches. The model reuses its state/error allocations, so graph tensor addresses stay stable.

## Stage 1.5
- Fixed Python -> cmd.exe quoting around vcvars64.bat.
- Added exact-toolset attempt with safe fallback.
- Force selected cl.exe directory to the front of PATH.
- Expanded toolchain diagnostics.


## Stage 1.8 toolchain bootstrap
`run_cli.py` and the kernel builders initialize native MSVC before TileLang JIT decorators are constructed. This is required because each command runs in a fresh Windows process.
