# iPC TileLang — Stage 1.11

## Scope

Stage 1.11 is a low-risk performance instrumentation/specialization stage on top of the validated Stage 1.10 code.
It does **not** change the iPC equations or the already-fixed non-square weight-update ABI.

Target:

- NVIDIA GeForce RTX 3060 Laptop GPU
- SM86
- 6 GiB VRAM
- TileLang 0.1.13
- Python 3.10+

## Changes

### 1. Thermal benchmark accounting

The graph/non-graph benchmark now counts the real number of iPC steps executed:

- `graph.replay()` represents `--steps` iPC steps;
- samples are counted as `batch * steps`;
- output reports `replays`, `ipc_steps`, and `samples/s` separately.

This removes the Stage-1.10 ambiguity where `--steps 4` was reported as one batch per replay.

### 2. Independent tuning tables

`src/ipc_tilelang/tuning.py` introduces an exact-shape tuning table keyed by:

`kind × M × K × B × activation × dtype`

The trainer accepts an optional `TuningTable` through `IPCConfig.tuning_table` and falls back to the Stage-1.10 `KernelConfig` when no exact entry exists.

Separate profiles are therefore possible for:

- prediction/error;
- inference update;
- weight update.

### 3. Shape-aware autotuning

`scripts/autotune.py` now:

- searches small SM86-friendly tiles including `BM=16/32`, `BN=32/64`;
- keeps the known Stage-1.10 baseline as the first A/B candidate;
- samples the whole candidate space instead of taking a fixed prefix;
- supports `--kind prediction|inference|weight|all`;
- supports the `mnist64` shape profile;
- writes one best exact-shape entry per kernel family;
- uses the correct inference ABI `W_lower[K,M]`;
- can optionally reject candidates when source inspection suspects local-memory use or an excessive parsed register count.

### 4. Source-level register/spill instrumentation

TileLang 0.1.13 does not expose a stable register-count API across builds. The autotuner therefore records a **best-effort** PTX-style register estimate and a `spill_suspected` flag when source inspection finds local-memory operations.

These fields are diagnostics, not a substitute for final `ptxas -v`/profiler evidence.

## Intentionally not implemented in Stage 1.11

The following remain later stages:

- `grid.z` layer-batched execution;
- L2-aware layer grouping;
- small-B outer-product crossover;
- fused optimizer epilogues;
- long-horizon numerical drift qualification.

The reason is to keep Stage 1.11 mechanically comparable with Stage 1.10 while changing the benchmark and per-shape tuning machinery first.

## Apply

From the root of an existing `Lich` checkout, run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\APPLY_STAGE_1_11.ps1
```

The script creates timestamped `.stage110.bak` backups for replaced files.

## Validation

Correctness smoke tests:

```powershell
.\scripts\run_cli.py smoke --M 64 --K 64 --B 128
.\scripts\run_cli.py smoke --M 10 --K 64 --B 128
```

Generate a Stage-1.11 MNIST/64-wide tuning table on the RTX 3060:

```powershell
.\scripts\run_cli.py autotune --profile mnist64 --kind all --batch 128 --max-configs 80 --out results\autotune_stage_1_11.json
```

If the local dispatcher does not expose `--batch` for `autotune`, use:

```powershell
.\scripts\run_cli.py autotune --profile mnist64 --kind all --B 128 --max-configs 80 --out results\autotune_stage_1_11.json
```

Run the sustained benchmark with the generated table:

```powershell
.\scripts\run_cli.py thermal-bench --seconds 120 --depth 3 --batch 128 --steps 4 --graph --tuning results\autotune_stage_1_11.json
```

## Acceptance

A Stage-1.11 candidate should be accepted only after:

1. both regular and non-square tail smoke tests pass;
2. warmed-up end-to-end iPC throughput is measured;
3. tuned results are compared against the Stage-1.10 baseline;
4. no intermittent illegal-access/race/correctness failures appear;
5. the sustained run stays below the project thermal ceiling.
