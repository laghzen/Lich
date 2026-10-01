from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import torch

from _bootstrap import bootstrap
bootstrap()

from ipc_tilelang.trainer import IPCConfig, TileLangIPC
from ipc_tilelang.tilelang_kernels import KernelConfig
from ipc_tilelang.layer_batching import find_internal_square_groups
from ipc_tilelang.hierarchical_policy import kernel_signature
from ipc_tilelang.adaptive import AdaptiveFiniteTuner


STAGE_VERSION = "2.3"
EXECUTION_CACHE_VERSION = 4
KERNEL_RESULT_SIDECAR_VERSION = 2
SEARCH_SPACE_VERSION = "sm86-v3-eval-order"
KernelKey = tuple[str, int, int, int, str, str]


def _dtype_name(dtype: Any) -> str:
    return str(dtype).split(".")[-1]


def _kernel_key_from_row(row: dict[str, Any]) -> KernelKey:
    shape = row.get("shape") or {}
    return (
        str(row["kind"]),
        int(shape["M"]),
        int(shape["K"]),
        int(shape["B"]),
        str(row.get("activation", "relu")),
        str(row.get("dtype", "float16")),
    )


def _kernel_key_label(key: KernelKey) -> str:
    kind, m, k, b, activation, dtype = key
    return f"{kind}:{m}x{k}x{b}:{activation}:{dtype}"


class ResultTuningTable:
    """Exact-key kernel table. Level 1 remains the byte-for-byte locked v12 engine."""

    def __init__(self, entries: list[dict[str, Any]]):
        self.entries = list(entries)
        self._map: dict[KernelKey, KernelConfig] = {}
        for row in self.entries:
            if not isinstance(row, dict) or not isinstance(row.get("config"), dict):
                continue
            try:
                self._map[_kernel_key_from_row(row)] = KernelConfig(**dict(row["config"]))
            except Exception:
                continue

    def lookup(self, *, kind: str, M: int, K: int, B: int, activation: str, dtype: str):
        return self._map.get((kind, int(M), int(K), int(B), activation, dtype))

    def kernel_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for key, cfg in sorted(self._map.items()):
            kind, M, K, B, activation, dtype = key
            rows.append({
                "kind": kind,
                "shape": {"M": M, "K": K, "B": B},
                "activation": activation,
                "dtype": dtype,
                "config": asdict(cfg),
            })
        return rows

    def copy_with_override(self, key: KernelKey, cfg: KernelConfig) -> "ResultTuningTable":
        return self.copy_with_overrides({key: cfg})

    def copy_with_overrides(self, overrides: dict[KernelKey, KernelConfig]) -> "ResultTuningTable":
        rows = self.kernel_rows()
        seen: set[KernelKey] = set()
        for row in rows:
            key = _kernel_key_from_row(row)
            if key in overrides:
                row["config"] = asdict(overrides[key])
                seen.add(key)
        missing = set(overrides) - seen
        if missing:
            raise KeyError("Kernel keys not present: " + ", ".join(_kernel_key_label(k) for k in sorted(missing)))
        return ResultTuningTable(rows)


def load_kernel_result(path: Path) -> tuple[ResultTuningTable, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"No Stage-1 kernel entries in {path}")
    return ResultTuningTable(entries), data


def _tilelang_version() -> str:
    try:
        import tilelang
        return str(getattr(tilelang, "__version__", "unknown"))
    except Exception:
        return "unknown"


def _hash_files(paths: Iterable[Path]) -> str:
    h = hashlib.sha256()
    found = 0
    for path in sorted({p.resolve() for p in paths}):
        if path.exists():
            found += 1
            h.update(str(path).encode("utf-8"))
            h.update(path.read_bytes())
    if not found:
        return "unknown"
    return h.hexdigest()[:20]


def source_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    return _hash_files([root / "src" / "ipc_tilelang" / "tilelang_kernels.py"])


def runtime_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    return _hash_files([
        root / "src" / "ipc_tilelang" / "trainer.py",
        root / "src" / "ipc_tilelang" / "layer_batching.py",
        root / "src" / "ipc_tilelang" / "cuda_graph.py",
        root / "src" / "ipc_tilelang" / "hierarchical_policy.py",
    ])


def _device_fingerprint(device: torch.device) -> str:
    props = torch.cuda.get_device_properties(device)
    payload = {
        "name": torch.cuda.get_device_name(device),
        "cc": list(torch.cuda.get_device_capability(device)),
        "total_memory": int(props.total_memory),
        "torch": torch.__version__,
        "tilelang": _tilelang_version(),
        "kernel_source": source_fingerprint(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:20]


def kernel_result_compatible(path: Path, *, device: torch.device) -> tuple[bool, str]:
    try:
        tuning, data = load_kernel_result(path)
    except Exception as exc:
        return False, f"invalid:{type(exc).__name__}"
    if not tuning.kernel_rows():
        return False, "no_kernel_rows"
    if str(data.get("stage")) != "1.19":
        return False, f"stage={data.get('stage')}"
    if str(data.get("device")) != torch.cuda.get_device_name(device):
        return False, "device_mismatch"
    if list(data.get("capability", [])) != list(torch.cuda.get_device_capability(device)):
        return False, "capability_mismatch"
    if str(data.get("tilelang_version", "unknown")) != _tilelang_version():
        return False, "tilelang_mismatch"
    engines = {
        str((row or {}).get("search_engine"))
        for row in (data.get("results") or [])
        if isinstance(row, dict)
    }
    if AdaptiveFiniteTuner.ENGINE not in engines:
        return False, "not_v12_engine"
    return True, "ok"


def write_kernel_sidecar(path: Path, *, device: torch.device) -> None:
    sidecar = path.with_suffix(path.suffix + ".stage2meta.json")
    payload = {
        "version": KERNEL_RESULT_SIDECAR_VERSION,
        "created_at": time.time(),
        "stage2": STAGE_VERSION,
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "tilelang_version": _tilelang_version(),
        "kernel_source_fingerprint": source_fingerprint(),
        "runtime_fingerprint": runtime_fingerprint(),
        "v12_engine": AdaptiveFiniteTuner.ENGINE,
    }
    sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sidecar_compatible(path: Path, *, device: torch.device) -> tuple[bool, str]:
    sidecar = path.with_suffix(path.suffix + ".stage2meta.json")
    if not sidecar.exists():
        return False, "sidecar_missing"
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"sidecar_invalid:{type(exc).__name__}"
    expected = {
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "tilelang_version": _tilelang_version(),
        "kernel_source_fingerprint": source_fingerprint(),
        "runtime_fingerprint": runtime_fingerprint(),
        "v12_engine": AdaptiveFiniteTuner.ENGINE,
    }
    for k, v in expected.items():
        if data.get(k) != v:
            return False, f"sidecar_{k}_mismatch"
    return True, "ok"


def run_kernel_stage(args: argparse.Namespace, kernel_out: Path) -> None:
    script = Path(__file__).resolve().parent / "autotune.py"
    cmd = [
        sys.executable, str(script),
        "--kind", "all",
        "--profile", "mnist64",
        "--hidden", str(args.hidden),
        "--B", str(args.batch),
        "--activation", args.activation,
        "--seed-evals", str(args.seed_evals),
        "--batch-size", str(args.kernel_batch_size),
        "--topk", str(args.kernel_topk),
        "--probe-warmup", str(args.probe_warmup),
        "--probe-reps", str(args.probe_reps),
        "--verify-warmup", str(args.verify_warmup),
        "--verify-reps", str(args.verify_reps),
        "--temperature-max", str(args.temperature_max),
        "--out", str(kernel_out),
        "--cache", str(args.kernel_cache),
    ]
    if args.warm_start_cache:
        cmd.append("--warm-start-cache")
    if args.reject_spill:
        cmd.append("--reject-spill")
    print("\n=== LEVEL 1: P/I/W kernel autotuning ===")
    print("$ " + subprocess.list2cmdline(cmd))
    subprocess.run(cmd, check=True)


def _best_measurement_per_config(observations: list[dict[str, Any]]) -> list[tuple[float, int, KernelConfig]]:
    by_cfg: dict[KernelConfig, tuple[float, int]] = {}
    for row in observations:
        if not isinstance(row, dict) or not row.get("successful") or row.get("latency_ms") is None:
            continue
        try:
            cfg = KernelConfig(**dict(row["config"]))
            ms = float(row["latency_ms"])
            fidelity = int(row.get("fidelity", 0))
        except Exception:
            continue
        current = by_cfg.get(cfg)
        # Prefer the highest-fidelity fresh observation; within a fidelity prefer the
        # best measured latency. A probe can therefore never displace a verified run.
        if current is None or fidelity > current[1] or (fidelity == current[1] and ms < current[0]):
            by_cfg[cfg] = (ms, fidelity)
    # `KernelConfig` is a frozen dataclass but intentionally has no ordering.
    # Never let equal latency/fidelity values fall through to Python trying to
    # compare KernelConfig objects. Use a deterministic structural tie-break.
    def _cfg_sort_key(cfg: KernelConfig) -> tuple:
        return (
            int(cfg.block_m), int(cfg.block_n), int(cfg.block_k),
            int(cfg.threads), int(cfg.num_stages), int(bool(cfg.swizzle)),
            int(cfg.swizzle_panel), int(bool(cfg.shared_swizzle)),
        )

    return sorted(
        ((ms, fidelity, cfg) for cfg, (ms, fidelity) in by_cfg.items()),
        key=lambda item: (float(item[0]), -int(item[1]), _cfg_sort_key(item[2])),
    )


def collect_topk_candidates_from_run(
    tuning: ResultTuningTable,
    kernel_data: dict[str, Any],
    *,
    topk: int,
) -> dict[KernelKey, list[KernelConfig]]:
    """Build exact top-K from the current run's fresh GPU observations.

    This is deliberately independent of every persistent measurement cache. The Level-1
    wrapper records the observations made by the locked v12 process and hands them to Level 2.
    """
    by_key: dict[KernelKey, list[dict[str, Any]]] = {key: [] for key in tuning._map}
    for row in kernel_data.get("results", []):
        if not isinstance(row, dict):
            continue
        try:
            key = _kernel_key_from_row(row)
        except Exception:
            continue
        for obs in row.get("fresh_observations", []):
            if isinstance(obs, dict):
                by_key.setdefault(key, []).append(obs)

    result: dict[KernelKey, list[KernelConfig]] = {}
    for key, top1 in tuning._map.items():
        ranked = _best_measurement_per_config(by_key.get(key, []))
        unique = [cfg for _, _, cfg in ranked[: max(1, topk)]]
        if top1 not in unique:
            unique.insert(0, top1)
        result[key] = unique[: max(1, topk)]
    return result


def make_batch(batch: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(12345)
    x = torch.randn((784, batch), device=device, dtype=dtype, generator=g) * 0.1
    labels = torch.arange(batch, device=device, dtype=torch.long) % 10
    y = torch.zeros((10, batch), device=device, dtype=dtype)
    y.scatter_(0, labels.unsqueeze(0), 1)
    return x, y


def topology_signature(dims: tuple[int, ...], cap: int):
    if cap < 2:
        return ()
    return tuple((g.start, g.end, g.count) for g in find_internal_square_groups(dims, max_group_count=cap))


def unique_caps(dims: tuple[int, ...], caps: list[int]) -> list[int]:
    seen = set()
    out = []
    for cap in caps:
        sig = topology_signature(dims, cap)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(int(cap))
    return out


def make_model(
    *,
    depth: int,
    batch: int,
    cap: int,
    recompute_activation: bool,
    tuning: ResultTuningTable,
    device: torch.device,
) -> TileLangIPC:
    dims = (10,) + (64,) * depth + (784,)
    return TileLangIPC(
        IPCConfig(
            dims=dims,
            alpha=1e-4,
            gamma=0.5,
            activation="relu",
            dtype=torch.float16,
            kernel=KernelConfig(64, 128, 16, 128, 2, True, 8, False),
            recompute_activation=bool(recompute_activation),
            tuning_table=tuning,
            use_grid_z=cap >= 2,
            grid_z_max_layers=max(0, int(cap)),
            grid_z_auto=False,
            hierarchical_policy_auto=False,
        ),
        device=device,
        seed=7,
    )


def run_direct(model: TileLangIPC, steps: int, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        for _ in range(steps):
            model.step_initialized(collect_metrics=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        for _ in range(steps):
            model.step_initialized(collect_metrics=False)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / (repeats * steps)


def run_graph(model: TileLangIPC, steps: int, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        for _ in range(steps):
            model.step_initialized(collect_metrics=False)
    torch.cuda.synchronize()
    runner = model.capture_graph(steps)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        runner.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / (repeats * steps)


def measure_execution_modes(
    *,
    depth: int,
    batch: int,
    cap: int,
    recompute_activation: bool,
    modes: set[bool],
    steps: int,
    warmup: int,
    repeats: int,
    tuning: ResultTuningTable,
    device: torch.device,
) -> tuple[dict[bool, float], tuple[tuple[int, int, int], ...], dict[bool, str]]:
    if not modes:
        return {}, tuple(), {}
    dims = (10,) + (64,) * depth + (784,)
    x, y = make_batch(batch, device, torch.float16)
    model = make_model(
        depth=depth, batch=batch, cap=cap, recompute_activation=recompute_activation,
        tuning=tuning, device=device,
    )
    model.initialize_batch(x, y)
    out: dict[bool, float] = {}
    errors: dict[bool, str] = {}
    try:
        # One model/compiled kernel set services both direct and graph A/B.
        if False in modes:
            try:
                out[False] = run_direct(model, steps, warmup, repeats)
            except Exception as exc:
                errors[False] = f"{type(exc).__name__}: {exc}"[:4000]
        if True in modes:
            try:
                out[True] = run_graph(model, steps, warmup, repeats)
            except Exception as exc:
                errors[True] = f"{type(exc).__name__}: {exc}"[:4000]
        groups = tuple((g.start, g.end, g.count) for g in model._layer_groups)
    finally:
        del model, x, y
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return out, groups, errors


def _load_execution_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": EXECUTION_CACHE_VERSION, "entries": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("entries"), dict):
            if int(data.get("version", 0)) != EXECUTION_CACHE_VERSION:
                return {"version": EXECUTION_CACHE_VERSION, "entries": {}}
            return data
    except Exception:
        pass
    return {"version": EXECUTION_CACHE_VERSION, "entries": {}}


def _save_execution_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _execution_cache_key(
    *,
    device: torch.device,
    kernel_sig: str,
    dims: tuple[int, ...],
    batch: int,
    cap: int,
    recompute_activation: bool,
    use_graph: bool,
    steps: int,
    warmup: int,
    repeats: int,
    activation: str,
    dtype: str,
    fidelity_tag: str,
) -> str:
    payload = {
        "executor_version": STAGE_VERSION,
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "tilelang": _tilelang_version(),
        "kernel_source": source_fingerprint(),
        "runtime_source": runtime_fingerprint(),
        "kernel_signature": kernel_sig,
        "dims": list(dims),
        "batch": int(batch),
        "grid_z_max_layers": int(cap),
        "recompute_activation": bool(recompute_activation),
        "use_cuda_graph": bool(use_graph),
        "steps": int(steps),
        "warmup": int(warmup),
        "repeats": int(repeats),
        "activation": activation,
        "dtype": dtype,
        "fidelity_tag": fidelity_tag,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _trial_row(*, depth: int, batch: int, cap: int, graph: bool, recompute: bool, ms: float, groups, source: str, fidelity_tag: str) -> dict[str, Any]:
    return {
        "depth": int(depth),
        "grid_z_max_layers": int(cap),
        "recompute_activation": bool(recompute),
        "use_cuda_graph": bool(graph),
        "step_ms": float(ms),
        "samples_per_s": float(1000.0 * batch / ms),
        "groups": [list(g) for g in groups],
        "status": "ok",
        "source": source,
        "fidelity": fidelity_tag,
    }


def _policy_key(cap: int, recompute: bool) -> tuple[int, bool]:
    return int(cap), bool(recompute)


def evaluate_policy_matrix(
    *,
    depth: int,
    batch: int,
    caps: list[int],
    include_graph: bool,
    recompute_modes: list[bool],
    steps: int,
    warmup: int,
    repeats: int,
    fidelity_tag: str,
    tuning: ResultTuningTable,
    device: torch.device,
    kernel_sig: str,
    execution_cache: dict[str, Any],
    cache_path: Path,
    use_execution_cache: bool,
) -> dict[str, Any]:
    dims = (10,) + (64,) * depth + (784,)
    caps_u = unique_caps(dims, caps)
    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None

    for cap, recompute in itertools.product(caps_u, recompute_modes):
        wanted = [False, True] if include_graph else [False]
        mode_results: dict[bool, float] = {}
        mode_groups: dict[bool, tuple[tuple[int, int, int], ...]] = {}
        missing: set[bool] = set()

        for use_graph in wanted:
            key = _execution_cache_key(
                device=device, kernel_sig=kernel_sig, dims=dims, batch=batch,
                cap=cap, recompute_activation=recompute, use_graph=use_graph,
                steps=steps, warmup=warmup, repeats=repeats,
                activation="relu", dtype="float16", fidelity_tag=fidelity_tag,
            )
            entry = execution_cache.setdefault("entries", {}).get(key) if use_execution_cache else None
            if isinstance(entry, dict) and entry.get("status") == "ok":
                try:
                    mode_results[use_graph] = float(entry["step_ms"])
                    mode_groups[use_graph] = tuple(tuple(int(v) for v in g) for g in entry.get("groups", []))
                except Exception:
                    missing.add(use_graph)
            elif isinstance(entry, dict) and entry.get("status") == "failed":
                print(f"cached failure depth={depth} cap={cap} recompute={int(recompute)} graph={int(use_graph)}: {entry.get('error', 'unknown')}")
            else:
                missing.add(use_graph)

        if missing:
            measured, groups, errors = measure_execution_modes(
                depth=depth, batch=batch, cap=cap, recompute_activation=recompute,
                modes=missing, steps=steps, warmup=warmup, repeats=repeats,
                tuning=tuning, device=device,
            )
            for use_graph in sorted(missing):
                key = _execution_cache_key(
                    device=device, kernel_sig=kernel_sig, dims=dims, batch=batch,
                    cap=cap, recompute_activation=recompute, use_graph=use_graph,
                    steps=steps, warmup=warmup, repeats=repeats,
                    activation="relu", dtype="float16", fidelity_tag=fidelity_tag,
                )
                if use_graph in measured:
                    execution_cache.setdefault("entries", {})[key] = {
                        "status": "ok", "step_ms": float(measured[use_graph]),
                        "groups": [list(g) for g in groups], "created_at": time.time(),
                    }
                    mode_results[use_graph] = float(measured[use_graph])
                    mode_groups[use_graph] = groups
                else:
                    execution_cache.setdefault("entries", {})[key] = {
                        "status": "failed", "error": errors.get(use_graph, "unknown"), "created_at": time.time(),
                    }
            _save_execution_cache(cache_path, execution_cache)

        for use_graph in wanted:
            if use_graph not in mode_results:
                rows.append({
                    "depth": depth,
                    "grid_z_max_layers": int(cap),
                    "recompute_activation": bool(recompute),
                    "use_cuda_graph": bool(use_graph),
                    "step_ms": None,
                    "samples_per_s": None,
                    "groups": [list(g) for g in topology_signature(dims, cap)],
                    "status": "failed",
                    "source": "gpu-failed",
                    "fidelity": fidelity_tag,
                })
                continue
            source = "cache" if use_execution_cache and not missing else "gpu"
            row = _trial_row(
                depth=depth, batch=batch, cap=cap, graph=use_graph,
                recompute=recompute, ms=mode_results[use_graph],
                groups=mode_groups.get(use_graph, tuple()), source=source,
                fidelity_tag=fidelity_tag,
            )
            rows.append(row)
            if best_row is None or float(row["step_ms"]) < float(best_row["step_ms"]):
                best_row = row
            print(
                f"depth={depth} cap={cap} recompute={int(recompute)} graph={int(use_graph)} "
                f"step={row['step_ms']:.4f} ms source={source} groups={tuple(tuple(g) for g in row['groups'])}"
            )

    if best_row is None:
        raise RuntimeError(f"No successful execution configuration for depth={depth}")
    return {
        "dims": list(dims),
        "batch": int(batch),
        "activation": "relu",
        "dtype": "float16",
        "candidates": rows,
        "best": dict(best_row),
        "kernel_signature": kernel_sig,
        "fidelity": fidelity_tag,
    }


def _best_trial(profile: dict[str, Any]) -> dict[str, Any]:
    return dict(profile["best"])


def _variant_from_single_swap(baseline: ResultTuningTable, key: KernelKey, cfg: KernelConfig) -> ResultTuningTable:
    return baseline.copy_with_override(key, cfg)


def _screen_variant(
    *,
    depth: int,
    batch: int,
    tuning: ResultTuningTable,
    variant_name: str,
    exec_policy: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    execution_cache: dict[str, Any],
    execution_cache_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cap = int(exec_policy["grid_z_max_layers"])
    recompute = bool(exec_policy["recompute_activation"])
    recompute_modes = [True, False] if args.search_recompute else [recompute]
    profile = evaluate_policy_matrix(
        depth=depth, batch=batch, caps=[cap], include_graph=args.include_graph,
        recompute_modes=recompute_modes, steps=args.level2_screen_steps,
        warmup=args.level2_screen_warmup, repeats=args.level2_screen_repeats,
        fidelity_tag="screen", tuning=tuning, device=device,
        kernel_sig=kernel_signature(tuning.kernel_rows()), execution_cache=execution_cache,
        cache_path=execution_cache_path, use_execution_cache=args.use_execution_cache,
    )
    trials = []
    for row in profile["candidates"]:
        rr = dict(row)
        rr["kernel_variant"] = variant_name
        rr["kernel_signature"] = profile["kernel_signature"]
        trials.append(rr)
    return profile, trials


def _select_single_swap_candidates(
    *,
    baseline: ResultTuningTable,
    topk_candidates: dict[KernelKey, list[KernelConfig]],
    depth: int,
    batch: int,
    execution_best: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    execution_cache: dict[str, Any],
    execution_cache_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Screen every fresh Level-1 top-K alternative under the current best runtime policy."""
    states: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    for key in sorted(topk_candidates):
        base_cfg = baseline.lookup(
            kind=key[0], M=key[1], K=key[2], B=key[3], activation=key[4], dtype=key[5]
        )
        for rank, cfg in enumerate(topk_candidates[key], start=1):
            if cfg == base_cfg:
                continue
            variant = _variant_from_single_swap(baseline, key, cfg)
            name = f"swap:{_kernel_key_label(key)}:rank{rank}"
            print(f"\nLEVEL 2 kernel interaction screen depth={depth} {name}")
            profile, local_trials = _screen_variant(
                depth=depth, batch=batch, tuning=variant, variant_name=name,
                exec_policy=execution_best, args=args, device=device,
                execution_cache=execution_cache, execution_cache_path=execution_cache_path,
            )
            best = _best_trial(profile)
            state = {
                "name": name,
                "key": key,
                "rank": rank,
                "config": cfg,
                "table": variant,
                "screen_best": best,
                "screen_step_ms": float(best["step_ms"]),
                "kernel_signature": profile["kernel_signature"],
            }
            states.append(state)
            trials.extend(local_trials)
    return states, trials


def _top_distinct_key_states(states: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    used_keys: set[KernelKey] = set()
    for state in sorted(states, key=lambda s: (s["screen_step_ms"], s["name"])):
        key = state["key"]
        if key in used_keys:
            continue
        used_keys.add(key)
        chosen.append(state)
        if len(chosen) >= max(1, limit):
            break
    return chosen


def _pairwise_rescue(
    *,
    baseline: ResultTuningTable,
    states: list[dict[str, Any]],
    depth: int,
    batch: int,
    execution_best: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    execution_cache: dict[str, Any],
    execution_cache_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chosen = _top_distinct_key_states(states, args.level2_pairwise_width)
    pair_states: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    for a, b in itertools.combinations(chosen, 2):
        if a["key"] == b["key"]:
            continue
        table = baseline.copy_with_overrides({a["key"]: a["config"], b["key"]: b["config"]})
        name = f"pair:{a['name']}+{b['name']}"
        print(f"\nLEVEL 2 pairwise rescue depth={depth} {name}")
        profile, local_trials = _screen_variant(
            depth=depth, batch=batch, tuning=table, variant_name=name,
            exec_policy=execution_best, args=args, device=device,
            execution_cache=execution_cache, execution_cache_path=execution_cache_path,
        )
        best = _best_trial(profile)
        pair_states.append({
            "name": name,
            "keys": [a["key"], b["key"]],
            "configs": [a["config"], b["config"]],
            "table": table,
            "screen_best": best,
            "screen_step_ms": float(best["step_ms"]),
            "kernel_signature": profile["kernel_signature"],
        })
        trials.extend(local_trials)
    return pair_states, trials


def tune_depth_full(
    *,
    depth: int,
    batch: int,
    caps: list[int],
    include_graph: bool,
    recompute_modes: list[bool],
    steps: int,
    warmup: int,
    repeats: int,
    baseline: ResultTuningTable,
    topk_candidates: dict[KernelKey, list[KernelConfig]],
    args: argparse.Namespace,
    device: torch.device,
    execution_cache: dict[str, Any],
    execution_cache_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], ResultTuningTable]:
    """Full hierarchical Level 2:

    1. Exhaustive finite execution matrix for the baseline v12 table.
    2. Fresh whole-step screen of EVERY Level-1 top-K alternative.
    3. Coordinate adoption of measured winners.
    4. Pairwise rescue for interactions between strong alternatives.
    5. Full execution-matrix refinement of the resulting joint kernel tables.
    6. One final coordinate pass around the refined table, then final exhaustive topology.
    """
    all_trials: list[dict[str, Any]] = []
    baseline_sig = kernel_signature(baseline.kernel_rows())

    baseline_exec = evaluate_policy_matrix(
        depth=depth, batch=batch, caps=caps, include_graph=include_graph,
        recompute_modes=recompute_modes, steps=steps, warmup=warmup, repeats=repeats,
        fidelity_tag="full", tuning=baseline, device=device, kernel_sig=baseline_sig,
        execution_cache=execution_cache, cache_path=execution_cache_path,
        use_execution_cache=args.use_execution_cache,
    )
    for row in baseline_exec["candidates"]:
        rr = dict(row)
        rr["kernel_variant"] = "baseline"
        rr["kernel_signature"] = baseline_sig
        all_trials.append(rr)

    baseline_best = _best_trial(baseline_exec)
    print(
        f"LEVEL 2 baseline depth={depth}: cap={baseline_best['grid_z_max_layers']} "
        f"recompute={int(bool(baseline_best['recompute_activation']))} "
        f"graph={int(bool(baseline_best['use_cuda_graph']))} step={baseline_best['step_ms']:.4f} ms"
    )

    states, screen_trials = _select_single_swap_candidates(
        baseline=baseline, topk_candidates=topk_candidates, depth=depth, batch=batch,
        execution_best=baseline_best, args=args, device=device,
        execution_cache=execution_cache, execution_cache_path=execution_cache_path,
    )
    all_trials.extend(screen_trials)

    current_table = baseline
    current_exec = baseline_exec
    current_best = baseline_best
    adopted_names: list[str] = []

    # Coordinate pass: every key gets its full top-K fresh interaction screen. We adopt only
    # an actually measured whole-step improvement, never a model-only prediction.
    for key in sorted(topk_candidates):
        candidates_for_key = [s for s in states if s["key"] == key]
        candidates_for_key.sort(key=lambda s: (s["screen_step_ms"], s["name"]))
        if not candidates_for_key:
            continue
        best_state = candidates_for_key[0]
        if best_state["screen_step_ms"] < float(current_best["step_ms"]):
            # Screen first, then demand a full-fidelity execution-matrix confirmation before
            # changing the incumbent. A noisy low-fidelity screen can never regress the result.
            candidate_sig = kernel_signature(best_state["table"].kernel_rows())
            candidate_exec = evaluate_policy_matrix(
                depth=depth, batch=batch, caps=caps, include_graph=include_graph,
                recompute_modes=recompute_modes, steps=steps, warmup=warmup, repeats=repeats,
                fidelity_tag="full", tuning=best_state["table"], device=device, kernel_sig=candidate_sig,
                execution_cache=execution_cache, cache_path=execution_cache_path,
                use_execution_cache=args.use_execution_cache,
            )
            for row in candidate_exec["candidates"]:
                rr = dict(row)
                rr["kernel_variant"] = best_state["name"]
                rr["kernel_signature"] = candidate_sig
                all_trials.append(rr)
            candidate_best = _best_trial(candidate_exec)
            if float(candidate_best["step_ms"]) < float(current_best["step_ms"]):
                current_table = best_state["table"]
                current_exec = candidate_exec
                current_best = candidate_best
                adopted_names.append(best_state["name"])
                print(
                    f"JOINT INCUMBENT depth={depth}: {best_state['name']} "
                    f"cap={current_best['grid_z_max_layers']} recompute={int(bool(current_best['recompute_activation']))} "
                    f"graph={int(bool(current_best['use_cuda_graph']))} step={current_best['step_ms']:.4f} ms"
                )

    # Second coordinate pass captures kernel-kernel interactions induced by the updated incumbent.
    refreshed_states, refreshed_trials = _select_single_swap_candidates(
        baseline=current_table, topk_candidates=topk_candidates, depth=depth, batch=batch,
        execution_best=current_best, args=args, device=device,
        execution_cache=execution_cache, execution_cache_path=execution_cache_path,
    )
    all_trials.extend(refreshed_trials)
    for key in sorted(topk_candidates):
        cand = [s for s in refreshed_states if s["key"] == key]
        cand.sort(key=lambda s: (s["screen_step_ms"], s["name"]))
        if not cand:
            continue
        best_state = cand[0]
        if best_state["screen_step_ms"] < float(current_best["step_ms"]):
            candidate_sig = kernel_signature(best_state["table"].kernel_rows())
            candidate_exec = evaluate_policy_matrix(
                depth=depth, batch=batch, caps=caps, include_graph=include_graph,
                recompute_modes=recompute_modes, steps=steps, warmup=warmup, repeats=repeats,
                fidelity_tag="full", tuning=best_state["table"], device=device, kernel_sig=candidate_sig,
                execution_cache=execution_cache, cache_path=execution_cache_path,
                use_execution_cache=args.use_execution_cache,
            )
            for row in candidate_exec["candidates"]:
                rr = dict(row)
                rr["kernel_variant"] = best_state["name"]
                rr["kernel_signature"] = candidate_sig
                all_trials.append(rr)
            candidate_best = _best_trial(candidate_exec)
            if float(candidate_best["step_ms"]) < float(current_best["step_ms"]):
                current_table = best_state["table"]
                current_exec = candidate_exec
                current_best = candidate_best
                adopted_names.append(best_state["name"])
                print(
                    f"JOINT INCUMBENT depth={depth}: variant={best_state['name']} "
                    f"cap={current_best['grid_z_max_layers']} recompute={int(bool(current_best['recompute_activation']))} "
                    f"graph={int(bool(current_best['use_cuda_graph']))} step={current_best['step_ms']:.4f} ms"
                )

    # Pairwise rescue uses the best measured alternative per distinct kernel key, without a
    # ratio gate. This is the compact interaction surface that can catch a combination where
    # neither constituent is the overall single-swap winner.
    combined_states = refreshed_states if refreshed_states else states
    pair_states, pair_trials = _pairwise_rescue(
        baseline=current_table, states=combined_states, depth=depth, batch=batch,
        execution_best=current_best, args=args, device=device,
        execution_cache=execution_cache, execution_cache_path=execution_cache_path,
    )
    all_trials.extend(pair_trials)
    if pair_states:
        best_pair = min(pair_states, key=lambda s: (s["screen_step_ms"], s["name"]))
        if best_pair["screen_step_ms"] < float(current_best["step_ms"]):
            candidate_sig = kernel_signature(best_pair["table"].kernel_rows())
            candidate_exec = evaluate_policy_matrix(
                depth=depth, batch=batch, caps=caps, include_graph=include_graph,
                recompute_modes=recompute_modes, steps=steps, warmup=warmup, repeats=repeats,
                fidelity_tag="full", tuning=best_pair["table"], device=device, kernel_sig=candidate_sig,
                execution_cache=execution_cache, cache_path=execution_cache_path,
                use_execution_cache=args.use_execution_cache,
            )
            for row in candidate_exec["candidates"]:
                rr = dict(row)
                rr["kernel_variant"] = best_pair["name"]
                rr["kernel_signature"] = candidate_sig
                all_trials.append(rr)
            candidate_best = _best_trial(candidate_exec)
            if float(candidate_best["step_ms"]) < float(current_best["step_ms"]):
                current_table = best_pair["table"]
                current_exec = candidate_exec
                current_best = candidate_best
                adopted_names.append(best_pair["name"])
                print(
                    f"JOINT PAIRWISE INCUMBENT depth={depth}: {best_pair['name']} "
                    f"cap={current_best['grid_z_max_layers']} recompute={int(bool(current_best['recompute_activation']))} "
                    f"graph={int(bool(current_best['use_cuda_graph']))} step={current_best['step_ms']:.4f} ms"
                )

    # Final exhaustive topology pass on the selected table is authoritative.
    final_sig = kernel_signature(current_table.kernel_rows())
    final_exec = evaluate_policy_matrix(
        depth=depth, batch=batch, caps=caps, include_graph=include_graph,
        recompute_modes=recompute_modes, steps=steps, warmup=warmup, repeats=repeats,
        fidelity_tag="full", tuning=current_table, device=device, kernel_sig=final_sig,
        execution_cache=execution_cache, cache_path=execution_cache_path,
        use_execution_cache=args.use_execution_cache,
    )
    for row in final_exec["candidates"]:
        rr = dict(row)
        rr["kernel_variant"] = "final-joint" if adopted_names else "baseline-final"
        rr["kernel_signature"] = final_sig
        all_trials.append(rr)

    profile = dict(final_exec)
    profile["kernel_signature"] = final_sig
    profile["kernel_variant"] = adopted_names[-1] if adopted_names else "baseline"
    profile["joint_adoptions"] = list(adopted_names)
    profile["single_swap_screen_count"] = len(states) + len(refreshed_states)
    profile["pairwise_screen_count"] = len(pair_states)
    profile["level2_search"] = {
        "baseline_execution_matrix": True,
        "all_topk_single_swaps_screened": True,
        "coordinate_passes": 2,
        "pairwise_rescue": True,
        "final_exhaustive_execution_matrix": True,
        "execution_parameters": ["grid_z_max_layers", "recompute_activation", "use_cuda_graph"],
    }
    return profile, all_trials, current_table


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 2.3 hierarchical iPC autotuner with locked v12 Level 1")
    p.add_argument("--depths", type=int, nargs="+", default=[3, 4, 6])
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--activation", default="relu")
    p.add_argument("--caps", type=int, nargs="+", default=[0, 2, 3, 4, 5, 6])
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--rep", type=int, default=5)
    p.add_argument("--kernel-out", default="results/autotune_stage_2_kernels.json")
    p.add_argument("--policy", default="results/hierarchical_policy.json")
    p.add_argument("--cache", default="results/hierarchical_execution_cache.json")
    p.add_argument("--kernel-cache", default="results/autotune_adaptive.sqlite")
    p.add_argument("--skip-kernel", action="store_true", help="Explicitly reuse existing --kernel-out; never implicit")
    p.add_argument(
        "--reuse-kernel-result", "--reuse-level1",
        action="store_true", dest="reuse_kernel_result",
        help="STRICT: reuse the existing compatible Level-1 result; NEVER launch autotune.py",
    )
    p.add_argument("--kernel-batch-size", type=int, default=6)
    p.add_argument("--seed-evals", type=int, default=10)
    p.add_argument("--kernel-topk", type=int, default=6)
    p.add_argument("--probe-warmup", type=int, default=2)
    p.add_argument("--probe-reps", type=int, default=5)
    p.add_argument("--verify-warmup", type=int, default=10)
    p.add_argument("--verify-reps", type=int, default=40)
    p.add_argument("--temperature-max", type=float, default=76.0)
    p.add_argument("--warm-start-cache", action="store_true", help="Optional Level-1 persistent measurement reuse; OFF by default")
    p.add_argument("--reject-spill", action="store_true")
    p.add_argument("--level2-screen-warmup", type=int, default=1)
    p.add_argument("--level2-screen-repeats", type=int, default=2)
    p.add_argument("--level2-screen-steps", type=int, default=2)
    p.add_argument("--level2-pairwise-width", type=int, default=4)
    p.add_argument("--use-execution-cache", action="store_true", help="Reuse exact execution measurements; OFF by default for self-contained calibration")
    p.add_argument("--fresh-execution-search", action="store_false", dest="use_execution_cache", help="Explicitly disable execution cache")
    p.add_argument("--no-recompute-search", action="store_false", dest="search_recompute", help="Only test recompute_activation=True")
    p.add_argument("--no-graph", dest="include_graph", action="store_false")
    p.set_defaults(include_graph=True, search_recompute=True)
    a = p.parse_args()

    if a.skip_kernel and a.reuse_kernel_result:
        raise ValueError("Use either --skip-kernel or --reuse-kernel-result, not both")
    if not torch.cuda.is_available() or tuple(torch.cuda.get_device_capability()) != (8, 6):
        raise RuntimeError("Expected CUDA SM86 / RTX 3060")

    device = torch.device("cuda")
    kernel_out = Path(a.kernel_out)
    policy_out = Path(a.policy)
    execution_cache_path = Path(a.cache)
    kernel_out.parent.mkdir(parents=True, exist_ok=True)
    policy_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Stage {STAGE_VERSION} hierarchical iPC autotuner")
    print(
        f"device={torch.cuda.get_device_name(device)} capability={torch.cuda.get_device_capability(device)} "
        f"depths={a.depths} batch={a.batch} caps={a.caps} graph={'on' if a.include_graph else 'off'} "
        f"recompute_search={'on' if a.search_recompute else 'locked-true'}"
    )
    print("Level 1 engine=Stage 1.19 v12 LOCKED (no search-budget/algorithm changes)")

    # Clean-run default: Level 1 is always launched fresh. Reuse is explicit because the
    # system must be able to discover its result after the user deletes every cache file.
    if a.skip_kernel:
        compatible, reason = kernel_result_compatible(kernel_out, device=device)
        side_ok, side_reason = sidecar_compatible(kernel_out, device=device) if compatible else (False, "kernel_incompatible")
        if not compatible or not side_ok:
            raise RuntimeError(f"--skip-kernel requested but kernel result is incompatible: {reason}/{side_reason}")
        print(f"Level 1: explicit reuse {kernel_out}")
    elif a.reuse_kernel_result:
        # STRICT reuse mode: this flag is a hard prohibition on launching Level 1.
        # A missing/incompatible result is an error, never a reason to fall back to v12.
        compatible, reason = kernel_result_compatible(kernel_out, device=device)
        side_ok, side_reason = True, "sidecar_missing_ignored"
        sidecar = kernel_out.with_suffix(kernel_out.suffix + ".stage2meta.json")
        if compatible and sidecar.exists():
            side_ok, side_reason = sidecar_compatible(kernel_out, device=device)
        if not compatible or not side_ok:
            raise RuntimeError(
                "STRICT Level-1 reuse requested; refusing to run Level 1. "
                f"Existing result is not reusable: {reason}/{side_reason}. "
                f"Provide a compatible measured JSON at {kernel_out}."
            )
        print(f"Level 1: REUSING MEASURED RESULT {kernel_out} (STRICT; autotune.py will NOT be launched)")
    else:
        print("Level 1: fresh locked-v12 search (no persistent measurement reuse)")
        run_kernel_stage(a, kernel_out)
        write_kernel_sidecar(kernel_out, device=device)

    tuning, kernel_data = load_kernel_result(kernel_out)
    kernel_rows = tuning.kernel_rows()
    baseline_sig = kernel_signature(kernel_rows)
    print(f"\nLEVEL 1 complete: kernel_entries={len(kernel_rows)} kernel_signature={baseline_sig}")

    topk_candidates = collect_topk_candidates_from_run(
        tuning, kernel_data, topk=max(1, a.kernel_topk)
    )
    print("Level 1 preserved top-K candidates from THIS FRESH RUN:")
    for key in sorted(topk_candidates):
        print(f"  {_kernel_key_label(key)} -> {len(topk_candidates[key])} candidates")

    execution_cache = _load_execution_cache(execution_cache_path)
    execution_entries: list[dict[str, Any]] = []
    all_execution_trials: list[dict[str, Any]] = []
    kernel_profiles: list[dict[str, Any]] = []
    recompute_modes = [True, False] if a.search_recompute else [True]

    for depth in a.depths:
        print(f"\n=== LEVEL 2: full execution + kernel interaction depth={depth} ===")
        profile, trials, selected = tune_depth_full(
            depth=depth,
            batch=a.batch,
            caps=a.caps,
            include_graph=a.include_graph,
            recompute_modes=recompute_modes,
            steps=a.steps,
            warmup=a.warmup,
            repeats=a.rep,
            baseline=tuning,
            topk_candidates=topk_candidates,
            args=a,
            device=device,
            execution_cache=execution_cache,
            execution_cache_path=execution_cache_path,
        )
        profile_sig = kernel_signature(selected.kernel_rows())
        execution_row = dict(profile)
        execution_row["kernel_signature"] = profile_sig
        execution_entries.append(execution_row)
        all_execution_trials.extend(trials)
        kernel_profiles.append({
            "dims": profile["dims"],
            "batch": profile["batch"],
            "activation": profile["activation"],
            "dtype": profile["dtype"],
            "kernel_signature": profile_sig,
            "selection_source": profile.get("kernel_variant", "baseline"),
            "kernel_entries": selected.kernel_rows(),
        })
        best = profile["best"]
        print(
            f"BEST depth={depth}: variant={profile.get('kernel_variant', 'baseline')} "
            f"cap={best['grid_z_max_layers']} recompute={int(bool(best['recompute_activation']))} "
            f"graph={int(bool(best['use_cuda_graph']))} step={best['step_ms']:.4f} ms"
        )

    payload = {
        "schema_version": 3,
        "stage": STAGE_VERSION,
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "tilelang_version": _tilelang_version(),
        "kernel_source_fingerprint": source_fingerprint(),
        "runtime_fingerprint": runtime_fingerprint(),
        "kernel_signature": baseline_sig,
        "kernel_entries": kernel_rows,
        "kernel_candidates": [
            {
                "key": _kernel_key_label(key),
                "kind": key[0],
                "shape": {"M": key[1], "K": key[2], "B": key[3]},
                "activation": key[4],
                "dtype": key[5],
                "candidates": [asdict(cfg) for cfg in cands],
            }
            for key, cands in sorted(topk_candidates.items())
        ],
        "kernel_profiles": kernel_profiles,
        "execution_entries": execution_entries,
        "execution_trials": all_execution_trials,
        "search": {
            "level_1": "Stage 1.19 v12 locked reference engine",
            "level_1_modification_policy": "none",
            "level_1_cache_mode": "fresh by default; persistent reuse is explicit",
            "level_1_topk_source": "fresh_observations recorded by current v12 wrapper run",
            "level_2": "full finite execution matrix + all top-K single swaps + two coordinate passes + pairwise rescue + final exhaustive topology",
            "execution_parameters": ["grid_z_max_layers", "recompute_activation", "use_cuda_graph"],
            "graph_steps": int(a.steps),
            "caps_requested": a.caps,
            "recompute_modes": recompute_modes,
            "level2_screen": {"warmup": a.level2_screen_warmup, "repeats": a.level2_screen_repeats, "steps": a.level2_screen_steps},
            "pairwise_width": int(a.level2_pairwise_width),
            "execution_cache_version": EXECUTION_CACHE_VERSION,
            "execution_cache_enabled": bool(a.use_execution_cache),
            "execution_cache": str(execution_cache_path),
            "kernel_result": str(kernel_out),
            "kernel_cache": str(a.kernel_cache),
            "claim": "empirically selected within measured finite candidates; no global optimum claim",
        },
    }
    policy_out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote hierarchical policy: {policy_out}")


if __name__ == "__main__":
    main()
