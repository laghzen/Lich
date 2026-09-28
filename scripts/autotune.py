from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from _bootstrap import bootstrap
bootstrap()

from ipc_tilelang.tilelang_kernels import (
    KernelConfig,
    build_inference_update,
    build_prediction_error,
    build_weight_update,
)
from ipc_tilelang.thermal import ThermalGuard, read_telemetry
from ipc_tilelang import (
    AdaptiveFiniteTuner,
    Fidelity,
    Measurement,
    ProblemKey,
    SearchConfig,
    SQLiteCache,
)

BASELINE = KernelConfig(64, 128, 16, 128, 2, True, 8, False)
SMEM_LIMIT = 64 * 1024
SEARCH_SPACE_VERSION = "sm86-v3-eval-order"


def configs() -> list[KernelConfig]:
    """Complete Stage-1 SM86 finite search space before family-specific legality."""
    out: list[KernelConfig] = []
    for bm, bn, bk, threads, stages, sw in itertools.product(
        (16, 32, 64, 128, 256),
        (32, 64, 128, 256),
        (16, 32, 64),
        (64, 128, 256),
        (1, 2, 3),
        (False, True),
    ):
        if threads == 64 and (bm >= 256 or bn >= 256):
            continue
        if threads == 256 and bm * bn > 8192:
            continue
        if stages == 3 and bk == 64 and bm * bk + bn * bk > 49152:
            continue
        if bm == 16 and threads == 256:
            continue
        out.append(KernelConfig(bm, bn, bk, threads, stages, sw, 8, False))
    return [BASELINE] + [c for c in out if c != BASELINE]


def _warp_partition_exists(m: int, n: int, threads: int) -> bool:
    if threads < 32 or threads % 32 != 0:
        return False
    if m % 16 != 0 or n % 8 != 0:
        return False
    warps = threads // 32
    for m_warps in range(1, warps + 1):
        if warps % m_warps:
            continue
        n_warps = warps // m_warps
        if m % (m_warps * 16) == 0 and n % (n_warps * 8) == 0:
            return True
    return False


def _resource_bytes(kind: str, cfg: KernelConfig) -> int:
    bm, bn, bk, stages = cfg.block_m, cfg.block_n, cfg.block_k, cfg.num_stages
    if kind == "prediction":
        return (2 * bk * bn + bm * bk) * 2 * stages
    if kind == "inference":
        return (bk * bn + bk * bm) * 2 * stages
    if kind == "weight":
        return (bm * bn + 2 * bk * bn) * 2 * stages
    raise ValueError(kind)


def legal_for_kind(kind: str, cfg: KernelConfig) -> tuple[bool, str]:
    """Hard pre-JIT legality only: reject configs known to violate tile/T.gemm resource rules."""
    if kind in ("prediction", "inference"):
        gm, gn = cfg.block_m, cfg.block_n
    elif kind == "weight":
        gm, gn = cfg.block_m, cfg.block_k
    else:
        raise ValueError(kind)
    if not _warp_partition_exists(gm, gn, cfg.threads):
        return False, f"warp_partition T.gemm M={gm} N={gn} threads={cfg.threads}"
    smem = _resource_bytes(kind, cfg)
    if smem > SMEM_LIMIT:
        return False, f"shared_memory {smem} > {SMEM_LIMIT}"
    # Tensor-core paths used by this project require K fragments in multiples of 16.
    if cfg.block_k % 16:
        return False, "block_k_not_tensorcore_multiple"
    return True, "ok"


def split_legality(kind: str, pool: list[KernelConfig]) -> tuple[list[KernelConfig], list[tuple[KernelConfig, str]], dict[str, int]]:
    legal: list[KernelConfig] = []
    invalid: list[tuple[KernelConfig, str]] = []
    rejected: dict[str, int] = {}
    for cfg in pool:
        ok, reason = legal_for_kind(kind, cfg)
        if ok:
            legal.append(cfg)
        else:
            invalid.append((cfg, reason))
            bucket = reason.split(" ", 1)[0]
            rejected[bucket] = rejected.get(bucket, 0) + 1
    return legal, invalid, rejected


def _act_torch(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "relu": return torch.relu(x)
    if name == "tanh": return torch.tanh(x)
    if name == "sigmoid": return torch.sigmoid(x)
    if name == "silu": return torch.nn.functional.silu(x)
    if name == "gelu": return torch.nn.functional.gelu(x, approximate="tanh")
    raise ValueError(name)


def _dact_torch(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "relu": return (x > 0).to(torch.float32)
    if name == "tanh":
        y = torch.tanh(x); return 1.0 - y * y
    if name == "sigmoid":
        y = torch.sigmoid(x); return y * (1.0 - y)
    if name == "silu":
        s = torch.sigmoid(x); return s * (1.0 + x * (1.0 - s))
    if name == "gelu":
        x32 = x.float()
        k = 0.7978845608028654
        c = 0.044715
        u = k * (x32 + c * x32 * x32 * x32)
        t = torch.tanh(u)
        du = k * (1.0 + 3.0 * c * x32 * x32)
        return 0.5 * (1.0 + t) + 0.5 * x32 * (1.0 - t * t) * du
    raise ValueError(name)


def _inspect_kernel(kernel) -> dict[str, object]:
    info: dict[str, object] = {"registers_estimate": None, "spill_suspected": None, "source_available": False}
    getter = getattr(kernel, "get_kernel_source", None)
    if getter is None:
        return info
    try:
        source = str(getter())
    except Exception:
        return info
    info["source_available"] = True
    total = 0
    found = False
    for bits_s, count_s in re.findall(r"\.reg\s+\.b(16|32|64|128)\s+%[A-Za-z_]+<([0-9]+)>", source):
        found = True
        total += int(count_s) * max(1, math.ceil(int(bits_s) / 32))
    if found:
        info["registers_estimate"] = total
    info["spill_suspected"] = bool(re.search(r"\b(?:ld|st)\.local\b", source) or ".local" in source)
    return info


def _time(fn, guard: ThermalGuard, warmup: int, rep: int) -> float:
    for _ in range(max(0, warmup)):
        guard.wait_until_safe()
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(max(2, rep)):
        guard.wait_until_safe()
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        fn()
        ev1.record()
        ev1.synchronize()
        samples.append(ev0.elapsed_time(ev1))
    return float(statistics.median(samples))


def _make_case(kind: str, M: int, K: int, B: int, device: torch.device, activation: str, cfg: KernelConfig):
    if kind == "prediction":
        x_up = torch.randn((K, B), device=device, dtype=torch.float16)
        w = torch.randn((M, K), device=device, dtype=torch.float16) * 0.05
        x_lo = torch.randn((M, B), device=device, dtype=torch.float16)
        e = torch.empty_like(x_lo)
        kernel = build_prediction_error(M, K, B, activation, "float16", cfg)
        def fn(): kernel(x_up, w, x_lo, e)
        ref = x_lo.float() - torch.matmul(w.float(), _act_torch(x_up.float(), activation))
        return kernel, fn, (e, ref, "prediction")
    if kind == "inference":
        x = torch.randn((M, B), device=device, dtype=torch.float16)
        e = torch.randn((M, B), device=device, dtype=torch.float16)
        w_lower = torch.randn((K, M), device=device, dtype=torch.float16) * 0.05
        e_lower = torch.randn((K, B), device=device, dtype=torch.float16)
        before = x.clone()
        kernel = build_inference_update(M, K, B, activation, "float16", cfg, 0.5, False)
        def fn(): kernel(x, e, w_lower, e_lower)
        ref = before.float() + 0.5 * (-e.float() + _dact_torch(before.float(), activation) * torch.matmul(w_lower.float().T, e_lower.float()))
        return kernel, fn, (x, ref, "inference")
    if kind == "weight":
        w = torch.randn((M, K), device=device, dtype=torch.float16) * 0.05
        e = torch.randn((M, B), device=device, dtype=torch.float16)
        x_up = torch.randn((K, B), device=device, dtype=torch.float16)
        before = w.clone()
        kernel = build_weight_update(M, K, B, activation, "float16", cfg, 1e-4, True)
        def fn(): kernel(w, e, x_up)
        ref = before.float() + 1e-4 * torch.matmul(e.float(), _act_torch(x_up.float(), activation).T)
        return kernel, fn, (w, ref, "weight")
    raise ValueError(kind)


def _correctness_probe(outputs: tuple[torch.Tensor, torch.Tensor, str], *, rtol: float, atol: float) -> tuple[bool, str | None]:
    out, ref, kind = outputs
    got = out.float()
    diff = (got - ref).abs()
    allowed = atol + rtol * ref.abs()
    bad = diff > allowed
    ratio = float(bad.float().mean().item())
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    if ratio > 0.01:
        return False, f"{kind} correctness mismatch ratio={ratio:.6f} max_abs={max_abs:.6g}"
    return True, None


def _tilelang_version() -> str:
    try:
        import tilelang
        return str(getattr(tilelang, "__version__", "unknown"))
    except Exception:
        return "unknown"


def _source_fingerprint() -> str:
    # Cache fingerprint deliberately tracks the kernel generator, not autotuner source.
    p = Path(__file__).resolve().parents[1] / "src" / "ipc_tilelang" / "tilelang_kernels.py"
    if not p.exists():
        return "unknown"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:20]


def _device_fingerprint(device: torch.device) -> str:
    props = torch.cuda.get_device_properties(device)
    payload = {
        "name": torch.cuda.get_device_name(device),
        "cc": list(torch.cuda.get_device_capability(device)),
        "total_memory": int(props.total_memory),
        "torch": torch.__version__,
        "tilelang": _tilelang_version(),
        "kernel_source": _source_fingerprint(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]


def profile_shapes(profile: str, B: int, hidden: int = 64) -> dict[str, list[tuple[int, int, int]]]:
    if profile == "mnist64":
        return {
            "prediction": [(10, hidden, B), (hidden, hidden, B), (hidden, 784, B)],
            "inference": [(hidden, 10, B), (hidden, hidden, B)],
            "weight": [(10, hidden, B), (hidden, hidden, B), (hidden, 784, B)],
        }
    raise ValueError(f"Unsupported profile: {profile}")



def _parse_legacy_cfg(cfg_data: Any) -> KernelConfig | None:
    if not isinstance(cfg_data, dict):
        return None
    try:
        return KernelConfig(**{
            k: cfg_data[k] for k in (
                "block_m", "block_n", "block_k", "threads", "num_stages",
                "swizzle", "swizzle_panel", "shared_swizzle",
            )
        })
    except Exception:
        return None


def _legacy_prior_configs(path: Path, *, kind: str, M: int, K: int, B: int, activation: str) -> list[KernelConfig]:
    """Return historical configs as deterministic warm-start anchors."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = data.get("results")
    if not isinstance(rows, dict):
        return []
    out: list[KernelConfig] = []
    for row in rows.values():
        if not isinstance(row, dict) or row.get("kind") != kind:
            continue
        shape = row.get("shape") or {}
        if (int(shape.get("M", -1)), int(shape.get("K", -1)), int(shape.get("B", -1))) != (M, K, B):
            continue
        if str(row.get("activation", "relu")) != activation or str(row.get("dtype", "float16")) != "float16":
            continue
        cfg = _parse_legacy_cfg(row.get("config"))
        if cfg is not None and cfg not in out:
            out.append(cfg)
    return out


def _import_legacy_measurements(
    cache: SQLiteCache,
    path: Path,
    *,
    problem: ProblemKey,
    legal_pool: list[KernelConfig],
    device: torch.device,
) -> tuple[int, int, str | None]:
    """Import valid historical Stage-1 JSON measurements into the v9 cache.

    The old JSON cache does not contain the new ProblemKey digest, so measurements
    are migrated only after exact shape/family/activation/dtype matching and an
    environment compatibility check when the old file records that metadata.
    Existing current-fingerprint data wins unless the historical value is strictly
    better and the current record is only a low-fidelity probe; the candidate is then
    re-verified by the v9 high-fidelity stage.
    """
    if not path.exists():
        return 0, 0, "file_missing"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return 0, 0, f"json_error:{type(exc).__name__}"

    expected_name = torch.cuda.get_device_name(device)
    expected_cc = list(torch.cuda.get_device_capability(device))
    expected_tl = _tilelang_version()
    mismatches: list[str] = []
    old_name = data.get("device")
    old_cc = data.get("capability")
    old_tl = data.get("tilelang_version")
    if old_name is not None and str(old_name) != expected_name:
        mismatches.append("device")
    if old_cc is not None and list(old_cc) != expected_cc:
        mismatches.append("capability")
    if old_tl is not None and str(old_tl) != expected_tl:
        mismatches.append("tilelang")
    if mismatches:
        return 0, 0, "metadata_mismatch:" + ",".join(mismatches)

    rows = data.get("results")
    if not isinstance(rows, dict):
        return 0, 0, "no_results"
    legal_set = set(legal_pool)
    best_legacy: dict[KernelConfig, tuple[float, dict[str, Any]]] = {}
    for row in rows.values():
        if not isinstance(row, dict) or row.get("kind") != problem.kind or row.get("status", "ok") != "ok":
            # Failures are intentionally not migrated from the legacy cache: their
            # failure semantics were tied to the old evaluator. Current v9 failures
            # are persisted immediately in SQLite instead.
            continue
        shape = row.get("shape") or {}
        if (int(shape.get("M", -1)), int(shape.get("K", -1)), int(shape.get("B", -1))) != (problem.M, problem.K, problem.B):
            continue
        if str(row.get("activation", "relu")) != problem.activation or str(row.get("dtype", "float16")) != problem.dtype:
            continue
        cfg = _parse_legacy_cfg(row.get("config"))
        ms = row.get("median_ms")
        if cfg is None or cfg not in legal_set or not isinstance(ms, (int, float)):
            continue
        ms_f = float(ms)
        if not math.isfinite(ms_f) or ms_f <= 0.0:
            continue
        prev = best_legacy.get(cfg)
        if prev is None or ms_f < prev[0]:
            best_legacy[cfg] = (ms_f, row)

    imported = 0
    replaced = 0
    for cfg, (ms, row) in best_legacy.items():
        existing = cache.get(problem, cfg)
        if existing is not None and existing.successful and existing.latency_ms is not None:
            # A better legacy measurement is useful as a prior, but it must still
            # receive the current v9 correctness/high-fidelity verification later.
            if existing.fidelity == 0 and ms < existing.latency_ms:
                cache.put(problem, cfg, Measurement.measured(
                    ms, correctness_ok=True, fidelity=0, origin="legacy_json"
                ))
                replaced += 1
            continue
        if existing is not None:
            continue
        cache.put(problem, cfg, Measurement.measured(
            ms, correctness_ok=True, fidelity=0, origin="legacy_json"
        ))
        imported += 1
    return imported, replaced, None


def _rows_from_result(kind: str, M: int, K: int, B: int, activation: str, result, *, device: torch.device) -> dict[str, Any]:
    row = {
        "kind": kind,
        "shape": {"M": M, "K": K, "B": B},
        "activation": activation,
        "dtype": "float16",
        "candidate_pool": result.space_total,
        "legal_pool": result.legal_total,
        "static_invalid": result.static_invalid,
        "tested": result.attempted,
        "successful": result.successful,
        "compile_failures": result.compile_failures,
        "correctness_failures": result.correctness_failures,
        "model_pruned": result.model_pruned,
        "region_pruned": result.region_pruned,
        "untested": result.untested,
        "search_engine": result.search_engine,
        "adaptive_search": result.adaptive_search,
        "hierarchical_pruning": result.hierarchical_pruning,
        "stratified_acquisition": result.stratified_acquisition,
        "stopped_by_global_bound": result.stopped_by_global_bound,
        "rounds": result.rounds,
        "best": [],
    }
    if result.best_config is not None and result.best_latency_ms is not None:
        row["best"] = [{"median_ms": result.best_latency_ms, "config": asdict(result.best_config)}]
    return row


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 1.19 unlimited-frontier adaptive finite-space iPC autotuner for SM86")
    p.add_argument("--kind", choices=["prediction", "inference", "weight", "all"], default="prediction")
    p.add_argument("--M", type=int, default=64)
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--B", type=int, default=128)
    p.add_argument("--activation", default="relu")
    p.add_argument("--profile", choices=["custom", "mnist64"], default="custom")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--warmup", type=int, default=5, help="Full verification warmup")
    p.add_argument("--rep", type=int, default=20, help="Full verification repetitions")
    # Backward-compatible hidden aliases. They DO NOT cap the search anymore.
    p.add_argument("--max-configs", "--max-evals", dest="deprecated_max_evals", type=int, default=None,
                   help=argparse.SUPPRESS)
    p.add_argument("--seed-evals", type=int, default=10)
    p.add_argument("--batch-configs", "--batch-size", dest="batch_size", type=int, default=6)
    p.add_argument("--topk", type=int, default=6)
    p.add_argument("--verify-reps", type=int, default=40)
    p.add_argument("--verify-warmup", type=int, default=10)
    p.add_argument("--probe-reps", type=int, default=5)
    p.add_argument("--probe-warmup", type=int, default=2)
    p.add_argument("--beta", type=float, default=3.5, help="LCB uncertainty multiplier for semantic-region pruning")
    p.add_argument("--prune-margin", type=float, default=0.0)
    p.add_argument("--cache", default="results/autotune_adaptive.sqlite")
    p.add_argument("--warm-start-cache", action="store_true", help="Optional stale measurement warm-start; OFF by default")
    p.add_argument("--legacy-cache", default="results/autotune_cache.json",
                   help="Legacy JSON warm-start source; only read with --warm-start-cache")
    p.add_argument("--out", default="results/autotune_stage_1_19.json")
    p.add_argument("--temperature-max", type=float, default=76.0)
    p.add_argument("--reject-spill", action="store_true", help="Treat detected local-memory spills as rejected candidates")
    p.add_argument("--rtol", type=float, default=2e-2)
    p.add_argument("--atol", type=float, default=2e-2)
    a = p.parse_args()

    if a.deprecated_max_evals is not None:
        print(
            f"WARNING: --max-evals/--max-configs={a.deprecated_max_evals} is deprecated and ignored; "
            "Stage 1.19 now searches the full unresolved frontier."
        )

    if not torch.cuda.is_available() or tuple(torch.cuda.get_device_capability()) != (8, 6):
        raise RuntimeError("Expected NVIDIA SM86 CUDA device (RTX 3060 Laptop class)")
    device = torch.device("cuda")
    guard = ThermalGuard(max_temp_c=a.temperature_max)

    if a.profile == "mnist64":
        shape_map = profile_shapes("mnist64", a.B, a.hidden)
        kinds = [a.kind] if a.kind != "all" else ["prediction", "inference", "weight"]
        shapes = [(k, s) for k in kinds for s in shape_map[k]]
    else:
        shapes = [(a.kind, (a.M, a.K, a.B))] if a.kind != "all" else [
            ("prediction", (a.M, a.K, a.B)),
            ("inference", (a.M, a.K, a.B)),
            ("weight", (a.M, a.K, a.B)),
        ]

    pool = configs()
    print("Stage 1.19 v11 adaptive finite-space autotuner")
    print(f"device={torch.cuda.get_device_name(device)} capability={torch.cuda.get_device_capability(device)} global_pool={len(pool)}")
    print("search=semantic-tree + online EI/local racing + global scouts + factorized GP + conservative region pruning")

    cache = SQLiteCache(a.cache)
    device_fp = _device_fingerprint(device)
    all_results: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []

    for shape_index, (kind, (M, K, B)) in enumerate(shapes, 1):
        legal_pool, invalid, rejected = split_legality(kind, pool)
        problem = ProblemKey(
            kind=kind, M=M, K=K, B=B, activation=a.activation, dtype="float16",
            device_fingerprint=device_fp, code_fingerprint=_source_fingerprint(),
            search_space_version=SEARCH_SPACE_VERSION,
        )
        if a.warm_start_cache:
            legacy_imported, legacy_replaced, legacy_status = _import_legacy_measurements(
                cache, Path(a.legacy_cache), problem=problem, legal_pool=legal_pool, device=device
            )
            if legacy_imported or legacy_replaced:
                print(
                    f"warm-start cache: imported={legacy_imported} replaced_probe={legacy_replaced} "
                    f"source={a.legacy_cache}"
                )
            elif legacy_status is not None:
                print(f"warm-start cache: status={legacy_status} source={a.legacy_cache}")
        else:
            legacy_imported = legacy_replaced = 0
            print("measurement cache: OFF (online search starts from fresh GPU measurements)")

        search_cfg = SearchConfig(
            seed_evals=max(0, a.seed_evals),
            batch_size=max(1, a.batch_size),
            verification_topk=max(1, a.topk),
            verification_reps=max(a.verify_reps, a.rep),
            verification_warmup=max(a.verify_warmup, a.warmup),
            probe_warmup=a.probe_warmup,
            probe_reps=a.probe_reps,
            beta=a.beta,
            prune_margin=a.prune_margin,
        )

        def evaluate(cfg: KernelConfig, fidelity: Fidelity) -> Measurement:
            kernel, fn, check = _make_case(kind, M, K, B, device, a.activation, cfg)
            inspection = _inspect_kernel(kernel)
            # Fast negative resource checks happen after compilation but before timing.
            if a.reject_spill and inspection.get("spill_suspected") is True:
                return Measurement.failed("spill", "generated kernel contains local-memory accesses", fidelity=fidelity.level)
            # The output/state tensor is not valid until the kernel has executed once.
            # v7 accidentally probed correctness before this first execution, so every
            # otherwise-compilable candidate was rejected as an uninitialized-output mismatch.
            # Execute one guarded correctness pass first, synchronize, then inspect the result.
            guard.wait_until_safe()
            fn()
            torch.cuda.synchronize()
            ok, reason = _correctness_probe(check, rtol=a.rtol, atol=a.atol)
            if not ok:
                print(f"      REJECT correctness: {reason}")
                return Measurement.failed("correctness", reason or "correctness mismatch", fidelity=fidelity.level, correctness_ok=False)
            ms = _time(fn, guard, fidelity.warmup, fidelity.rep)
            t = read_telemetry(0)
            return Measurement.measured(
                ms,
                correctness_ok=True,
                fidelity=fidelity.level,
                temperature_c=t.temperature_c,
                power_w=t.power_w,
                clock_mhz=t.clock_sm_mhz,
                spill_bytes=None,
                registers_per_thread=(int(inspection["registers_estimate"]) if isinstance(inspection.get("registers_estimate"), int) else None),
            )

        tuner = AdaptiveFiniteTuner(cache, search_cfg)
        legacy_priors = _legacy_prior_configs(Path(a.legacy_cache), kind=kind, M=M, K=K, B=B, activation=a.activation) if a.warm_start_cache else []
        prior_configs = [BASELINE] + legacy_priors
        result = tuner.run(problem, legal_pool, evaluate, static_invalid=invalid, prior_configs=prior_configs, use_cache=a.warm_start_cache)
        row = _rows_from_result(kind, M, K, B, a.activation, result, device=device)
        row["rejected"] = rejected
        row["cache"] = str(a.cache)
        row["legacy_cache"] = str(a.legacy_cache)
        row["legacy_cache_imported"] = legacy_imported
        row["legacy_cache_replaced_probe"] = legacy_replaced
        row["new_gpu_evals"] = result.metadata.get("new_gpu_evals", 0)
        row["verification_gpu_evals"] = result.metadata.get("verification_gpu_evals", 0)
        row["cache_hits"] = result.metadata.get("cache_hits", 0)
        row["active_untested"] = result.metadata.get("active_untested", result.untested)
        all_results.append(row)
        print(
            f"\n[{shape_index}/{len(shapes)}] {kind} M={M} K={K} B={B} "
            f"legal={result.legal_total}/{len(pool)} static_invalid={result.static_invalid} "
            f"cache_hits={result.metadata.get('cache_hits', 0)} "
            f"new_gpu_evals={result.metadata.get('new_gpu_evals', 0)} "
            f"tested={result.attempted} successful={result.successful} "
            f"model_pruned={result.model_pruned} region_pruned={result.region_pruned} "
            f"active_untested={result.metadata.get('active_untested', result.untested)} "
            f"untested_total={result.untested}"
        )
        if result.best_config is not None:
            print(f"BEST {kind} M={M} K={K} B={B}: {result.best_latency_ms:.4f} ms {result.best_config}")
            entries.append({
                "kind": kind,
                "shape": {"M": M, "K": K, "B": B},
                "activation": a.activation,
                "dtype": "float16",
                "median_ms": result.best_latency_ms,
                "config": asdict(result.best_config),
            })
        else:
            raise RuntimeError(f"No successful configuration for {kind} M={M} K={K} B={B}")

    payload = {
        "schema_version": 2,
        "stage": "1.19",
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "tilelang_version": _tilelang_version(),
        "entries": entries,
        "results": all_results,
        "search": {
            "search_engine": AdaptiveFiniteTuner.ENGINE,
            "global_candidate_pool": len(pool),
            "evaluation_budget": "unlimited_until_frontier_exhausted_or_pruned",
            "deprecated_max_evals_argument": a.deprecated_max_evals,
            "seed_evals": a.seed_evals,
            "batch_size": a.batch_size,
            "probe": {"warmup": a.probe_warmup, "rep": a.probe_reps},
            "verification": {"warmup": a.verify_warmup, "rep": max(a.verify_reps, a.rep), "topk": a.topk},
            "beta": a.beta,
            "prune_margin": a.prune_margin,
            "cache": str(a.cache),
            "legacy_cache": str(a.legacy_cache),
            "semantic_tree": True,
            "factorized_surrogate": True,
            "hierarchical_pruning": True,
            "stratified_acquisition": True,
            "persistent_failure_memory": True,
            "full_space_representation": True,
            "claim": "entire finite legal space represented; only a data-dependent subset is benchmarked; statistical pruning is not a proof of a global optimum",
        },
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"saved={out}")


if __name__ == "__main__":
    main()
