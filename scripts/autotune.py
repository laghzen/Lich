from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import statistics
import hashlib
from dataclasses import asdict
from pathlib import Path

import torch

from _bootstrap import bootstrap
bootstrap()

from ipc_tilelang.tilelang_kernels import (
    KernelConfig,
    build_inference_update,
    build_prediction_error,
    build_weight_update,
)
from ipc_tilelang.thermal import ThermalGuard


BASELINE = KernelConfig(64, 128, 16, 128, 2, True, 8, False)
SMEM_LIMIT = 64 * 1024


def configs() -> list[KernelConfig]:
    """Stage 1.12 SM86 candidate space.

    The global search space is generated once, but legality/resource filtering
    is applied per kernel family (prediction/inference/weight).
    """
    out: list[KernelConfig] = []
    for bm, bn, bk, threads, stages, sw in itertools.product(
        (16, 32, 64, 128, 256),
        (32, 64, 128, 256),
        (16, 32, 64),
        (64, 128, 256),
        (1, 2, 3),
        (False, True),
    ):
        # Keep the existing conservative Stage-1.11 search-space guards.
        if threads == 64 and (bm >= 256 or bn >= 256):
            continue
        if threads == 256 and bm * bn > 8192:
            continue
        if stages == 3 and bk == 64 and bm * bk + bn * bk > 49152:
            continue
        if bm == 16 and threads == 256:
            continue
        out.append(KernelConfig(bm, bn, bk, threads, stages, sw, 8, False))

    # Keep the known Stage-1.10 baseline first for deterministic A/B comparison.
    return [BASELINE] + [c for c in out if c != BASELINE]


def _warp_partition_exists(m: int, n: int, threads: int) -> bool:
    """Conservative SM86 TileLang T.gemm warp-partition legality check.

    TileLang's CUDA GEMM partitioning needs a warp decomposition that covers
    the output tile in 16x8 fragments. This mirrors the relevant legality
    condition without compiling the kernel.
    """
    if threads < 32 or threads % 32 != 0:
        return False
    if m % 16 != 0 or n % 8 != 0:
        return False

    num_warps = threads // 32
    for m_warps in range(1, num_warps + 1):
        if num_warps % m_warps != 0:
            continue
        n_warps = num_warps // m_warps
        if m % (m_warps * 16) == 0 and n % (n_warps * 8) == 0:
            return True
    return False


def _resource_bytes(kind: str, cfg: KernelConfig) -> int:
    """Return the existing conservative shared-memory estimate for one kernel."""
    bm, bn, bk, stages = (
        cfg.block_m,
        cfg.block_n,
        cfg.block_k,
        cfg.num_stages,
    )

    if kind == "prediction":
        return (2 * bk * bn + bm * bk) * 2 * stages
    if kind == "inference":
        return (bk * bn + bk * bm) * 2 * stages
    if kind == "weight":
        return (bm * bn + 2 * bk * bn) * 2 * stages
    raise ValueError(kind)


def legal_for_kind(kind: str, cfg: KernelConfig) -> tuple[bool, str]:
    """Static legality/resource filter, applied before JIT compilation."""
    if kind in ("prediction", "inference"):
        gm, gn = cfg.block_m, cfg.block_n
    elif kind == "weight":
        # Weight path computes E @ A^T. The logical T.gemm output tile is BM x BK.
        gm, gn = cfg.block_m, cfg.block_k
    else:
        raise ValueError(kind)

    if not _warp_partition_exists(gm, gn, cfg.threads):
        return False, f"no valid SM86 warp partition for T.gemm M={gm}, N={gn}, threads={cfg.threads}"

    smem = _resource_bytes(kind, cfg)
    if smem > SMEM_LIMIT:
        return False, f"estimated shared memory {smem} > {SMEM_LIMIT}"

    return True, "ok"


def legal_configs(kind: str, pool: list[KernelConfig]) -> tuple[list[KernelConfig], dict[str, int]]:
    """Filter a global pool into the family-specific legal pool."""
    legal: list[KernelConfig] = []
    rejected: dict[str, int] = {}
    for cfg in pool:
        ok, reason = legal_for_kind(kind, cfg)
        if ok:
            legal.append(cfg)
        else:
            bucket = reason.split(" ", 1)[0]
            rejected[bucket] = rejected.get(bucket, 0) + 1
    return legal, rejected


def _cfg_key(cfg: KernelConfig) -> str:
    return json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":"))


def _config_features(cfg: KernelConfig) -> tuple[float, ...]:
    # Log tiles because performance changes approximately multiplicatively
    # across this discrete search space; threads/stages remain direct features.
    return (
        math.log2(cfg.block_m),
        math.log2(cfg.block_n),
        math.log2(cfg.block_k),
        cfg.threads / 32.0,
        float(cfg.num_stages),
        float(cfg.swizzle),
        float(cfg.shared_swizzle),
    )


def _distance(a: KernelConfig, b: KernelConfig) -> float:
    xa = _config_features(a)
    xb = _config_features(b)
    weights = (1.5, 1.5, 1.25, 1.0, 0.6, 0.4, 0.4)
    return math.sqrt(sum(w * (u - v) ** 2 for u, v, w in zip(xa, xb, weights)))


def _maximin_pick(
    candidates: list[KernelConfig],
    chosen: list[KernelConfig],
) -> KernelConfig:
    if not chosen:
        return candidates[0]

    best = None
    best_score = -float("inf")
    chosen_set = set(chosen)
    for cfg in candidates:
        if cfg in chosen_set:
            continue
        score = min(_distance(cfg, old) for old in chosen)
        if score > best_score:
            best_score = score
            best = cfg
    if best is None:
        raise RuntimeError("maximin selection exhausted candidates")
    return best


def _predict_latency(
    cfg: KernelConfig,
    observations: list[tuple[KernelConfig, float]],
    k: int = 7,
) -> tuple[float, float]:
    """Distance-weighted kNN surrogate -> (predicted_ms, uncertainty)."""
    if not observations:
        return float("inf"), float("inf")

    distances = sorted(
        ((_distance(cfg, c), ms) for c, ms in observations),
        key=lambda x: x[0],
    )
    nearest = distances[: min(k, len(distances))]

    # Exact duplicate should be essentially noise-free.
    if nearest[0][0] < 1e-12:
        return nearest[0][1], 0.0

    eps = 1e-6
    weights = [1.0 / (d + eps) for d, _ in nearest]
    total_w = sum(weights)
    pred = sum(w * ms for w, (_, ms) in zip(weights, nearest)) / total_w

    # Local weighted spread is a useful acquisition uncertainty proxy.
    variance = sum(w * (ms - pred) ** 2 for w, (_, ms) in zip(weights, nearest)) / total_w
    uncertainty = math.sqrt(max(variance, 0.0)) + nearest[-1][0] * max(pred, 1e-6) * 0.05
    return pred, uncertainty


def adaptive_order(
    legal_pool: list[KernelConfig],
    seed_configs: list[KernelConfig],
    observations: list[tuple[KernelConfig, float]],
) -> list[KernelConfig]:
    """Return the next deterministic batch ordered by acquisition value.

    Already observed configurations are excluded. Before enough observations
    exist, use maximin exploration. Afterwards use lower-confidence-bound
    selection with a small exploration bonus.
    """
    remaining = [c for c in legal_pool if c not in {x[0] for x in observations}]
    if not remaining:
        return []

    chosen = list(seed_configs)
    while chosen and chosen[-1] not in {c for c, _ in observations}:
        # Seed configs are candidate reservations, not fake observations.
        break

    ordered: list[KernelConfig] = []

    # Exploration phase: spread measurements over the discrete space.
    while len(observations) + len(ordered) < 8 and remaining:
        pick = _maximin_pick(remaining, chosen + ordered)
        ordered.append(pick)
        remaining.remove(pick)

    if not remaining:
        return ordered

    # Exploitation + exploration.
    scored: list[tuple[float, float, KernelConfig]] = []
    obs_for_model = list(observations)
    for cfg in remaining:
        pred, unc = _predict_latency(cfg, obs_for_model)
        # Lower is better. beta=1.0 gives enough exploration to avoid a narrow
        # local minimum on the first few measured points.
        acquisition = pred - 1.0 * unc
        scored.append((acquisition, pred, cfg))
    scored.sort(key=lambda x: (x[0], x[1], _cfg_key(x[2])))
    ordered.extend(cfg for _, _, cfg in scored)
    return ordered


def select_adaptive_batch(
    legal_pool: list[KernelConfig],
    observations: list[tuple[KernelConfig, float]],
    attempted: set[KernelConfig],
    budget: int,
    batch_size: int,
) -> list[KernelConfig]:
    if budget <= 0:
        return []

    remaining_budget = budget - len(attempted)
    if remaining_budget <= 0:
        return []

    remaining = [cfg for cfg in legal_pool if cfg not in attempted]
    if not remaining:
        return []

    # Baseline is always measured first if legal and not already attempted.
    priority: list[KernelConfig] = []
    if BASELINE in remaining:
        priority.append(BASELINE)

    # Before enough successful observations exist, deliberately spread probes
    # through the legal space. Once the initial design is populated, switch to
    # surrogate-guided exploitation/exploration.
    chosen = [c for c, _ in observations] + priority
    ordered: list[KernelConfig] = []
    sim_remaining = [c for c in remaining if c not in priority]

    target_initial = max(
        0,
        min(8 - len(observations), remaining_budget - len(priority)),
    )
    while len(ordered) < target_initial and sim_remaining:
        pick = _maximin_pick(sim_remaining, chosen + ordered)
        ordered.append(pick)
        sim_remaining.remove(pick)

    if not sim_remaining:
        return (priority + ordered)[: min(batch_size, remaining_budget)]

    scored: list[tuple[float, float, str, KernelConfig]] = []
    if observations:
        for cfg in sim_remaining:
            pred, unc = _predict_latency(cfg, observations)
            acquisition = pred - unc
            scored.append((acquisition, pred, _cfg_key(cfg), cfg))
        scored.sort(key=lambda x: (x[0], x[1], x[2]))
        ordered.extend(x[3] for x in scored)
    else:
        # No successful measurements exist yet and the initial design filled
        # the requested batch. The deterministic maximin points are enough.
        pass

    result = priority + ordered
    return result[: min(batch_size, remaining_budget)]


def _time(fn, guard: ThermalGuard, warmup: int, rep: int) -> float:
    for _ in range(warmup):
        guard.wait_until_safe()
        fn()
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(rep):
        guard.wait_until_safe()
        ev0.record()
        fn()
        ev1.record()
        ev1.synchronize()
        samples.append(ev0.elapsed_time(ev1))
    return statistics.median(samples)


def _inspect_kernel(kernel) -> dict[str, object]:
    info: dict[str, object] = {
        "registers_estimate": None,
        "spill_suspected": None,
        "source_available": False,
    }
    getter = getattr(kernel, "get_kernel_source", None)
    if getter is None:
        return info
    try:
        source = str(getter())
    except Exception:
        return info
    info["source_available"] = True
    total = 0
    found_reg_decl = False
    for bits_s, count_s in re.findall(
        r"\.reg\s+\.b(16|32|64|128)\s+%[A-Za-z_]+<([0-9]+)>",
        source,
    ):
        found_reg_decl = True
        bits = int(bits_s)
        count = int(count_s)
        total += count * max(1, math.ceil(bits / 32))
    if found_reg_decl:
        info["registers_estimate"] = total
    info["spill_suspected"] = bool(
        re.search(r"\b(?:ld|st)\.local\b", source) or ".local" in source
    )
    return info


def _make_case(
    kind: str,
    M: int,
    K: int,
    B: int,
    device: torch.device,
    activation: str,
    cfg: KernelConfig,
):
    if kind == "prediction":
        x_up = torch.randn((K, B), device=device, dtype=torch.float16)
        w = torch.randn((M, K), device=device, dtype=torch.float16) * 0.05
        x_lo = torch.randn((M, B), device=device, dtype=torch.float16)
        e = torch.empty_like(x_lo)
        kernel = build_prediction_error(M, K, B, activation, "float16", cfg)
        return kernel, lambda: kernel(x_up, w, x_lo, e)

    if kind == "inference":
        x = torch.randn((M, B), device=device, dtype=torch.float16)
        e = torch.randn((M, B), device=device, dtype=torch.float16)
        # Stage 1.11 keeps the real kernel ABI W_lower[K,M].
        w_lower = torch.randn((K, M), device=device, dtype=torch.float16) * 0.05
        e_lower = torch.randn((K, B), device=device, dtype=torch.float16)
        kernel = build_inference_update(
            M, K, B, activation, "float16", cfg, 0.5, False
        )
        return kernel, lambda: kernel(x, e, w_lower, e_lower)

    if kind == "weight":
        w = torch.randn((M, K), device=device, dtype=torch.float16) * 0.05
        e = torch.randn((M, B), device=device, dtype=torch.float16)
        x_up = torch.randn((K, B), device=device, dtype=torch.float16)
        kernel = build_weight_update(M, K, B, activation, "float16", cfg, 1e-4, True)
        return kernel, lambda: kernel(w, e, x_up)

    raise ValueError(kind)


def profile_shapes(
    profile: str, B: int, hidden: int = 64
) -> dict[str, list[tuple[int, int, int]]]:
    if profile == "mnist64":
        return {
            "prediction": [
                (10, hidden, B),
                (hidden, hidden, B),
                (hidden, 784, B),
            ],
            "inference": [
                (hidden, 10, B),
                (hidden, hidden, B),
            ],
            "weight": [
                (10, hidden, B),
                (hidden, hidden, B),
                (hidden, 784, B),
            ],
        }
    raise ValueError(f"Unsupported profile: {profile}")


def _load_cache(path: Path) -> dict:
    if not path.exists():
        return {"schema_version": 1, "results": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != 1 or not isinstance(data.get("results"), dict):
            return {"schema_version": 1, "results": {}}
        return data
    except Exception:
        return {"schema_version": 1, "results": {}}


def _save_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    tmp.replace(path)


def _tilelang_version() -> str | None:
    try:
        import tilelang
        return str(getattr(tilelang, "__version__", "unknown"))
    except Exception:
        return None


def _cache_key(
    kind: str,
    M: int,
    K: int,
    B: int,
    activation: str,
    cfg: KernelConfig,
    device: torch.device,
) -> str:
    payload = {
        "kind": kind,
        "M": M,
        "K": K,
        "B": B,
        "activation": activation,
        "dtype": "float16",
        "cfg": cfg.__dict__,
        "device_name": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "tilelang": _tilelang_version(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _cached_observations(
    cache: dict,
    kind: str,
    M: int,
    K: int,
    B: int,
    activation: str,
    legal_pool: list[KernelConfig],
    device: torch.device,
) -> tuple[list[tuple[KernelConfig, float]], set[KernelConfig], dict[KernelConfig, dict]]:
    observations: list[tuple[KernelConfig, float]] = []
    attempted: set[KernelConfig] = set()
    rows: dict[KernelConfig, dict] = {}

    for cfg in legal_pool:
        key = _cache_key(kind, M, K, B, activation, cfg, device)
        row = cache["results"].get(key)
        if not isinstance(row, dict):
            continue

        status = row.get("status")
        if status in {"ok", "error", "reject_spill", "reject_registers"}:
            attempted.add(cfg)

        if status != "ok":
            continue

        ms = row.get("median_ms")
        if isinstance(ms, (int, float)) and math.isfinite(float(ms)):
            observations.append((cfg, float(ms)))
            rows[cfg] = {
                "kind": kind,
                "shape": {"M": M, "K": K, "B": B},
                "activation": activation,
                "dtype": "float16",
                "median_ms": float(ms),
                "config": cfg.__dict__,
                "cached": True,
                "registers_estimate": row.get("registers_estimate"),
                "spill_suspected": row.get("spill_suspected"),
                "source_available": row.get("source_available"),
            }

    return observations, attempted, rows


def main() -> None:
    p = argparse.ArgumentParser(
        description="Stage 1.12 SM86 legality-aware adaptive iPC autotuner"
    )
    p.add_argument(
        "--kind",
        choices=["prediction", "inference", "weight", "all"],
        default="prediction",
    )
    p.add_argument("--M", type=int, default=64)
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--B", type=int, default=128)
    p.add_argument("--activation", default="relu")
    p.add_argument("--profile", choices=["custom", "mnist64"], default="custom")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--rep", type=int, default=20)
    p.add_argument("--max-configs", type=int, default=80,
                   help="Maximum UNIQUE configs benchmarked per exact shape/family, including cache hits.")
    p.add_argument("--batch-configs", type=int, default=8,
                   help="Number of new configs selected per adaptive round.")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--cache", default="results/autotune_cache.json")
    p.add_argument("--out", default="results/autotune_stage_1_12.json")
    p.add_argument("--temperature-max", type=float, default=78.0)
    p.add_argument("--reject-spill", action="store_true")
    p.add_argument("--max-registers", type=int, default=160)
    a = p.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 6):
        raise RuntimeError("Expected an SM86 CUDA device")

    device = torch.device("cuda")
    guard = ThermalGuard(max_temp_c=a.temperature_max)

    if a.profile == "mnist64":
        shape_map = profile_shapes("mnist64", a.B, a.hidden)
        kinds = [a.kind] if a.kind != "all" else ["prediction", "inference", "weight"]
        shapes = [(k, s) for k in kinds for s in shape_map[k]]
    else:
        shapes = [(a.kind, (a.M, a.K, a.B))]
        if a.kind == "all":
            shapes = [
                ("prediction", (a.M, a.K, a.B)),
                ("inference", (a.M, a.K, a.B)),
                ("weight", (a.M, a.K, a.B)),
            ]

    pool = configs()
    cache_path = Path(a.cache)
    cache = _load_cache(cache_path)

    print("Stage 1.12 legality-aware adaptive autotuner")
    print(
        f"device={torch.cuda.get_device_name(device)} "
        f"capability={torch.cuda.get_device_capability(device)} "
        f"global_pool={len(pool)} cache={cache_path}"
    )

    output_rows: list[dict[str, object]] = []

    for shape_index, (kind, (M, K, B)) in enumerate(shapes, 1):
        legal_pool, rejected = legal_configs(kind, pool)
        observations, attempted, cached_rows = _cached_observations(
            cache, kind, M, K, B, a.activation, legal_pool, device
        )

        # max-configs is a hard UNIQUE-config budget for this exact shape/family.
        # Cache hits (including remembered failures/rejections) count as
        # already-attempted points, so enlarging the budget can only add points.
        budget = min(a.max_configs, len(legal_pool))

        print(
            f"\n[{shape_index}/{len(shapes)}] {kind} M={M} K={K} B={B} "
            f"legal={len(legal_pool)}/{len(pool)} cached={len(observations)} "
            f"rejected={rejected}"
        )

        rows_by_cfg: dict[KernelConfig, dict[str, object]] = dict(cached_rows)
        for cfg, ms in observations:
            key = _cache_key(kind, M, K, B, a.activation, cfg, device)
            crow = cache["results"].get(key, {})
            rows_by_cfg[cfg] = {
                "kind": kind,
                "shape": {"M": M, "K": K, "B": B},
                "activation": a.activation,
                "dtype": "float16",
                "median_ms": ms,
                "config": asdict(cfg),
                "cached": True,
                **{
                    k: crow.get(k)
                    for k in ("registers_estimate", "spill_suspected", "source_available")
                    if k in crow
                },
            }

        while len(attempted) < budget:
            remaining = [cfg for cfg in legal_pool if cfg not in attempted]
            if not remaining:
                break

            batch = select_adaptive_batch(
                legal_pool,
                observations,
                attempted,
                budget,
                max(1, a.batch_configs),
            )
            if not batch:
                break

            print(
                f"  round: measured={len(observations)} "
                f"remaining_budget={budget - len(attempted)} "
                f"batch={len(batch)}"
            )

            for cfg in batch:
                idx = len(observations) + 1
                attempted.add(cfg)
                try:
                    kernel, fn = _make_case(
                        kind, M, K, B, device, a.activation, cfg
                    )
                    inspection = _inspect_kernel(kernel)

                    if a.reject_spill and inspection.get("spill_suspected") is True:
                        print(
                            f"    SKIP {idx:03d}: suspected local-memory use: {cfg}"
                        )
                        key = _cache_key(kind, M, K, B, a.activation, cfg, device)
                        cache["results"][key] = {
                            "status": "reject_spill",
                            "config": asdict(cfg),
                            **inspection,
                        }
                        _save_cache(cache_path, cache)
                        continue

                    regs = inspection.get("registers_estimate")
                    if (
                        a.reject_spill
                        and isinstance(regs, int)
                        and regs > a.max_registers
                    ):
                        print(
                            f"    SKIP {idx:03d}: registers_estimate={regs} "
                            f"> {a.max_registers}: {cfg}"
                        )
                        key = _cache_key(kind, M, K, B, a.activation, cfg, device)
                        cache["results"][key] = {
                            "status": "reject_registers",
                            "config": asdict(cfg),
                            **inspection,
                        }
                        _save_cache(cache_path, cache)
                        continue

                    ms = _time(fn, guard, a.warmup, a.rep)
                    observations.append((cfg, ms))
                    rows_by_cfg[cfg] = {
                        "kind": kind,
                        "shape": {"M": M, "K": K, "B": B},
                        "activation": a.activation,
                        "dtype": "float16",
                        "median_ms": ms,
                        "config": asdict(cfg),
                        "cached": False,
                        **inspection,
                    }

                    key = _cache_key(kind, M, K, B, a.activation, cfg, device)
                    cache["results"][key] = {
                        "status": "ok",
                        "kind": kind,
                        "shape": {"M": M, "K": K, "B": B},
                        "activation": a.activation,
                        "dtype": "float16",
                        "median_ms": ms,
                        "config": asdict(cfg),
                        **inspection,
                    }
                    _save_cache(cache_path, cache)

                    print(
                        f"    {len(observations):03d}/{budget:03d} "
                        f"{ms:8.4f} ms {cfg}"
                    )
                except Exception as exc:
                    # Failed JIT/runtime configurations are remembered as failed
                    # so subsequent runs do not repeatedly compile the same bad point.
                    key = _cache_key(kind, M, K, B, a.activation, cfg, device)
                    cache["results"][key] = {
                        "status": "error",
                        "kind": kind,
                        "shape": {"M": M, "K": K, "B": B},
                        "activation": a.activation,
                        "dtype": "float16",
                        "config": asdict(cfg),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    _save_cache(cache_path, cache)
                    print(
                        f"    SKIP {idx:03d}: {cfg}: "
                        f"{type(exc).__name__}: {exc}"
                    )

            if len(attempted) >= budget:
                break

        rows = sorted(rows_by_cfg.values(), key=lambda r: float(r["median_ms"]))
        if not rows:
            raise RuntimeError(
                f"No valid configurations for {kind} M={M} K={K} B={B}"
            )

        output_rows.append(
            {
                "kind": kind,
                "shape": {"M": M, "K": K, "B": B},
                "activation": a.activation,
                "dtype": "float16",
                "candidate_pool": len(pool),
                "legal_pool": len(legal_pool),
                "rejected": rejected,
                "tested": len(attempted),
                "successful": len(rows),
                "budget": budget,
                "best": rows[: a.topk],
            }
        )
        best = rows[0]
        print(
            f"BEST {kind} M={M} K={K} B={B}: "
            f"{float(best['median_ms']):.4f} ms {best['config']}"
        )

    entries = []
    for result in output_rows:
        best = result["best"][0]
        entries.append(
            {
                "kind": result["kind"],
                "shape": result["shape"],
                "activation": result["activation"],
                "dtype": result["dtype"],
                "median_ms": best["median_ms"],
                "registers_estimate": best.get("registers_estimate"),
                "spill_suspected": best.get("spill_suspected"),
                "config": best["config"],
            }
        )

    payload = {
        "schema_version": 2,
        "stage": "1.12",
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "tilelang_version": _tilelang_version(),
        "search": {
            "global_candidate_pool": len(pool),
            "warmup": a.warmup,
            "rep": a.rep,
            "max_configs": a.max_configs,
            "budget_semantics": "unique legal configurations per exact shape/family; cache-aware and monotonic",
            "batch_configs": a.batch_configs,
            "cache": str(cache_path),
            "reject_spill": a.reject_spill,
            "max_registers": a.max_registers,
            "selection": "deterministic maximin + distance-weighted kNN lower-confidence-bound",
        },
        "results": output_rows,
        "entries": entries,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
