from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


CACHE_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class ProblemKey:
    kind: str
    M: int
    K: int
    B: int
    activation: str
    dtype: str
    device_fingerprint: str
    code_fingerprint: str
    # Measurement validity is tied to hardware + kernel generator, not to the
    # autotuner implementation. Keep this stable so autotuner upgrades reuse
    # already-valid GPU observations.
    search_space_version: str = "sm86-v2"

    def payload(self) -> dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        raw = json.dumps(self.payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Measurement:
    status: str
    latency_ms: float | None = None
    correctness_ok: bool | None = None
    fidelity: int = 0
    temperature_c: float | None = None
    power_w: float | None = None
    clock_mhz: float | None = None
    registers_per_thread: int | None = None
    spill_bytes: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    origin: str = "gpu"
    created_at: float = field(default_factory=time.time)

    @classmethod
    def measured(
        cls,
        latency_ms: float,
        *,
        correctness_ok: bool = True,
        fidelity: int = 0,
        temperature_c: float | None = None,
        power_w: float | None = None,
        clock_mhz: float | None = None,
        registers_per_thread: int | None = None,
        spill_bytes: int | None = None,
        origin: str = "gpu",
    ) -> "Measurement":
        return cls(
            status="ok",
            latency_ms=float(latency_ms),
            correctness_ok=bool(correctness_ok),
            fidelity=int(fidelity),
            temperature_c=temperature_c,
            power_w=power_w,
            clock_mhz=clock_mhz,
            registers_per_thread=registers_per_thread,
            spill_bytes=spill_bytes,
            origin=str(origin),
        )

    @classmethod
    def failed(
        cls,
        error_type: str,
        error_message: str,
        *,
        fidelity: int = 0,
        correctness_ok: bool | None = None,
        origin: str = "gpu",
    ) -> "Measurement":
        return cls(
            status="failed",
            correctness_ok=correctness_ok,
            fidelity=int(fidelity),
            error_type=str(error_type),
            error_message=str(error_message)[:8000],
            origin=str(origin),
        )

    @property
    def successful(self) -> bool:
        return (
            self.status == "ok"
            and self.latency_ms is not None
            and self.correctness_ok is not False
            and self.latency_ms > 0.0
        )


@dataclass(frozen=True)
class Fidelity:
    level: int
    warmup: int
    rep: int
    label: str


@dataclass(frozen=True)
class SearchConfig:
    # There is deliberately NO max_evals/max_configs field here. The finite legal
    # space itself determines how many GPU evaluations are required. The tuner
    # stops only when every unresolved point is evaluated or statistically pruned.
    seed_evals: int = 10
    batch_size: int = 6
    verification_topk: int = 6
    verification_reps: int = 40
    verification_warmup: int = 10
    probe_warmup: int = 2
    probe_reps: int = 5
    min_model_points: int = 8
    beta: float = 3.5
    prune_margin: float = 0.0
    prune_stability_rounds: int = 4
    min_region_observations: int = 2
    min_leaf_size: int = 2
    diversity_weight: float = 0.08
    novelty_weight: float = 0.12
    robust_noise_floor: float = 0.015
    random_seed: int = 0
    # Optional emergency fuse. None means unlimited and is the default.
    max_rounds: int | None = None

    def fidelities(self) -> tuple[Fidelity, Fidelity]:
        return (
            Fidelity(0, max(0, self.probe_warmup), max(2, self.probe_reps), "probe"),
            Fidelity(1, max(0, self.verification_warmup), max(3, self.verification_reps), "verify"),
        )


@dataclass
class SearchResult:
    best_config: Any | None
    best_latency_ms: float | None
    space_total: int
    legal_total: int
    static_invalid: int
    attempted: int
    successful: int
    compile_failures: int
    correctness_failures: int
    model_pruned: int
    region_pruned: int
    untested: int
    adaptive_search: bool
    hierarchical_pruning: bool
    stratified_acquisition: bool
    search_engine: str
    stopped_by_global_bound: bool
    rounds: int
    metadata: dict[str, Any] = field(default_factory=dict)


class SQLiteCache:
    """Persistent success/failure memory for one fixed hardware/code fingerprint."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), timeout=60.0, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS measurements (
                cache_key TEXT PRIMARY KEY,
                problem_digest TEXT NOT NULL,
                problem_json TEXT NOT NULL,
                config_json TEXT NOT NULL,
                measurement_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_measurements_problem ON measurements(problem_digest)"
        )
        self._conn.commit()

    @staticmethod
    def _cfg_digest(problem: ProblemKey, cfg: Any) -> str:
        cfg_json = json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256((problem.digest() + "|" + cfg_json).encode("utf-8")).hexdigest()

    def get(self, problem: ProblemKey, cfg: Any) -> Measurement | None:
        key = self._cfg_digest(problem, cfg)
        with self._lock:
            row = self._conn.execute(
                "SELECT measurement_json FROM measurements WHERE cache_key=?", (key,)
            ).fetchone()
        if row is None:
            return None
        data = json.loads(row[0])
        # Backward compatibility with v2 cache records that predate origin.
        data.setdefault("origin", "gpu")
        return Measurement(**data)

    def put(self, problem: ProblemKey, cfg: Any, measurement: Measurement) -> None:
        key = self._cfg_digest(problem, cfg)
        problem_json = json.dumps(problem.payload(), sort_keys=True, separators=(",", ":"))
        config_json = json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":"))
        measurement_json = json.dumps(asdict(measurement), sort_keys=True, separators=(",", ":"))
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO measurements(cache_key, problem_digest, problem_json, config_json, measurement_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    measurement_json=excluded.measurement_json,
                    updated_at=excluded.updated_at
                """,
                (key, problem.digest(), problem_json, config_json, measurement_json, time.time()),
            )
            self._conn.commit()

    def iter_problem(self, problem: ProblemKey) -> Iterable[tuple[dict[str, Any], Measurement]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT config_json, measurement_json FROM measurements WHERE problem_digest=?",
                (problem.digest(),),
            ).fetchall()
        for cfg_json, meas_json in rows:
            data = json.loads(meas_json)
            data.setdefault("origin", "gpu")
            yield json.loads(cfg_json), Measurement(**data)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "SQLiteCache":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
