from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

from .tilelang_kernels import KernelConfig

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TuningKey:
    kind: str
    M: int
    K: int
    B: int
    activation: str
    dtype: str


@dataclass(frozen=True)
class TuningEntry:
    key: TuningKey
    config: KernelConfig
    median_ms: float | None = None
    spill_suspected: bool | None = None


@dataclass(frozen=True)
class TuningTable:
    entries: tuple[TuningEntry, ...]

    @classmethod
    def from_entries(cls, entries: Iterable[TuningEntry]) -> "TuningTable":
        return cls(tuple(entries))

    @classmethod
    def from_json(cls, path: str | Path) -> "TuningTable":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported tuning schema: {payload.get('schema_version')!r}; "
                f"expected {SCHEMA_VERSION}"
            )
        entries: list[TuningEntry] = []
        for row in payload.get("entries", []):
            shape = row["shape"]
            cfg_data = row["config"]
            cfg = KernelConfig(**cfg_data)
            key = TuningKey(
                kind=row["kind"],
                M=int(shape["M"]),
                K=int(shape["K"]),
                B=int(shape["B"]),
                activation=str(row.get("activation", "relu")),
                dtype=str(row.get("dtype", "float16")),
            )
            entries.append(
                TuningEntry(
                    key=key,
                    config=cfg,
                    median_ms=(None if row.get("median_ms") is None else float(row["median_ms"])),
                    spill_suspected=row.get("spill_suspected"),
                )
            )
        return cls(tuple(entries))

    def lookup(
        self,
        *,
        kind: str,
        M: int,
        K: int,
        B: int,
        activation: str,
        dtype: str,
    ) -> KernelConfig | None:
        key = TuningKey(kind, int(M), int(K), int(B), activation, dtype)
        for entry in self.entries:
            if entry.key == key:
                return entry.config
        return None

    def to_json(self, path: str | Path, *, device: str | None = None, capability: tuple[int, int] | None = None) -> None:
        rows = []
        for entry in self.entries:
            rows.append(
                {
                    "kind": entry.key.kind,
                    "shape": {"M": entry.key.M, "K": entry.key.K, "B": entry.key.B},
                    "activation": entry.key.activation,
                    "dtype": entry.key.dtype,
                    "median_ms": entry.median_ms,
                    "spill_suspected": entry.spill_suspected,
                    "config": entry.config.__dict__,
                }
            )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "device": device,
            "capability": list(capability) if capability is not None else None,
            "entries": rows,
        }
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
