from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .tilelang_kernels import KernelConfig


def _dtype_name(dtype: Any) -> str:
    return str(dtype).split(".")[-1]


def _dims_key(dims: tuple[int, ...] | list[int]) -> str:
    return "x".join(str(int(d)) for d in dims)


def make_kernel_key(kind: str, M: int, K: int, B: int, activation: str, dtype: Any) -> str:
    return json.dumps(
        {
            "kind": str(kind),
            "M": int(M),
            "K": int(K),
            "B": int(B),
            "activation": str(activation),
            "dtype": _dtype_name(dtype),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def kernel_signature(rows: list[Mapping[str, Any]]) -> str:
    """Stable fingerprint of the exact P/I/W kernel table used by execution tuning."""
    normalized: list[dict[str, Any]] = []
    for row in rows:
        shape = row.get("shape") or {}
        normalized.append(
            {
                "kind": str(row["kind"]),
                "M": int(shape["M"]),
                "K": int(shape["K"]),
                "B": int(shape["B"]),
                "activation": str(row.get("activation", "relu")),
                "dtype": str(row.get("dtype", "float16")),
                "config": dict(row["config"]),
            }
        )
    normalized.sort(key=lambda r: (r["kind"], r["M"], r["K"], r["B"], r["activation"], r["dtype"]))
    raw = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


class HierarchicalPolicy:
    """Runtime reader for the Stage-2 kernel+execution autotuning artifact.

    The policy is deliberately read-only during training. Kernel choices are exact-key
    lookups; execution choices additionally require the kernel-table fingerprint that was
    actually used when the execution candidate was measured.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            self.data: dict[str, Any] = {"version": 1, "kernel_entries": [], "execution_entries": []}
        else:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError(f"Invalid hierarchical policy: {self.path}")
            self.data = data

        self._kernels: dict[str, KernelConfig] = {}
        for row in self.data.get("kernel_entries", []):
            if not isinstance(row, dict) or "config" not in row or "kind" not in row:
                continue
            shape = row.get("shape") or {}
            try:
                key = make_kernel_key(
                    str(row["kind"]), int(shape["M"]), int(shape["K"]), int(shape["B"]),
                    str(row.get("activation", "relu")), str(row.get("dtype", "float16")),
                )
                self._kernels[key] = KernelConfig(**dict(row["config"]))
            except Exception:
                continue

        self._kernel_profiles: dict[str, dict[str, KernelConfig]] = {}
        self._kernel_profile_meta: dict[str, dict[str, Any]] = {}
        for profile in self.data.get("kernel_profiles", []):
            if not isinstance(profile, dict):
                continue
            try:
                dims = tuple(int(x) for x in profile["dims"])
                batch = int(profile["batch"])
                activation = str(profile.get("activation", "relu"))
                dtype = str(profile.get("dtype", "float16"))
                profile_key = self.make_kernel_profile_key(
                    dims=dims, batch=batch, activation=activation, dtype=dtype
                )
                mapping: dict[str, KernelConfig] = {}
                for row in profile.get("kernel_entries", []):
                    if not isinstance(row, dict) or "config" not in row or "kind" not in row:
                        continue
                    shape = row.get("shape") or {}
                    key = make_kernel_key(
                        str(row["kind"]), int(shape["M"]), int(shape["K"]), int(shape["B"]),
                        str(row.get("activation", activation)), str(row.get("dtype", dtype)),
                    )
                    mapping[key] = KernelConfig(**dict(row["config"]))
                if mapping:
                    self._kernel_profiles[profile_key] = mapping
                    self._kernel_profile_meta[profile_key] = profile
            except Exception:
                continue

        self._execution: dict[str, dict[str, Any]] = {}
        for row in self.data.get("execution_entries", []):
            if not isinstance(row, dict):
                continue
            try:
                key = self.make_execution_key(
                    dims=tuple(int(x) for x in row["dims"]),
                    batch=int(row["batch"]),
                    activation=str(row.get("activation", "relu")),
                    dtype=str(row.get("dtype", "float16")),
                    kernel_sig=str(row["kernel_signature"]),
                )
                self._execution[key] = row
            except Exception:
                continue

    @staticmethod
    def make_kernel_profile_key(
        *, dims: tuple[int, ...], batch: int, activation: str, dtype: Any
    ) -> str:
        return json.dumps(
            {
                "dims": _dims_key(dims),
                "batch": int(batch),
                "activation": str(activation),
                "dtype": _dtype_name(dtype),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def make_execution_key(
        *, dims: tuple[int, ...], batch: int, activation: str, dtype: Any, kernel_sig: str
    ) -> str:
        return json.dumps(
            {
                "dims": _dims_key(dims),
                "batch": int(batch),
                "activation": str(activation),
                "dtype": _dtype_name(dtype),
                "kernel_signature": str(kernel_sig),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def exists(self) -> bool:
        return self.path.exists()

    def lookup_kernel(self, *, kind: str, M: int, K: int, B: int, activation: str, dtype: Any) -> KernelConfig | None:
        return self._kernels.get(make_kernel_key(kind, M, K, B, activation, dtype))

    def lookup_profile_kernel(
        self, *, dims: tuple[int, ...], batch: int, kind: str, M: int, K: int,
        activation: str, dtype: Any
    ) -> KernelConfig | None:
        profile_key = self.make_kernel_profile_key(
            dims=dims, batch=batch, activation=activation, dtype=dtype
        )
        mapping = self._kernel_profiles.get(profile_key)
        if mapping is None:
            return None
        return mapping.get(make_kernel_key(kind, M, K, int(batch), activation, dtype))

    def profile_kernel_signature(
        self, *, dims: tuple[int, ...], batch: int, activation: str, dtype: Any
    ) -> str | None:
        profile_key = self.make_kernel_profile_key(
            dims=dims, batch=batch, activation=activation, dtype=dtype
        )
        meta = self._kernel_profile_meta.get(profile_key)
        if isinstance(meta, dict):
            value = meta.get("kernel_signature")
            return str(value) if value is not None else None
        return None

    def lookup_execution(
        self, *, dims: tuple[int, ...], batch: int, activation: str, dtype: Any, kernel_sig: str
    ) -> dict[str, Any] | None:
        return self._execution.get(
            self.make_execution_key(
                dims=dims, batch=batch, activation=activation, dtype=dtype, kernel_sig=kernel_sig
            )
        )

    def selected_grid_z(
        self, *, dims: tuple[int, ...], batch: int, activation: str, dtype: Any, kernel_sig: str, default: int
    ) -> int:
        row = self.lookup_execution(
            dims=dims, batch=batch, activation=activation, dtype=dtype, kernel_sig=kernel_sig
        )
        if not isinstance(row, dict):
            return int(default)
        best = row.get("best") or {}
        try:
            return int(best.get("grid_z_max_layers", default))
        except Exception:
            return int(default)

    def selected_recompute_activation(
        self, *, dims: tuple[int, ...], batch: int, activation: str, dtype: Any, kernel_sig: str, default: bool
    ) -> bool:
        row = self.lookup_execution(
            dims=dims, batch=batch, activation=activation, dtype=dtype, kernel_sig=kernel_sig
        )
        if not isinstance(row, dict):
            return bool(default)
        best = row.get("best") or {}
        return bool(best.get("recompute_activation", default))

    def selected_use_graph(
        self, *, dims: tuple[int, ...], batch: int, activation: str, dtype: Any, kernel_sig: str, default: bool
    ) -> bool:
        row = self.lookup_execution(
            dims=dims, batch=batch, activation=activation, dtype=dtype, kernel_sig=kernel_sig
        )
        if not isinstance(row, dict):
            return bool(default)
        best = row.get("best") or {}
        return bool(best.get("use_cuda_graph", default))


def load_hierarchical_policy(path: str | Path) -> HierarchicalPolicy | None:
    p = Path(path)
    if not p.exists():
        return None
    return HierarchicalPolicy(p)
