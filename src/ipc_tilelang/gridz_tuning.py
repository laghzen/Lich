from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _dtype_name(dtype: Any) -> str:
    return str(dtype).split(".")[-1]


def _dims_key(dims: tuple[int, ...]) -> str:
    return "x".join(str(int(d)) for d in dims)


def make_policy_key(
    *,
    device_name: str,
    capability: tuple[int, int],
    dims: tuple[int, ...],
    batch: int,
    activation: str,
    dtype: Any,
) -> str:
    return json.dumps(
        {
            "device": device_name,
            "capability": list(capability),
            "dims": _dims_key(dims),
            "batch": int(batch),
            "activation": activation,
            "dtype": _dtype_name(dtype),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def load_policy(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"version": 1, "entries": {}}
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict) or not isinstance(obj.get("entries", {}), dict):
        raise ValueError(f"Invalid grid.z policy file: {p}")
    return obj


def lookup_grid_z_cap(
    path: str | Path,
    *,
    device_name: str,
    capability: tuple[int, int],
    dims: tuple[int, ...],
    batch: int,
    activation: str,
    dtype: Any,
    default: int = 2,
) -> int:
    policy = load_policy(path)
    key = make_policy_key(
        device_name=device_name,
        capability=capability,
        dims=dims,
        batch=batch,
        activation=activation,
        dtype=dtype,
    )
    row = policy["entries"].get(key)
    if not isinstance(row, dict):
        return int(default)
    return int(row.get("grid_z_max_layers", default))
