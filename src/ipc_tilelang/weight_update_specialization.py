from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def _dtype_name(dtype: Any) -> str:
    return str(dtype).split(".")[-1]


@dataclass(frozen=True)
class WeightPolicyKey:
    M: int
    K: int
    B: int
    activation: str = "relu"
    dtype: str = "float16"


class SmallBWeightPolicy:
    """Read-only exact-shape policy produced by the target-GPU crossover benchmark."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with self.path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid weight-update policy: {self.path}")
        self.data = data
        self.entries: dict[tuple[int, int, int, int, str, str], str] = {}
        for row in data.get("entries", []):
            if not isinstance(row, dict):
                continue
            try:
                key = (
                    int(row["M"]), int(row["K"]), int(row["B"]),
                    int(row.get("count", 1)),
                    str(row.get("activation", "relu")),
                    str(row.get("dtype", "float16")),
                )
                backend = str(row.get("backend", "gemm")).lower()
                if backend in {"gemm", "outer"}:
                    self.entries[key] = backend
            except (KeyError, TypeError, ValueError):
                continue

    def backend(
        self, *, M: int, K: int, B: int, activation: str, dtype: Any, count: int = 1
    ) -> str:
        return self.entries.get(
            (
                int(M), int(K), int(B), int(count),
                str(activation), _dtype_name(dtype),
            ),
            "gemm",
        )


class SmallBWeightUpdater:
    """Small-B outer-product update using only eager CUDA tensor operators.

    There is deliberately no PyTorch graph compiler here.  The implementation uses
    ``torch.bmm`` to form one outer product per batch item and then reduces the B
    products in FP32.  This stays on the normal CUDA execution path and does not
    require any separate kernel compiler package.
    """

    def __init__(self, *, alpha: float):
        self.alpha = float(alpha)
        self._single_cache: dict[tuple, object] = {}
        self._group_cache: dict[tuple, object] = {}

    @staticmethod
    def _supported_activation(name: str) -> bool:
        return str(name).lower() in {"relu", "identity", "linear"}

    @staticmethod
    def _activation(x: torch.Tensor, name: str) -> torch.Tensor:
        n = str(name).lower()
        if n == "relu":
            return torch.relu(x)
        if n in {"identity", "linear"}:
            return x
        raise ValueError(f"Small-B outer update does not support activation={name!r}")

    def _make_single_eager(self, *, M: int, K: int, B: int, activation: str):
        if not self._supported_activation(activation):
            return None
        alpha = self.alpha

        @torch.no_grad()
        def eager(w: torch.Tensor, e: torch.Tensor, src: torch.Tensor) -> None:
            # e[M,B], src[K,B] -> sum_b e[:,b] outer f(src[:,b]) [M,K].
            # Keep the reduction in FP32 to match the TileLang reference path.
            e_b = e.float().transpose(0, 1).unsqueeze(2)       # [B,M,1]
            a_b = self._activation(src.float(), activation).transpose(0, 1).unsqueeze(1)  # [B,1,K]
            grad = torch.bmm(e_b, a_b).sum(dim=0)               # [M,K]
            w.add_((-alpha * grad).to(dtype=w.dtype))

        return eager

    def _make_group_eager(self, *, count: int, M: int, K: int, B: int, activation: str):
        # Grouped outer-product updates are intentionally not enabled yet.  They
        # require an independent benchmark because their temporary BxMxK tensor
        # and launch pattern differ from the singleton case.  Trainer therefore
        # keeps grouped layers on the measured TileLang GEMM path.
        del count, M, K, B, activation
        return None

    def make_singleton(self, *, M: int, K: int, B: int, activation: str, dtype: Any):
        key = (int(M), int(K), int(B), str(activation), _dtype_name(dtype), "single")
        if key not in self._single_cache:
            self._single_cache[key] = self._make_single_eager(
                M=M, K=K, B=B, activation=activation
            )
        return self._single_cache[key]

    def make_grouped(self, *, count: int, M: int, K: int, B: int, activation: str, dtype: Any):
        key = (int(count), int(M), int(K), int(B), str(activation), _dtype_name(dtype), "group")
        if key not in self._group_cache:
            self._group_cache[key] = self._make_group_eager(
                count=count, M=M, K=K, B=B, activation=activation
            )
        return self._group_cache[key]
