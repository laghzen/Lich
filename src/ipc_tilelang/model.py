from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


FEATURE_FIELDS = (
    "block_m", "block_n", "block_k", "threads",
    "num_stages", "swizzle", "swizzle_panel", "shared_swizzle",
)


@dataclass(frozen=True)
class Prediction:
    mean: float
    std: float
    lcb: float


def config_vector(cfg: Any, *, M: int, K: int, B: int, kind: str) -> np.ndarray:
    bm = float(cfg.block_m)
    bn = float(cfg.block_n)
    bk = float(cfg.block_k)
    th = float(cfg.threads)
    stages = float(cfg.num_stages)
    tile_out = bm * bn
    gemm_n = bk if kind == "weight" else bn
    cta_m = math.ceil(M / max(1.0, bm))
    cta_n = math.ceil(B / max(1.0, bn))
    smem_units = bm * bk + bk * bn
    if kind == "weight":
        smem_units = bm * bn + 2.0 * bk * bn
    return np.asarray([
        math.log2(max(1.0, bm)), math.log2(max(1.0, bn)), math.log2(max(1.0, bk)),
        th / 32.0, stages,
        1.0 if bool(cfg.swizzle) else 0.0,
        math.log2(max(1.0, float(cfg.swizzle_panel))) if bool(cfg.swizzle) else 0.0,
        1.0 if bool(cfg.shared_swizzle) else 0.0,
        min(8.0, M / max(1.0, bm)), min(8.0, B / max(1.0, bn)), min(8.0, K / max(1.0, bk)),
        cta_m * cta_n, M % int(bm), B % int(bn), K % int(bk),
        tile_out / 8192.0, gemm_n / 128.0, smem_units / 65536.0,
    ], dtype=np.float64)


class FactorizedGaussianSurrogate:
    """Fast nearest-neighbour surrogate for a small discrete GPU search space.

    The predictor is intentionally local: it never extrapolates an unseen
    categorical combination to a latency far below the measured range. This is
    important for TileLang/SM86 because interactions such as tile shape x threads
    can be jagged. The search layer supplies exploration through distance-aware
    uncertainty and explicit global scouts.
    """

    def __init__(self, *, noise_floor: float = 0.025, ridge: float = 3e-3, max_neighbors: int = 10):
        self.noise_floor = float(noise_floor)
        self.ridge = float(ridge)  # kept for API compatibility
        self.max_neighbors = int(max(4, max_neighbors))
        self._fit = False
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None
        self._bias = -4.0
        self._y_lo = -5.0
        self._y_hi = -2.0

    @staticmethod
    def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mu = np.mean(X, axis=0)
        sd = np.std(X, axis=0)
        sd[sd < 1e-9] = 1.0
        return (X - mu) / sd, mu, sd

    @staticmethod
    def _feature_distance(X: np.ndarray, x: np.ndarray) -> np.ndarray:
        # Give tile geometry more influence than boolean switches. The exact
        # parameters have deliberately smooth logarithmic representations.
        w = np.asarray([
            1.45, 1.45, 1.15, 0.80, 0.55, 0.35, 0.20, 0.25,
            0.90, 0.90, 0.75, 0.50, 0.40, 0.40, 0.40, 0.35, 0.30, 0.35,
        ], dtype=np.float64)
        return np.sqrt(np.sum(((X - x) ** 2) * w[None, :], axis=1))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if len(X) == 0:
            self._fit = False
            return
        Xn, mu, sd = self._standardize(X)
        self._X = Xn
        self._y = y
        self._mu = mu
        self._sd = sd
        self._bias = float(np.median(y))
        # Allow modest improvement extrapolation but never the wild orders of
        # magnitude that the previous residual-GP could create.
        self._y_lo = float(np.min(y) - 0.07)
        self._y_hi = float(np.max(y) + 0.12)
        self._fit = True

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float64)
        if not self._fit or self._X is None or self._y is None:
            return np.full(len(X), self._bias), np.full(len(X), 0.35)
        Xn = (X - self._mu) / self._sd
        pred = np.empty(len(Xn), dtype=np.float64)
        log_std = np.empty(len(Xn), dtype=np.float64)
        for i, x in enumerate(Xn):
            d = self._feature_distance(self._X, x)
            k = min(self.max_neighbors, len(d))
            idx = np.argpartition(d, k - 1)[:k]
            dk = d[idx]
            scale = max(float(np.median(dk)), 0.55)
            w = np.exp(-0.5 * (dk / scale) ** 2)
            ws = float(np.sum(w))
            local_mean = float(np.sum(w * self._y[idx]) / max(ws, 1e-12))
            mad = float(np.median(np.abs(self._y[idx] - local_mean))) if len(idx) else 0.0
            robust = 1.4826 * mad / max(abs(local_mean), 1e-6)
            nearest = float(np.min(dk)) if len(dk) else 3.0
            epistemic = 0.045 + 0.105 * math.tanh(nearest / 1.5)
            pred[i] = local_mean
            log_std[i] = min(0.42, max(self.noise_floor, robust + epistemic))
        pred = np.clip(pred, self._y_lo, self._y_hi)
        return pred, log_std

    def predict_latency(self, X: np.ndarray, *, beta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        log_mean, log_std = self.predict(X)
        log_std = np.clip(log_std, self.noise_floor, 0.42)
        mean = np.exp(log_mean)
        std = mean * np.sqrt(np.maximum(np.exp(np.minimum(log_std ** 2, 0.75)) - 1.0, 1e-12))
        lcb = np.exp(log_mean - float(beta) * log_std)
        return mean, std, lcb

