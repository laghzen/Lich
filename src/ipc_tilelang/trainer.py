from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .activations import ActivationSpec, get_activation
from .reference import one_hot
from .tilelang_kernels import (
    KernelConfig,
    build_inference_update,
    build_prediction_error,
    build_weight_update,
)
from .tuning import TuningTable


@dataclass(frozen=True)
class IPCConfig:
    dims: tuple[int, ...]
    alpha: float = 1e-4
    gamma: float = 0.5
    activation: str = "relu"
    dtype: torch.dtype = torch.float16
    kernel: KernelConfig = KernelConfig()
    recompute_activation: bool = True
    tuning_table: TuningTable | None = None


class TileLangIPC:
    def __init__(self, cfg: IPCConfig, *, device: torch.device, seed: int = 0):
        if device.type != "cuda":
            raise ValueError("TileLangIPC requires a CUDA device")
        if torch.cuda.get_device_capability(device) != (8, 6):
            raise ValueError(
                f"This stage is intentionally pinned to RTX 3060/SM86; got {torch.cuda.get_device_capability(device)}"
            )
        self.cfg = cfg
        self.device = device
        self.act = get_activation(cfg.activation)
        g = torch.Generator(device=device); g.manual_seed(seed)
        self.w: list[torch.Tensor] = []
        for n_l, n_u in zip(cfg.dims[:-1], cfg.dims[1:]):
            w = torch.empty((n_l, n_u), device=device, dtype=cfg.dtype)
            torch.nn.init.kaiming_uniform_(w, a=5 ** 0.5, generator=g)
            self.w.append(w)
        self.x: list[torch.Tensor] | None = None
        self.e: list[torch.Tensor] | None = None
        self.a: list[torch.Tensor | None] | None = None

    @property
    def L(self) -> int:
        return len(self.w)

    def _tuned_or_default(self, kind: str, M: int, K: int, B: int) -> KernelConfig:
        if self.cfg.tuning_table is not None:
            tuned = self.cfg.tuning_table.lookup(
                kind=kind,
                M=M,
                K=K,
                B=B,
                activation=self.cfg.activation,
                dtype=str(self.cfg.dtype).split(".")[-1],
            )
            if tuned is not None:
                return tuned
        return self.cfg.kernel

    def initialize_batch(self, x_input: torch.Tensor, y_target: torch.Tensor) -> None:
        """Reset endpoints and initialize hidden states by a forward prediction.
        Buffers are allocated once per batch-size shape and reused afterwards.
        """
        B = x_input.shape[1]
        if x_input.shape[0] != self.cfg.dims[-1] or y_target.shape[0] != self.cfg.dims[0]:
            raise ValueError("Batch dimensions do not match network dims")
        if self.x is None or self.e is None or self.x[0].shape[1] != B:
            self.x = [torch.empty((d, B), device=self.device, dtype=self.cfg.dtype) for d in self.cfg.dims]
            self.e = [torch.empty_like(v) for v in self.x]
            self.a = [None] + [torch.empty_like(v) for v in self.x[1:]] if not self.cfg.recompute_activation else None
        self.x[-1].copy_(x_input)
        if self.a is not None:
            self.a[-1].copy_(self.act.forward(self.x[-1]))
        for l in reversed(range(self.L)):
            pred = self.w[l] @ self.act.forward(self.x[l + 1])
            if l > 0:
                self.x[l].copy_(pred)
        self.x[0].copy_(y_target)

    def _get_prediction_kernel(self, l: int):
        return build_prediction_error(
            self.cfg.dims[l],
            self.cfg.dims[l + 1],
            self.x[l].shape[1],
            self.cfg.activation,
            str(self.cfg.dtype).split(".")[-1],
            self._tuned_or_default(
                "prediction", self.cfg.dims[l], self.cfg.dims[l + 1], self.x[l].shape[1]
            ),
        )

    def _get_inference_kernel(self, l: int):
        return build_inference_update(
            self.cfg.dims[l],
            self.cfg.dims[l - 1],
            self.x[l].shape[1],
            self.cfg.activation,
            str(self.cfg.dtype).split(".")[-1],
            self._tuned_or_default(
                "inference", self.cfg.dims[l], self.cfg.dims[l - 1], self.x[l].shape[1]
            ),
            self.cfg.gamma,
            not self.cfg.recompute_activation,
        )

    def _get_weight_kernel(self, l: int):
        return build_weight_update(
            self.cfg.dims[l],
            self.cfg.dims[l + 1],
            self.x[l].shape[1],
            self.cfg.activation,
            str(self.cfg.dtype).split(".")[-1],
            self._tuned_or_default(
                "weight", self.cfg.dims[l], self.cfg.dims[l + 1], self.x[l].shape[1]
            ),
            self.cfg.alpha,
            self.cfg.recompute_activation,
        )

    @torch.no_grad()
    def step_initialized(self, *, collect_metrics: bool = False) -> dict[str, float]:
        if self.x is None or self.e is None:
            raise RuntimeError("Call initialize_batch before step_initialized")
        for l in range(self.L):
            self._get_prediction_kernel(l)(self.x[l + 1], self.w[l], self.x[l], self.e[l])
        for l in range(1, self.L):
            if self.a is None:
                self._get_inference_kernel(l)(self.x[l], self.e[l], self.w[l - 1], self.e[l - 1])
            else:
                assert self.a[l] is not None
                self._get_inference_kernel(l)(self.x[l], self.e[l], self.w[l - 1], self.e[l - 1], self.a[l])
        for l in range(self.L):
            if self.a is None:
                self._get_weight_kernel(l)(self.w[l], self.e[l], self.x[l + 1])
            else:
                assert self.a[l + 1] is not None
                self._get_weight_kernel(l)(self.w[l], self.e[l], self.a[l + 1])
        if collect_metrics:
            assert self.e is not None
            return {"max_abs_error": float(max(v.float().abs().max().item() for v in self.e[:-1]))}
        return {}

    @torch.no_grad()
    def step(self, x_input: torch.Tensor, y_target: torch.Tensor) -> dict[str, float]:
        self.initialize_batch(x_input, y_target)
        return self.step_initialized(collect_metrics=True)

    def precompile(self) -> None:
        """Force JIT compilation of the current shape without changing model state.
        TileLang's eager JIT object exposes kernel source generation; calling it
        is sufficient to force construction/compilation while avoiding a hidden
        extra optimization step before CUDA-Graph capture.  A runtime fallback
        executes one warm-up step for older TileLang builds, then restores all
        mutable buffers.
        """
        if self.x is None or self.e is None:
            raise RuntimeError("Call initialize_batch before precompile")
        kernels = []
        for l in range(self.L):
            kernels.append(self._get_prediction_kernel(l))
        for l in range(1, self.L):
            kernels.append(self._get_inference_kernel(l))
        for l in range(self.L):
            kernels.append(self._get_weight_kernel(l))

        if all(hasattr(k, "get_kernel_source") for k in kernels):
            for k in kernels:
                _ = k.get_kernel_source()
            torch.cuda.synchronize()
            return
        w_backup = [w.clone() for w in self.w]
        x_backup = [x.clone() for x in self.x]
        e_backup = [e.clone() for e in self.e]
        self.step_initialized(collect_metrics=False)
        torch.cuda.synchronize()
        for dst, src in zip(self.w, w_backup):
            dst.copy_(src)
        for dst, src in zip(self.x, x_backup):
            dst.copy_(src)
        for dst, src in zip(self.e, e_backup):
            dst.copy_(src)
        torch.cuda.synchronize()

    def capture_graph(self, steps: int):
        if self.x is None or self.e is None:
            raise RuntimeError("Call initialize_batch before capture_graph")
        from .cuda_graph import IPCGraphRunner
        runner = IPCGraphRunner(self, steps)
        runner.capture()
        return runner

    @torch.no_grad()
    def predict(self, x_input: torch.Tensor) -> torch.Tensor:
        B = x_input.shape[1]
        h = x_input
        for l in reversed(range(self.L)):
            h = self.w[l] @ self.act.forward(h)
        return h


def prepare_batch(images: torch.Tensor, labels: torch.Tensor, *, device: torch.device, dtype: torch.dtype, num_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = images.flatten(1).T.contiguous().to(device=device, dtype=dtype, non_blocking=True)
    y = one_hot(labels.to(device, non_blocking=True), num_classes, dtype=dtype)
    return x, y
