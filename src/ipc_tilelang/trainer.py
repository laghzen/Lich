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
from .layer_batching import (
    LayerBatchGroup,
    build_inference_update_batched,
    build_prediction_error_batched,
    build_weight_update_batched,
    find_internal_square_groups,
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
    use_grid_z: bool = True
    grid_z_max_layers: int = 2
    grid_z_auto: bool = True
    grid_z_policy_path: str = "results/gridz_policy.json"


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
        grid_z_cap = int(cfg.grid_z_max_layers)
        self._grid_z_cap = max(0, grid_z_cap)
        self._layer_groups: tuple[LayerBatchGroup, ...] = (
            find_internal_square_groups(cfg.dims, max_group_count=self._grid_z_cap) if cfg.use_grid_z and self._grid_z_cap >= 2 else ()
        )
        self._edge_to_group: dict[int, int] = {}
        for gi, group in enumerate(self._layer_groups):
            for l in group.edge_indices:
                self._edge_to_group[l] = gi

        g = torch.Generator(device=device)
        g.manual_seed(seed)
        self.w: list[torch.Tensor] = [None] * (len(cfg.dims) - 1)  # type: ignore[list-item]
        self._w_grouped: list[torch.Tensor] = []
        for group in self._layer_groups:
            self._w_grouped.append(
                torch.empty((group.count, group.width, group.width), device=device, dtype=cfg.dtype)
            )

        # Preserve the exact pre-Stage-1.13 RNG consumption order: initialize
        # weights strictly by layer index, even when their storage is grouped.
        for l, (n_l, n_u) in enumerate(zip(cfg.dims[:-1], cfg.dims[1:])):
            gi = self._edge_to_group.get(l)
            if gi is None:
                w = torch.empty((n_l, n_u), device=device, dtype=cfg.dtype)
            else:
                group = self._layer_groups[gi]
                z = l - group.start
                w = self._w_grouped[gi][z]
            torch.nn.init.kaiming_uniform_(w, a=5 ** 0.5, generator=g)
            self.w[l] = w

        self.x: list[torch.Tensor] | None = None
        self.e: list[torch.Tensor] | None = None
        self.a: list[torch.Tensor | None] | None = None
        self._x_grouped: list[torch.Tensor] = []
        self._e_grouped: list[torch.Tensor] = []
        self._a_grouped: list[torch.Tensor] | None = None

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

    def _grouped_edge_indices(self) -> set[int]:
        return set(self._edge_to_group)

    def _apply_grid_z_policy(self, batch: int) -> None:
        if not self.cfg.use_grid_z or not self.cfg.grid_z_auto:
            return
        from .gridz_tuning import lookup_grid_z_cap
        cap = lookup_grid_z_cap(
            self.cfg.grid_z_policy_path,
            device_name=torch.cuda.get_device_name(self.device),
            capability=torch.cuda.get_device_capability(self.device),
            dims=self.cfg.dims,
            batch=batch,
            activation=self.cfg.activation,
            dtype=self.cfg.dtype,
            default=int(self.cfg.grid_z_max_layers),
        )
        cap = max(0, int(cap))
        if cap == self._grid_z_cap:
            return
        self._grid_z_cap = cap
        self._layer_groups = (
            find_internal_square_groups(self.cfg.dims, max_group_count=cap) if cap >= 2 else ()
        )
        self._edge_to_group = {}
        for gi, group in enumerate(self._layer_groups):
            for l in group.edge_indices:
                self._edge_to_group[l] = gi
        old_w = list(self.w)
        self._w_grouped = []
        self.w = list(old_w)
        for group in self._layer_groups:
            # Policy changes are applied before the first training step. Repack existing
            # initial weights once, then rebind each layer to the packed storage so all
            # future updates remain in the active Z-group tensor.
            stacked = torch.stack([old_w[l] for l in group.edge_indices], dim=0)
            self._w_grouped.append(stacked)
            for z, l in enumerate(group.edge_indices):
                self.w[l] = stacked[z]
        self.x = None
        self.e = None
        self.a = None
        self._x_grouped = []
        self._e_grouped = []
        self._a_grouped = None

    def initialize_batch(self, x_input: torch.Tensor, y_target: torch.Tensor) -> None:
        """Reset endpoints and initialize hidden states by a forward prediction.
        Repeated internal layers use zero-copy contiguous [group, state, :] views.
        """
        B = x_input.shape[1]
        self._apply_grid_z_policy(B)
        if x_input.shape[0] != self.cfg.dims[-1] or y_target.shape[0] != self.cfg.dims[0]:
            raise ValueError("Batch dimensions do not match network dims")
        if self.x is None or self.e is None or self.x[0].shape[1] != B:
            self.x = [None] * len(self.cfg.dims)  # type: ignore[list-item]
            self.e = [None] * len(self.cfg.dims)  # type: ignore[list-item]
            self._x_grouped = []
            self._e_grouped = []
            self._a_grouped = None

            if self.cfg.recompute_activation:
                self.a = None
            else:
                self.a = [None] * len(self.cfg.dims)  # type: ignore[list-item]
                self._a_grouped = []

            grouped_states: set[int] = set()
            for gi, group in enumerate(self._layer_groups):
                xbuf = torch.empty(
                    (group.state_count, group.width, B),
                    device=self.device,
                    dtype=self.cfg.dtype,
                )
                ebuf = torch.empty_like(xbuf)
                self._x_grouped.append(xbuf)
                self._e_grouped.append(ebuf)
                if self._a_grouped is not None:
                    self._a_grouped.append(torch.empty_like(xbuf))

                for z, layer in enumerate(group.state_indices):
                    grouped_states.add(layer)
                    self.x[layer] = xbuf[z]
                    self.e[layer] = ebuf[z]
                    if self.a is not None and self._a_grouped is not None:
                        self.a[layer] = self._a_grouped[gi][z]

            for layer, d in enumerate(self.cfg.dims):
                if layer in grouped_states:
                    continue
                self.x[layer] = torch.empty((d, B), device=self.device, dtype=self.cfg.dtype)
                self.e[layer] = torch.empty_like(self.x[layer])
                if self.a is not None:
                    self.a[layer] = torch.empty_like(self.x[layer])

        assert self.x is not None and self.e is not None
        self.x[-1].copy_(x_input)
        if self.a is not None:
            self.a[-1].copy_(self.act.forward(self.x[-1]))
        for l in reversed(range(self.L)):
            pred = self.w[l] @ self.act.forward(self.x[l + 1])
            if l > 0:
                self.x[l].copy_(pred)
        self.x[0].copy_(y_target)

    def _get_prediction_kernel(self, l: int):
        assert self.x is not None
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
        assert self.x is not None
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
        assert self.x is not None
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

    def _get_prediction_batch_kernel(self, group: LayerBatchGroup):
        B = self.x[group.start].shape[1]  # type: ignore[index]
        return build_prediction_error_batched(
            group.width,
            group.width,
            B,
            group.count,
            self.cfg.activation,
            str(self.cfg.dtype).split(".")[-1],
            self._tuned_or_default("prediction", group.width, group.width, B),
        )

    def _get_inference_batch_kernel(self, group: LayerBatchGroup):
        B = self.x[group.start + 1].shape[1]  # type: ignore[index]
        return build_inference_update_batched(
            group.width,
            group.width,
            B,
            group.count,
            self.cfg.activation,
            str(self.cfg.dtype).split(".")[-1],
            self._tuned_or_default("inference", group.width, group.width, B),
            self.cfg.gamma,
            not self.cfg.recompute_activation,
        )

    def _get_weight_batch_kernel(self, group: LayerBatchGroup):
        B = self.x[group.start].shape[1]  # type: ignore[index]
        return build_weight_update_batched(
            group.width,
            group.width,
            B,
            group.count,
            self.cfg.activation,
            str(self.cfg.dtype).split(".")[-1],
            self._tuned_or_default("weight", group.width, group.width, B),
            self.cfg.alpha,
            self.cfg.recompute_activation,
        )

    @torch.no_grad()
    def step_initialized(self, *, collect_metrics: bool = False) -> dict[str, float]:
        if self.x is None or self.e is None:
            raise RuntimeError("Call initialize_batch before step_initialized")

        grouped = self._grouped_edge_indices()

        # Phase 1: prediction/error. All groups are independent and can be launched
        # as layer-batched 3-D kernels. Singleton edges keep the old kernel path.
        for gi, group in enumerate(self._layer_groups):
            xs = self._x_grouped[gi]
            es = self._e_grouped[gi]
            self._get_prediction_batch_kernel(group)(
                xs[1:], self._w_grouped[gi], xs[:-1], es[:-1]
            )
        for l in range(self.L):
            if l in grouped:
                continue
            self._get_prediction_kernel(l)(self.x[l + 1], self.w[l], self.x[l], self.e[l])

        # Phase 2: inference. For a square hidden run, l=start+1..end+1 are
        # updated together; W_lower/e_lower correspond to the preceding edges.
        for gi, group in enumerate(self._layer_groups):
            xs = self._x_grouped[gi]
            es = self._e_grouped[gi]
            ws = self._w_grouped[gi]
            if self.a is None:
                self._get_inference_batch_kernel(group)(
                    xs[1:], es[1:], ws, es[:-1]
                )
            else:
                assert self._a_grouped is not None
                self._get_inference_batch_kernel(group)(
                    xs[1:], es[1:], ws, es[:-1], self._a_grouped[gi][1:]
                )

        grouped_inference_layers: set[int] = set()
        for group in self._layer_groups:
            grouped_inference_layers.update(range(group.start + 1, group.end + 2))
        for l in range(1, self.L):
            if l in grouped_inference_layers:
                continue
            if self.a is None:
                self._get_inference_kernel(l)(self.x[l], self.e[l], self.w[l - 1], self.e[l - 1])
            else:
                assert self.a[l] is not None
                self._get_inference_kernel(l)(self.x[l], self.e[l], self.w[l - 1], self.e[l - 1], self.a[l])

        # Phase 3: weight update. The same packed edge/state slices are reused;
        # there is no pack/unpack kernel and no additional global-memory copy.
        for gi, group in enumerate(self._layer_groups):
            es = self._e_grouped[gi]
            ws = self._w_grouped[gi]
            if self.a is None:
                xs = self._x_grouped[gi]
                self._get_weight_batch_kernel(group)(ws, es[:-1], xs[1:])
            else:
                assert self._a_grouped is not None
                self._get_weight_batch_kernel(group)(ws, es[:-1], self._a_grouped[gi][1:])

        for l in range(self.L):
            if l in grouped:
                continue
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
        """Force JIT compilation of the current shape, including grid.z kernels."""
        if self.x is None or self.e is None:
            raise RuntimeError("Call initialize_batch before precompile")

        kernels = []
        grouped = self._grouped_edge_indices()
        for gi, group in enumerate(self._layer_groups):
            kernels.append(self._get_prediction_batch_kernel(group))
        for l in range(self.L):
            if l not in grouped:
                kernels.append(self._get_prediction_kernel(l))

        grouped_inference_layers: set[int] = set()
        for gi, group in enumerate(self._layer_groups):
            kernels.append(self._get_inference_batch_kernel(group))
            grouped_inference_layers.update(range(group.start + 1, group.end + 2))
        for l in range(1, self.L):
            if l not in grouped_inference_layers:
                kernels.append(self._get_inference_kernel(l))

        for gi, group in enumerate(self._layer_groups):
            kernels.append(self._get_weight_batch_kernel(group))
        for l in range(self.L):
            if l not in grouped:
                kernels.append(self._get_weight_kernel(l))

        if all(hasattr(k, "get_kernel_source") for k in kernels):
            for k in kernels:
                _ = k.get_kernel_source()
            torch.cuda.synchronize()
            return

        w_backup = [w.clone() for w in self.w]
        x_backup = [x.clone() for x in self.x]
        e_backup = [e.clone() for e in self.e]
        a_backup = None if self.a is None else [None if v is None else v.clone() for v in self.a]
        self.step_initialized(collect_metrics=False)
        torch.cuda.synchronize()
        for dst, src in zip(self.w, w_backup):
            dst.copy_(src)
        for dst, src in zip(self.x, x_backup):
            dst.copy_(src)
        for dst, src in zip(self.e, e_backup):
            dst.copy_(src)
        if self.a is not None and a_backup is not None:
            for dst, src in zip(self.a, a_backup):
                if dst is not None and src is not None:
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
        h = x_input
        for l in reversed(range(self.L)):
            h = self.w[l] @ self.act.forward(h)
        return h


def prepare_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = images.flatten(1).T.contiguous().to(device=device, dtype=dtype, non_blocking=True)
    y = one_hot(labels.to(device, non_blocking=True), num_classes, dtype=dtype)
    return x, y
