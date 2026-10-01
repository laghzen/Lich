from __future__ import annotations

from dataclasses import asdict, dataclass
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
from .hierarchical_policy import HierarchicalPolicy, kernel_signature
from .weight_update_specialization import SmallBWeightPolicy, SmallBWeightUpdater


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
    hierarchical_policy_auto: bool = True
    hierarchical_policy_path: str = "results/hierarchical_policy.json"
    # Stage 2.4: steady-state graph execution.  In auto mode the existing
    # Stage-2 policy decides whether CUDA Graph should be used; without a
    # policy, the default remains the legacy direct-launch path.
    steady_state_graph_auto: bool = True
    steady_state_graph_default: bool = False
    # Optional measured small-B weight-update specialization.  Default is GEMM
    # until an exact-shape policy has been benchmarked on the target GPU.
    weight_update_mode: str = "gemm"
    weight_update_policy_path: str = "results/weight_update_policy.json"
    weight_update_outer_max_batch: int = 16


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
        self._hierarchical_policy: HierarchicalPolicy | None = None
        self._runtime_recompute_activation = bool(cfg.recompute_activation)
        if cfg.hierarchical_policy_auto:
            try:
                self._hierarchical_policy = HierarchicalPolicy(cfg.hierarchical_policy_path)
            except Exception:
                self._hierarchical_policy = None
        grid_z_cap = int(cfg.grid_z_max_layers)
        self._grid_z_cap = max(0, grid_z_cap)
        self._layer_groups: tuple[LayerBatchGroup, ...] = (
            find_internal_square_groups(cfg.dims, max_group_count=self._grid_z_cap) if cfg.use_grid_z and self._grid_z_cap >= 2 else ()
        )
        self._edge_to_group: dict[int, int] = {}
        for gi, group in enumerate(self._layer_groups):
            for l in group.edge_indices:
                self._edge_to_group[l] = gi
        self._grouped_edge_indices_cached: frozenset[int] = frozenset(self._edge_to_group)
        self._grouped_inference_layers_cached: frozenset[int] = frozenset(
            l for group in self._layer_groups for l in range(group.start + 1, group.end + 2)
        )

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

        # Stage 2.3: immutable steady-state execution plan.  The policy/kernel
        # selection is performed once per active batch/shape and the resulting
        # TileLang callable objects are reused by every following step.  This
        # removes repeated Python lru_cache/policy lookups from the hot path.
        self._runtime_plan_key: tuple | None = None
        self._pred_singleton_kernels: list[object | None] = []
        self._inf_singleton_kernels: list[object | None] = []
        self._weight_singleton_kernels: list[object | None] = []
        self._pred_singleton_ops: tuple[tuple[int, object], ...] = ()
        self._inf_singleton_ops: tuple[tuple[int, object], ...] = ()
        self._weight_singleton_ops: tuple[tuple[int, object], ...] = ()
        self._pred_group_kernels: tuple[object, ...] = ()
        self._inf_group_kernels: tuple[object, ...] = ()
        self._weight_group_kernels: tuple[object, ...] = ()
        self._active_kernel_signature_cache: dict[int, str] = {}
        self._runtime_recompute_policy_cache: dict[int, bool] = {}
        self._runtime_grid_z_policy_applied_for_batch: int | None = None
        # Stage 2.4: persistent graph runner for the currently prepared
        # batch topology.  The runner is reused across batch re-initialization
        # as long as tensor shape, runtime plan and graph step count remain
        # identical.
        self._steady_state_runner = None
        self._steady_state_graph_key: tuple | None = None
        self._steady_state_graph_steps: int | None = None

        # Optional Stage-2.5 small-B weight specialization.  No behavior
        # changes unless the caller explicitly enables mode="auto" and an
        # exact measured policy row selects the outer backend.
        self._weight_update_policy: SmallBWeightPolicy | None = None
        self._small_b_updater = SmallBWeightUpdater(alpha=float(cfg.alpha))
        if str(cfg.weight_update_mode).lower() == "auto":
            try:
                self._weight_update_policy = SmallBWeightPolicy(cfg.weight_update_policy_path)
            except Exception:
                self._weight_update_policy = None
        self._weight_singleton_outer_ops: tuple[tuple[int, object], ...] = ()
        self._weight_group_outer_ops: tuple[object | None, ...] = ()

    @property
    def L(self) -> int:
        return len(self.w)

    def _apply_hierarchical_recompute_policy(self, batch: int) -> None:
        """Apply the exact depth/kernel-signature recompute policy before buffer allocation."""
        batch = int(batch)
        cached = self._runtime_recompute_policy_cache.get(batch)
        if cached is not None:
            self._runtime_recompute_activation = bool(cached)
            return
        selected = bool(self.cfg.recompute_activation)
        if self._hierarchical_policy is not None:
            try:
                sig = self._active_kernel_signature(batch)
                selected = self._hierarchical_policy.selected_recompute_activation(
                    dims=self.cfg.dims, batch=batch, activation=self.cfg.activation,
                    dtype=self.cfg.dtype, kernel_sig=sig, default=selected,
                )
            except Exception:
                selected = bool(self.cfg.recompute_activation)
        self._runtime_recompute_activation = bool(selected)
        self._runtime_recompute_policy_cache[batch] = bool(selected)

    @property
    def recompute_activation(self) -> bool:
        return bool(self._runtime_recompute_activation)

    def _tuned_or_default(self, kind: str, M: int, K: int, B: int) -> KernelConfig:
        if self._hierarchical_policy is not None:
            try:
                tuned = self._hierarchical_policy.lookup_profile_kernel(
                    dims=tuple(int(d) for d in self.cfg.dims),
                    batch=int(B),
                    kind=kind,
                    M=M,
                    K=K,
                    activation=self.cfg.activation,
                    dtype=self.cfg.dtype,
                )
            except Exception:
                tuned = None
            if tuned is not None:
                return tuned
            tuned = self._hierarchical_policy.lookup_kernel(
                kind=kind, M=M, K=K, B=B,
                activation=self.cfg.activation, dtype=self.cfg.dtype,
            )
            if tuned is not None:
                return tuned
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

    def _active_kernel_signature(self, batch: int) -> str:
        batch = int(batch)
        cached = self._active_kernel_signature_cache.get(batch)
        if cached is not None:
            return cached
        rows = []
        seen = set()
        for l in range(self.L):
            shapes = (
                ("prediction", self.cfg.dims[l], self.cfg.dims[l + 1]),
                ("weight", self.cfg.dims[l], self.cfg.dims[l + 1]),
            )
            if l > 0:
                shapes += (("inference", self.cfg.dims[l], self.cfg.dims[l - 1]),)
            for kind, M, K in shapes:
                key = (kind, M, K, batch)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "kind": kind,
                    "shape": {"M": M, "K": K, "B": batch},
                    "activation": self.cfg.activation,
                    "dtype": str(self.cfg.dtype).split(".")[-1],
                    "config": asdict(self._tuned_or_default(kind, M, K, batch)),
                })
        sig = kernel_signature(rows)
        self._active_kernel_signature_cache[batch] = sig
        return sig

    def recommended_use_cuda_graph(self, batch: int | None = None, *, default: bool = False) -> bool:
        """Return the Stage-2 graph recommendation for the active shape, when available."""
        if self._hierarchical_policy is None:
            return bool(default)
        if batch is None:
            if self.x is None:
                return bool(default)
            batch = int(self.x[-1].shape[1])
        try:
            sig = self._active_kernel_signature(int(batch))
            return self._hierarchical_policy.selected_use_graph(
                dims=self.cfg.dims, batch=int(batch), activation=self.cfg.activation,
                dtype=self.cfg.dtype, kernel_sig=sig, default=default,
            )
        except Exception:
            return bool(default)

    def _grouped_edge_indices(self) -> frozenset[int]:
        return self._grouped_edge_indices_cached

    def _invalidate_steady_state_graph(self) -> None:
        self._steady_state_runner = None
        self._steady_state_graph_key = None
        self._steady_state_graph_steps = None

    def _apply_grid_z_policy(self, batch: int) -> None:
        batch = int(batch)
        if not self.cfg.use_grid_z or not self.cfg.grid_z_auto:
            return
        if self._runtime_grid_z_policy_applied_for_batch == batch:
            return
        cap = None
        if self._hierarchical_policy is not None:
            try:
                sig = self._active_kernel_signature(batch)
                row = self._hierarchical_policy.lookup_execution(
                    dims=self.cfg.dims, batch=batch, activation=self.cfg.activation,
                    dtype=self.cfg.dtype, kernel_sig=sig,
                )
                if row is not None:
                    cap = int((row.get("best") or {}).get("grid_z_max_layers", self.cfg.grid_z_max_layers))
            except Exception:
                cap = None
        if cap is None:
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
            self._runtime_grid_z_policy_applied_for_batch = batch
            return
        self._grid_z_cap = cap
        self._layer_groups = (
            find_internal_square_groups(self.cfg.dims, max_group_count=cap) if cap >= 2 else ()
        )
        self._edge_to_group = {}
        for gi, group in enumerate(self._layer_groups):
            for l in group.edge_indices:
                self._edge_to_group[l] = gi
        self._grouped_edge_indices_cached = frozenset(self._edge_to_group)
        self._grouped_inference_layers_cached = frozenset(
            l for group in self._layer_groups for l in range(group.start + 1, group.end + 2)
        )
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
        self._invalidate_steady_state_graph()
        self._runtime_plan_key = None
        self._pred_singleton_kernels = []
        self._inf_singleton_kernels = []
        self._weight_singleton_kernels = []
        self._pred_singleton_ops = ()
        self._inf_singleton_ops = ()
        self._weight_singleton_ops = ()
        self._pred_group_kernels = ()
        self._inf_group_kernels = ()
        self._weight_group_kernels = ()
        self._runtime_grid_z_policy_applied_for_batch = batch

    def _outer_weight_enabled(self, M: int, K: int, B: int, count: int = 1) -> bool:
        if str(self.cfg.weight_update_mode).lower() != "auto":
            return False
        if self._weight_update_policy is None:
            return False
        if int(B) > int(self.cfg.weight_update_outer_max_batch):
            return False
        if self.cfg.activation.lower() not in {"relu", "identity", "linear"}:
            return False
        try:
            return self._weight_update_policy.backend(
                M=int(M), K=int(K), B=int(B),
                activation=self.cfg.activation,
                dtype=self.cfg.dtype,
                count=int(count),
            ) == "outer"
        except Exception:
            return False

    def _prepare_runtime_plan(self, batch: int) -> None:
        """Build the active kernel-call plan exactly once for the current execution state."""
        batch = int(batch)
        key = (
            tuple(int(d) for d in self.cfg.dims),
            batch,
            self.cfg.activation,
            str(self.cfg.dtype).split(".")[-1],
            bool(self.recompute_activation),
            int(self._grid_z_cap),
            self._active_kernel_signature(batch),
        )
        if key == self._runtime_plan_key:
            return

        pred_single: list[object | None] = [None] * self.L
        inf_single: list[object | None] = [None] * self.L
        weight_single: list[object | None] = [None] * self.L
        weight_outer_single: list[tuple[int, object]] = []
        pred_group: list[object] = []
        inf_group: list[object] = []
        weight_group: list[object] = []
        weight_outer_group: list[object | None] = []

        grouped = self._grouped_edge_indices()
        for l in range(self.L):
            if l not in grouped:
                pred_single[l] = self._get_prediction_kernel(l)

        grouped_inference_layers: set[int] = set()
        for group in self._layer_groups:
            grouped_inference_layers.update(range(group.start + 1, group.end + 2))

        for l in range(1, self.L):
            if l not in grouped_inference_layers:
                inf_single[l] = self._get_inference_kernel(l)

        for l in range(self.L):
            if l not in grouped:
                M, K = self.cfg.dims[l], self.cfg.dims[l + 1]
                if self._outer_weight_enabled(M, K, batch):
                    outer = self._small_b_updater.make_singleton(
                        M=M, K=K, B=batch, activation=self.cfg.activation, dtype=self.cfg.dtype
                    )
                    if outer is not None:
                        weight_outer_single.append((l, outer))
                        continue
                weight_single[l] = self._get_weight_kernel(l)

        for group in self._layer_groups:
            pred_group.append(self._get_prediction_batch_kernel(group))
            inf_group.append(self._get_inference_batch_kernel(group))
            if self._outer_weight_enabled(group.width, group.width, batch, group.count):
                outer = self._small_b_updater.make_grouped(
                    count=group.count, M=group.width, K=group.width, B=batch,
                    activation=self.cfg.activation, dtype=self.cfg.dtype
                )
            else:
                outer = None
            weight_outer_group.append(outer)
            weight_group.append(self._get_weight_batch_kernel(group) if outer is None else None)

        self._pred_singleton_kernels = pred_single
        self._inf_singleton_kernels = inf_single
        self._weight_singleton_kernels = weight_single
        self._weight_singleton_outer_ops = tuple(weight_outer_single)
        self._weight_group_outer_ops = tuple(weight_outer_group)
        self._pred_singleton_ops = tuple(
            (l, kernel) for l, kernel in enumerate(pred_single) if kernel is not None
        )
        self._inf_singleton_ops = tuple(
            (l, kernel) for l, kernel in enumerate(inf_single) if kernel is not None
        )
        self._weight_singleton_ops = tuple(
            (l, kernel) for l, kernel in enumerate(weight_single) if kernel is not None
        )
        self._pred_group_kernels = tuple(pred_group)
        self._inf_group_kernels = tuple(inf_group)
        self._weight_group_kernels = tuple(weight_group)
        self._runtime_plan_key = key

    def runtime_plan_ready(self) -> bool:
        """Return whether the current batch/shape has a prepared steady-state plan."""
        return self._runtime_plan_key is not None

    def initialize_batch(self, x_input: torch.Tensor, y_target: torch.Tensor) -> None:
        """Reset endpoints and initialize hidden states by a forward prediction.
        Repeated internal layers use zero-copy contiguous [group, state, :] views.
        """
        B = x_input.shape[1]
        self._apply_grid_z_policy(B)
        self._apply_hierarchical_recompute_policy(B)
        if x_input.shape[0] != self.cfg.dims[-1] or y_target.shape[0] != self.cfg.dims[0]:
            raise ValueError("Batch dimensions do not match network dims")
        if self.x is None or self.e is None or self.x[0].shape[1] != B:
            # Device-buffer reallocation invalidates any CUDA Graph that captured
            # the previous tensor addresses. Keep the old graph alive only until
            # this point so it can be released before new buffers are allocated.
            self._invalidate_steady_state_graph()
            self.x = [None] * len(self.cfg.dims)  # type: ignore[list-item]
            self.e = [None] * len(self.cfg.dims)  # type: ignore[list-item]
            self._x_grouped = []
            self._e_grouped = []
            self._a_grouped = None

            if self.recompute_activation:
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
        self._prepare_runtime_plan(B)
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
            not self.recompute_activation,
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
            self.recompute_activation,
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
            not self.recompute_activation,
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
            self.recompute_activation,
        )

    @torch.no_grad()
    def step_initialized(self, *, collect_metrics: bool = False) -> dict[str, float]:
        if self.x is None or self.e is None:
            raise RuntimeError("Call initialize_batch before step_initialized")

        if self._runtime_plan_key is None:
            self._prepare_runtime_plan(int(self.x[-1].shape[1]))

        # Phase 1: prediction/error. All groups are independent and can be launched
        # as layer-batched 3-D kernels. Singleton edges use the prepared kernel handle.
        for gi, group in enumerate(self._layer_groups):
            xs = self._x_grouped[gi]
            es = self._e_grouped[gi]
            self._pred_group_kernels[gi](
                xs[1:], self._w_grouped[gi], xs[:-1], es[:-1]
            )
        for l, kernel in self._pred_singleton_ops:
            kernel(self.x[l + 1], self.w[l], self.x[l], self.e[l])

        # Phase 2: inference. Group topology and singleton operations are fixed by
        # the prepared execution plan and therefore require no per-step set building.
        for gi, group in enumerate(self._layer_groups):
            xs = self._x_grouped[gi]
            es = self._e_grouped[gi]
            ws = self._w_grouped[gi]
            kernel = self._inf_group_kernels[gi]
            if self.a is None:
                kernel(xs[1:], es[1:], ws, es[:-1])
            else:
                assert self._a_grouped is not None
                kernel(xs[1:], es[1:], ws, es[:-1], self._a_grouped[gi][1:])

        for l, kernel in self._inf_singleton_ops:
            if self.a is None:
                kernel(self.x[l], self.e[l], self.w[l - 1], self.e[l - 1])
            else:
                assert self.a[l] is not None
                kernel(self.x[l], self.e[l], self.w[l - 1], self.e[l - 1], self.a[l])

        # Phase 3: weight update. The same packed edge/state slices are reused;
        # there is no pack/unpack kernel and no additional global-memory copy.
        for gi, group in enumerate(self._layer_groups):
            es = self._e_grouped[gi]
            ws = self._w_grouped[gi]
            outer = self._weight_group_outer_ops[gi] if gi < len(self._weight_group_outer_ops) else None
            if outer is not None:
                if self.a is None:
                    xs = self._x_grouped[gi]
                    outer(ws, es[:-1], xs[1:])
                else:
                    assert self._a_grouped is not None
                    outer(ws, es[:-1], self._a_grouped[gi][1:])
                continue
            kernel = self._weight_group_kernels[gi]
            if self.a is None:
                xs = self._x_grouped[gi]
                kernel(ws, es[:-1], xs[1:])
            else:
                assert self._a_grouped is not None
                kernel(ws, es[:-1], self._a_grouped[gi][1:])

        for l, outer in self._weight_singleton_outer_ops:
            if self.a is None:
                outer(self.w[l], self.e[l], self.x[l + 1])
            else:
                assert self.a[l + 1] is not None
                outer(self.w[l], self.e[l], self.a[l + 1])

        for l, kernel in self._weight_singleton_ops:
            if self.a is None:
                kernel(self.w[l], self.e[l], self.x[l + 1])
            else:
                assert self.a[l + 1] is not None
                kernel(self.w[l], self.e[l], self.a[l + 1])

        if collect_metrics:
            assert self.e is not None
            return {"max_abs_error": float(max(v.float().abs().max().item() for v in self.e[:-1]))}
        return {}

    @torch.no_grad()
    def step(self, x_input: torch.Tensor, y_target: torch.Tensor) -> dict[str, float]:
        self.initialize_batch(x_input, y_target)
        return self.step_initialized(collect_metrics=True)

    def steady_state_graph_ready(self, steps: int) -> bool:
        return (
            self._steady_state_runner is not None
            and self._steady_state_graph_steps == int(steps)
            and self._steady_state_graph_key == self._runtime_plan_key
        )

    def prepare_steady_state(
        self,
        x_input: torch.Tensor,
        y_target: torch.Tensor,
        *,
        steps: int = 1,
        use_graph: bool | None = None,
    ) -> bool:
        """Prepare one fixed-shape batch for repeated low-overhead execution.

        Returns True when the CUDA Graph path is active.  The input/target are copied
        into reusable device buffers once at the batch boundary; subsequent calls to
        run_prepared_steady_state() replay the captured graph without Python kernel
        selection/lookup.
        """
        steps = max(1, int(steps))
        self.initialize_batch(x_input, y_target)
        B = int(x_input.shape[1])
        if use_graph is None:
            use_graph = self.recommended_use_cuda_graph(
                B, default=bool(self.cfg.steady_state_graph_default)
            ) if self.cfg.steady_state_graph_auto else bool(self.cfg.steady_state_graph_default)
        use_graph = bool(use_graph)
        if not use_graph:
            return False
        if not self.steady_state_graph_ready(steps):
            # capture_graph() internally calls precompile(), so compilation happens
            # before entering graph capture rather than lazily during replay.
            self._steady_state_runner = self.capture_graph(steps)
            self._steady_state_graph_steps = steps
            self._steady_state_graph_key = self._runtime_plan_key
        return True

    @torch.no_grad()
    def run_prepared_steady_state(
        self, *,
        steps: int | None = None,
        repeats: int = 1,
        use_graph: bool | None = None,
        collect_metrics: bool = False,
    ) -> dict[str, float]:
        """Execute a prepared batch repeatedly; no batch initialization is performed."""
        if self.x is None or self.e is None or self._runtime_plan_key is None:
            raise RuntimeError("Call prepare_steady_state/initialize_batch first")
        steps = max(1, int(self._steady_state_graph_steps if steps is None else steps))
        repeats = max(1, int(repeats))
        if use_graph is None:
            use_graph = self.steady_state_graph_ready(steps)
        if use_graph:
            if not self.steady_state_graph_ready(steps):
                raise RuntimeError("Requested graph replay without a matching captured graph")
            assert self._steady_state_runner is not None
            for _ in range(repeats):
                self._steady_state_runner.replay()
        else:
            for _ in range(repeats):
                for _ in range(steps):
                    self.step_initialized(collect_metrics=False)
        torch.cuda.synchronize(self.device)
        if collect_metrics:
            assert self.e is not None
            return {"max_abs_error": float(max(v.float().abs().max().item() for v in self.e[:-1]))}
        return {}

    def reset_steady_state(self) -> None:
        """Drop only the cached graph runner; kernel/runtime plan remains intact."""
        self._invalidate_steady_state_graph()

    @property
    def steady_state_uses_graph(self) -> bool:
        return self._steady_state_runner is not None

    def run_steady_state_batch(
        self,
        x_input: torch.Tensor,
        y_target: torch.Tensor,
        *,
        steps: int = 1,
        repeats: int = 1,
        use_graph: bool | None = None,
        collect_metrics: bool = False,
    ) -> dict[str, float]:
        self.prepare_steady_state(
            x_input, y_target, steps=steps, use_graph=use_graph
        )
        return self.run_prepared_steady_state(
            steps=steps, repeats=repeats, use_graph=use_graph,
            collect_metrics=collect_metrics,
        )

    def precompile(self) -> None:
        """Force JIT compilation of the current shape, including grid.z kernels."""
        if self.x is None or self.e is None:
            raise RuntimeError("Call initialize_batch before precompile")

        self._prepare_runtime_plan(int(self.x[-1].shape[1]))
        kernels = [k for k in self._pred_group_kernels if k is not None]
        kernels += [k for k in self._pred_singleton_kernels if k is not None]
        kernels += [k for k in self._inf_group_kernels if k is not None]
        kernels += [k for k in self._inf_singleton_kernels if k is not None]
        kernels += [k for k in self._weight_group_kernels if k is not None]
        kernels += [k for k in self._weight_singleton_kernels if k is not None]

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
