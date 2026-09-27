from dataclasses import dataclass
from functools import lru_cache

import torch

from .tilelang_kernels import (
    KernelConfig,
    _act,
    _dact,
    _decorate_shared,
    _jit_compile_flags,
    _require_tilelang,
)


@dataclass(frozen=True)
class LayerBatchGroup:
    """A contiguous run of identical internal square weight layers.

    For dims [..., D, D, D, ...] we batch the repeated edges
    l=start..end and the hidden states start..end+1. Endpoints are never
    included, so inference never updates x[0] or x[L].
    """

    start: int
    end: int
    width: int

    @property
    def count(self) -> int:
        return self.end - self.start + 1

    @property
    def state_count(self) -> int:
        return self.count + 1

    @property
    def edge_indices(self) -> tuple[int, ...]:
        return tuple(range(self.start, self.end + 1))

    @property
    def state_indices(self) -> tuple[int, ...]:
        return tuple(range(self.start, self.end + 2))


def find_internal_square_groups(
    dims: tuple[int, ...],
    *,
    max_group_count: int = 2,
) -> tuple[LayerBatchGroup, ...]:
    """Find conservative, non-overlapping grid.z groups.

    Stage 1.14 showed that a 2-layer Z-group is beneficial on the target
    RTX 3060, while a 4-layer Z-group regressed on depth=6.  We therefore
    cap the group size and leave one edge between groups so the corresponding
    hidden-state buffers never overlap.
    """
    if len(dims) < 4 or max_group_count < 2:
        return ()

    L = len(dims) - 1
    out: list[LayerBatchGroup] = []
    l = 1
    while l <= L - 2:
        d = int(dims[l])
        if int(dims[l + 1]) != d:
            l += 1
            continue

        run_start = l
        while l <= L - 2 and int(dims[l]) == d and int(dims[l + 1]) == d:
            l += 1
        run_end = l - 1

        p = run_start
        while p <= run_end:
            q = min(p + max_group_count - 1, run_end)
            count = q - p + 1
            if count >= 2:
                out.append(LayerBatchGroup(start=p, end=q, width=d))
                # Do not create another group sharing the boundary state.
                p = q + 2
            else:
                p += 1

    return tuple(out)


@lru_cache(maxsize=128)
def build_prediction_error_batched(
    M: int,
    K: int,
    B: int,
    groups: int,
    activation: str,
    dtype: str,
    cfg: KernelConfig,
):
    """Batched prediction/error: W[g] @ f(X_upper[g]) -> E[g]."""
    if groups < 2:
        raise ValueError("batched prediction kernel requires groups >= 2")
    if M != K:
        raise ValueError("Stage 1.13 batches only square internal layers")

    _require_tilelang()
    import tilelang
    import tilelang.language as T

    @tilelang.jit(target={"kind": "cuda", "arch": "sm_86"}, compile_flags=_jit_compile_flags())
    def kernel():
        @T.prim_func
        def main(
            X_upper: T.Tensor((groups, K, B), dtype),
            W: T.Tensor((groups, M, K), dtype),
            X_lower: T.Tensor((groups, M, B), dtype),
            E: T.Tensor((groups, M, B), dtype),
        ):
            with T.Kernel(
                T.ceildiv(B, cfg.block_n),
                T.ceildiv(M, cfg.block_m),
                groups,
                threads=cfg.threads,
            ) as (bx, by, bz):
                a_raw = T.alloc_shared((cfg.block_k, cfg.block_n), dtype)
                a_shared = T.alloc_shared((cfg.block_k, cfg.block_n), dtype)
                w_shared = T.alloc_shared((cfg.block_m, cfg.block_k), dtype)
                c_local = T.alloc_fragment((cfg.block_m, cfg.block_n), T.float32)
                _decorate_shared(tilelang, T, w_shared, a_shared, cfg.shared_swizzle)
                if cfg.swizzle:
                    T.use_swizzle(panel_size=cfg.swizzle_panel)
                T.clear(c_local)
                for ko in T.Pipelined(T.ceildiv(K, cfg.block_k), num_stages=cfg.num_stages):
                    T.copy(X_upper[bz, ko * cfg.block_k, bx * cfg.block_n], a_raw)
                    T.copy(W[bz, by * cfg.block_m, ko * cfg.block_k], w_shared)
                    for k, b in T.Parallel(cfg.block_k, cfg.block_n):
                        gi = ko * cfg.block_k + k
                        gj = bx * cfg.block_n + b
                        if (gi < K) and (gj < B):
                            a_shared[k, b] = _act(a_raw[k, b], activation, T)
                        else:
                            a_shared[k, b] = T.cast(0, dtype)
                    T.gemm(w_shared, a_shared, c_local, transpose_B=False)
                for i, j in T.Parallel(cfg.block_m, cfg.block_n):
                    ii = by * cfg.block_m + i
                    jj = bx * cfg.block_n + j
                    if (ii < M) and (jj < B):
                        E[bz, ii, jj] = T.cast(X_lower[bz, ii, jj], dtype) - T.cast(c_local[i, j], dtype)
        return main

    return kernel()


@lru_cache(maxsize=128)
def build_inference_update_batched(
    M: int,
    K: int,
    B: int,
    groups: int,
    activation: str,
    dtype: str,
    cfg: KernelConfig,
    gamma: float,
    save_activation: bool = False,
):
    """Batched iPC inference for repeated internal square layers."""
    if groups < 2:
        raise ValueError("batched inference kernel requires groups >= 2")
    if M != K:
        raise ValueError("Stage 1.13 batches only square internal layers")

    _require_tilelang()
    import tilelang
    import tilelang.language as T

    @tilelang.jit(target={"kind": "cuda", "arch": "sm_86"}, compile_flags=_jit_compile_flags())
    def kernel():
        if save_activation:
            @T.prim_func
            def main(
                X: T.Tensor((groups, M, B), dtype),
                E: T.Tensor((groups, M, B), dtype),
                W_lower: T.Tensor((groups, K, M), dtype),
                E_lower: T.Tensor((groups, K, B), dtype),
                A_out: T.Tensor((groups, M, B), dtype),
            ):
                with T.Kernel(
                    T.ceildiv(B, cfg.block_n),
                    T.ceildiv(M, cfg.block_m),
                    groups,
                    threads=cfg.threads,
                ) as (bx, by, bz):
                    e_shared = T.alloc_shared((cfg.block_k, cfg.block_n), dtype)
                    w_shared = T.alloc_shared((cfg.block_k, cfg.block_m), dtype)
                    c_local = T.alloc_fragment((cfg.block_m, cfg.block_n), T.float32)
                    _decorate_shared(tilelang, T, w_shared, e_shared, cfg.shared_swizzle)
                    if cfg.swizzle:
                        T.use_swizzle(panel_size=cfg.swizzle_panel)
                    T.clear(c_local)
                    for ko in T.Pipelined(T.ceildiv(K, cfg.block_k), num_stages=cfg.num_stages):
                        T.copy(E_lower[bz, ko * cfg.block_k, bx * cfg.block_n], e_shared)
                        T.copy(W_lower[bz, ko * cfg.block_k, by * cfg.block_m], w_shared)
                        T.gemm(w_shared, e_shared, c_local, transpose_A=True, transpose_B=False)
                    for i, j in T.Parallel(cfg.block_m, cfg.block_n):
                        ii = by * cfg.block_m + i
                        jj = bx * cfg.block_n + j
                        if (ii < M) and (jj < B):
                            x = T.cast(X[bz, ii, jj], T.float32)
                            e = T.cast(E[bz, ii, jj], T.float32)
                            x_new = x + T.float32(gamma) * (-e + _dact(x, activation, T) * c_local[i, j])
                            X[bz, ii, jj] = T.cast(x_new, dtype)
                            A_out[bz, ii, jj] = T.cast(_act(x_new, activation, T), dtype)
            return main

        @T.prim_func
        def main(
            X: T.Tensor((groups, M, B), dtype),
            E: T.Tensor((groups, M, B), dtype),
            W_lower: T.Tensor((groups, K, M), dtype),
            E_lower: T.Tensor((groups, K, B), dtype),
        ):
            with T.Kernel(
                T.ceildiv(B, cfg.block_n),
                T.ceildiv(M, cfg.block_m),
                groups,
                threads=cfg.threads,
            ) as (bx, by, bz):
                e_shared = T.alloc_shared((cfg.block_k, cfg.block_n), dtype)
                w_shared = T.alloc_shared((cfg.block_k, cfg.block_m), dtype)
                c_local = T.alloc_fragment((cfg.block_m, cfg.block_n), T.float32)
                _decorate_shared(tilelang, T, w_shared, e_shared, cfg.shared_swizzle)
                if cfg.swizzle:
                    T.use_swizzle(panel_size=cfg.swizzle_panel)
                T.clear(c_local)
                for ko in T.Pipelined(T.ceildiv(K, cfg.block_k), num_stages=cfg.num_stages):
                    T.copy(E_lower[bz, ko * cfg.block_k, bx * cfg.block_n], e_shared)
                    T.copy(W_lower[bz, ko * cfg.block_k, by * cfg.block_m], w_shared)
                    T.gemm(w_shared, e_shared, c_local, transpose_A=True, transpose_B=False)
                for i, j in T.Parallel(cfg.block_m, cfg.block_n):
                    ii = by * cfg.block_m + i
                    jj = bx * cfg.block_n + j
                    if (ii < M) and (jj < B):
                        x = T.cast(X[bz, ii, jj], T.float32)
                        e = T.cast(E[bz, ii, jj], T.float32)
                        update = T.float32(gamma) * (-e + _dact(x, activation, T) * c_local[i, j])
                        X[bz, ii, jj] = T.cast(x + update, dtype)
        return main

    return kernel()


@lru_cache(maxsize=128)
def build_weight_update_batched(
    M: int,
    K: int,
    B: int,
    groups: int,
    activation: str,
    dtype: str,
    cfg: KernelConfig,
    alpha: float,
    recompute_activation: bool = True,
):
    """Batched W += alpha * E @ A.T for repeated internal square layers."""
    if groups < 2:
        raise ValueError("batched weight kernel requires groups >= 2")
    if M != K:
        raise ValueError("Stage 1.13 batches only square internal layers")

    _require_tilelang()
    import tilelang
    import tilelang.language as T

    @tilelang.jit(target={"kind": "cuda", "arch": "sm_86"}, compile_flags=_jit_compile_flags())
    def kernel():
        if recompute_activation:
            @T.prim_func
            def main(
                W: T.Tensor((groups, M, K), dtype),
                E: T.Tensor((groups, M, B), dtype),
                X_upper: T.Tensor((groups, K, B), dtype),
            ):
                with T.Kernel(
                    T.ceildiv(K, cfg.block_k),
                    T.ceildiv(M, cfg.block_m),
                    groups,
                    threads=cfg.threads,
                ) as (bx, by, bz):
                    e_shared = T.alloc_shared((cfg.block_m, cfg.block_n), dtype)
                    a_raw = T.alloc_shared((cfg.block_k, cfg.block_n), dtype)
                    a_shared = T.alloc_shared((cfg.block_k, cfg.block_n), dtype)
                    c_local = T.alloc_fragment((cfg.block_m, cfg.block_k), T.float32)
                    _decorate_shared(tilelang, T, e_shared, a_shared, cfg.shared_swizzle)
                    if cfg.swizzle:
                        T.use_swizzle(panel_size=cfg.swizzle_panel)
                    T.clear(c_local)
                    for bo in T.Pipelined(T.ceildiv(B, cfg.block_n), num_stages=cfg.num_stages):
                        T.copy(E[bz, by * cfg.block_m, bo * cfg.block_n], e_shared)
                        T.copy(X_upper[bz, bx * cfg.block_k, bo * cfg.block_n], a_raw)
                        for k, b in T.Parallel(cfg.block_k, cfg.block_n):
                            gi = bx * cfg.block_k + k
                            gj = bo * cfg.block_n + b
                            if (gi < K) and (gj < B):
                                a_shared[k, b] = _act(a_raw[k, b], activation, T)
                            else:
                                a_shared[k, b] = T.cast(0, dtype)
                        T.gemm(e_shared, a_shared, c_local, transpose_B=True)
                    for i, k in T.Parallel(cfg.block_m, cfg.block_k):
                        ii = by * cfg.block_m + i
                        kk = bx * cfg.block_k + k
                        if (ii < M) and (kk < K):
                            w_old = T.cast(W[bz, ii, kk], T.float32)
                            W[bz, ii, kk] = T.cast(w_old + T.float32(alpha) * c_local[i, k], dtype)
            return main

        @T.prim_func
        def main(
            W: T.Tensor((groups, M, K), dtype),
            E: T.Tensor((groups, M, B), dtype),
            A_upper: T.Tensor((groups, K, B), dtype),
        ):
            with T.Kernel(
                T.ceildiv(K, cfg.block_k),
                T.ceildiv(M, cfg.block_m),
                groups,
                threads=cfg.threads,
            ) as (bx, by, bz):
                e_shared = T.alloc_shared((cfg.block_m, cfg.block_n), dtype)
                a_shared = T.alloc_shared((cfg.block_k, cfg.block_n), dtype)
                c_local = T.alloc_fragment((cfg.block_m, cfg.block_k), T.float32)
                _decorate_shared(tilelang, T, e_shared, a_shared, cfg.shared_swizzle)
                if cfg.swizzle:
                    T.use_swizzle(panel_size=cfg.swizzle_panel)
                T.clear(c_local)
                for bo in T.Pipelined(T.ceildiv(B, cfg.block_n), num_stages=cfg.num_stages):
                    T.copy(E[bz, by * cfg.block_m, bo * cfg.block_n], e_shared)
                    T.copy(A_upper[bz, bx * cfg.block_k, bo * cfg.block_n], a_shared)
                    T.gemm(e_shared, a_shared, c_local, transpose_B=True)
                for i, k in T.Parallel(cfg.block_m, cfg.block_k):
                    ii = by * cfg.block_m + i
                    kk = bx * cfg.block_k + k
                    if (ii < M) and (kk < K):
                        w_old = T.cast(W[bz, ii, kk], T.float32)
                        W[bz, ii, kk] = T.cast(w_old + T.float32(alpha) * c_local[i, k], dtype)
        return main

    return kernel()

