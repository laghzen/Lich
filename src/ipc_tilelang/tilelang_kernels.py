from dataclasses import dataclass
from functools import lru_cache
import os

import torch


@dataclass(frozen=True)
class KernelConfig:
    block_m: int = 128
    block_n: int = 128
    block_k: int = 32
    threads: int = 128
    num_stages: int = 2
    swizzle: bool = True
    swizzle_panel: int = 8
    shared_swizzle: bool = False




def _jit_compile_flags():
    """Return only flags that are safe for TileLang 0.1.13 on Windows.

    TileLang 0.1.13 already propagates the process-level NVCC_CCBIN setting
    into the generated NVCC command. Passing a Windows absolute path through
    compile_flags is unsafe in this release because flag normalization can strip
    backslashes, producing a malformed second ``-ccbin=...``.

    The Windows bootstrap sets NVCC_CCBIN=cl.exe and places the selected native
    MSVC bin directory first in PATH. Return None so the generated command has
    exactly one valid ``-ccbin=cl.exe``.
    """
    return None

_WINDOWS_TOOLCHAIN_READY = False

def _ensure_windows_toolchain():
    global _WINDOWS_TOOLCHAIN_READY
    if os.name != "nt" or _WINDOWS_TOOLCHAIN_READY:
        return
    from .windows_toolchain import configure_windows_msvc
    configure_windows_msvc(quiet=True)
    _WINDOWS_TOOLCHAIN_READY = True


def _require_tilelang():
    # Kernel builders may be called directly (without scripts/run_cli.py).
    # Configure MSVC before the @tilelang.jit decorator is constructed so
    # _jit_compile_flags() sees IPC_MSVC_CL in the current process.
    _ensure_windows_toolchain()
    try:
        import tilelang  # noqa: F401
        import tilelang.language as T  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "TileLang is not installed. Run scripts/install_windows.ps1 in the project venv."
        ) from exc


def _act(x, name: str, T):
    if name == "relu":
        return T.max(x, 0)
    if name == "tanh":
        return T.tanh(x)
    if name == "sigmoid":
        return T.sigmoid(x)
    if name == "silu":
        s = T.sigmoid(x)
        return x * s
    if name == "gelu":
        u = T.float32(0.7978845608028654) * (x + T.float32(0.044715) * x * x * x)
        return T.float32(0.5) * x * (T.float32(1.0) + T.tanh(u))
    raise ValueError(f"Unsupported activation {name!r}")


def _dact(x, name: str, T):
    if name == "relu":
        return T.cast(x > 0, "float32")
    if name == "tanh":
        y = T.tanh(x)
        return T.float32(1.0) - y * y
    if name == "sigmoid":
        y = T.sigmoid(x)
        return y * (T.float32(1.0) - y)
    if name == "silu":
        s = T.sigmoid(x)
        return s * (T.float32(1.0) + x * (T.float32(1.0) - s))
    if name == "gelu":
        c = T.float32(0.044715)
        k = T.float32(0.7978845608028654)
        u = k * (x + c * x * x * x)
        t = T.tanh(u)
        du = k * (T.float32(1.0) + T.float32(3.0) * c * x * x)
        return T.float32(0.5) * (T.float32(1.0) + t) + T.float32(0.5) * x * (T.float32(1.0) - t * t) * du
    raise ValueError(f"Unsupported activation {name!r}")


def _decorate_shared(tilelang, T, a_shared, b_shared, enabled: bool):
    if not enabled:
        return
    try:
        T.annotate_layout({
            a_shared: tilelang.layout.make_swizzled_layout(a_shared, k_major=True),
            b_shared: tilelang.layout.make_swizzled_layout(b_shared, k_major=True),
        })
    except TypeError:
        T.annotate_layout({
            a_shared: tilelang.layout.make_swizzled_layout(a_shared),
            b_shared: tilelang.layout.make_swizzled_layout(b_shared),
        })


@lru_cache(maxsize=256)
def build_prediction_error(
    M: int,
    K: int,
    B: int,
    activation: str,
    dtype: str,
    cfg: KernelConfig,
):
    """Compile W[M,K] @ f(X_upper[K,B]) -> E[M,B]."""
    _require_tilelang()
    import tilelang
    import tilelang.language as T

    tail = (M % cfg.block_m) != 0 or (K % cfg.block_k) != 0 or (B % cfg.block_n) != 0

    @tilelang.jit(target={"kind": "cuda", "arch": "sm_86"}, compile_flags=_jit_compile_flags())
    def kernel():
        @T.prim_func
        def main(
            X_upper: T.Tensor((K, B), dtype),
            W: T.Tensor((M, K), dtype),
            X_lower: T.Tensor((M, B), dtype),
            E: T.Tensor((M, B), dtype),
        ):
            with T.Kernel(T.ceildiv(B, cfg.block_n), T.ceildiv(M, cfg.block_m), threads=cfg.threads) as (bx, by):
                # Keep raw activation input separate from the transformed GEMM tile.
                # In-place writes to a pipelined shared buffer are rejected by
                # PipelinePlanning; two buffers preserve the producer/consumer DAG.
                a_raw = T.alloc_shared((cfg.block_k, cfg.block_n), dtype)
                a_shared = T.alloc_shared((cfg.block_k, cfg.block_n), dtype)
                w_shared = T.alloc_shared((cfg.block_m, cfg.block_k), dtype)
                c_local = T.alloc_fragment((cfg.block_m, cfg.block_n), T.float32)
                _decorate_shared(tilelang, T, w_shared, a_shared, cfg.shared_swizzle)
                if cfg.swizzle:
                    T.use_swizzle(panel_size=cfg.swizzle_panel)
                T.clear(c_local)
                for ko in T.Pipelined(T.ceildiv(K, cfg.block_k), num_stages=cfg.num_stages):
                    # T.copy is the pipeline producer. Legalization supplies boundary
                    # predicates for non-multiple shapes, so no custom Parallel copy
                    # helper is needed here.
                    T.copy(X_upper[ko * cfg.block_k, bx * cfg.block_n], a_raw)
                    T.copy(W[by * cfg.block_m, ko * cfg.block_k], w_shared)
                    # Transform raw input into a distinct shared buffer. This is a
                    # single write to a_shared, avoiding overlapping producer writes.
                    for k, b in T.Parallel(cfg.block_k, cfg.block_n):
                        gi = ko * cfg.block_k + k
                        gj = bx * cfg.block_n + b
                        if (gi < K) and (gj < B):
                            a_shared[k, b] = _act(a_raw[k, b], activation, T)
                        else:
                            # Padding must stay zero even for activations where f(0) != 0
                            # (sigmoid/tanh), otherwise tail elements would contribute to GEMM.
                            a_shared[k, b] = T.cast(0, dtype)
                    T.gemm(w_shared, a_shared, c_local, transpose_B=False)
                for i, j in T.Parallel(cfg.block_m, cfg.block_n):
                    ii = by * cfg.block_m + i
                    jj = bx * cfg.block_n + j
                    if (ii < M) and (jj < B):
                        E[ii, jj] = T.cast(X_lower[ii, jj], dtype) - T.cast(c_local[i, j], dtype)

        return main

    return kernel()


@lru_cache(maxsize=256)
def build_inference_update(
    M: int,
    K: int,
    B: int,
    activation: str,
    dtype: str,
    cfg: KernelConfig,
    gamma: float,
    save_activation: bool = False,
):
    """Compile X += gamma*(-E + f'(X)*W.T@E_lower), optionally materializing f(X)."""
    _require_tilelang()
    import tilelang
    import tilelang.language as T

    tail = (M % cfg.block_m) != 0 or (K % cfg.block_k) != 0 or (B % cfg.block_n) != 0

    @tilelang.jit(target={"kind": "cuda", "arch": "sm_86"}, compile_flags=_jit_compile_flags())
    def kernel():
        if save_activation:
            @T.prim_func
            def main(
                X: T.Tensor((M, B), dtype),
                E: T.Tensor((M, B), dtype),
                W_lower: T.Tensor((K, M), dtype),
                E_lower: T.Tensor((K, B), dtype),
                A_out: T.Tensor((M, B), dtype),
            ):
                with T.Kernel(T.ceildiv(B, cfg.block_n), T.ceildiv(M, cfg.block_m), threads=cfg.threads) as (bx, by):
                    e_shared = T.alloc_shared((cfg.block_k, cfg.block_n), dtype)
                    # W_lower is physically [K, M]. Store a KxBM tile and
                    # transpose it in T.gemm, giving [M, K] @ [K, B].
                    w_shared = T.alloc_shared((cfg.block_k, cfg.block_m), dtype)
                    c_local = T.alloc_fragment((cfg.block_m, cfg.block_n), T.float32)
                    _decorate_shared(tilelang, T, w_shared, e_shared, cfg.shared_swizzle)
                    if cfg.swizzle:
                        T.use_swizzle(panel_size=cfg.swizzle_panel)
                    T.clear(c_local)
                    for ko in T.Pipelined(T.ceildiv(K, cfg.block_k), num_stages=cfg.num_stages):
                        T.copy(E_lower[ko * cfg.block_k, bx * cfg.block_n], e_shared)
                        T.copy(W_lower[ko * cfg.block_k, by * cfg.block_m], w_shared)
                        T.gemm(w_shared, e_shared, c_local, transpose_A=True, transpose_B=False)
                    for i, j in T.Parallel(cfg.block_m, cfg.block_n):
                        ii = by * cfg.block_m + i
                        jj = bx * cfg.block_n + j
                        if (ii < M) and (jj < B):
                            x = T.cast(X[ii, jj], T.float32)
                            e = T.cast(E[ii, jj], T.float32)
                            x_new = x + T.float32(gamma) * (-e + _dact(x, activation, T) * c_local[i, j])
                            X[ii, jj] = T.cast(x_new, dtype)
                            A_out[ii, jj] = T.cast(_act(x_new, activation, T), dtype)
            return main
        else:
            @T.prim_func
            def main(
                X: T.Tensor((M, B), dtype),
                E: T.Tensor((M, B), dtype),
                W_lower: T.Tensor((K, M), dtype),
                E_lower: T.Tensor((K, B), dtype),
            ):
                with T.Kernel(T.ceildiv(B, cfg.block_n), T.ceildiv(M, cfg.block_m), threads=cfg.threads) as (bx, by):
                    e_shared = T.alloc_shared((cfg.block_k, cfg.block_n), dtype)
                    # W_lower is physically [K, M]. Store a KxBM tile and
                    # transpose it in T.gemm, giving [M, K] @ [K, B].
                    w_shared = T.alloc_shared((cfg.block_k, cfg.block_m), dtype)
                    c_local = T.alloc_fragment((cfg.block_m, cfg.block_n), T.float32)
                    _decorate_shared(tilelang, T, w_shared, e_shared, cfg.shared_swizzle)
                    if cfg.swizzle:
                        T.use_swizzle(panel_size=cfg.swizzle_panel)
                    T.clear(c_local)
                    for ko in T.Pipelined(T.ceildiv(K, cfg.block_k), num_stages=cfg.num_stages):
                        T.copy(E_lower[ko * cfg.block_k, bx * cfg.block_n], e_shared)
                        T.copy(W_lower[ko * cfg.block_k, by * cfg.block_m], w_shared)
                        T.gemm(w_shared, e_shared, c_local, transpose_A=True, transpose_B=False)
                    for i, j in T.Parallel(cfg.block_m, cfg.block_n):
                        ii = by * cfg.block_m + i
                        jj = bx * cfg.block_n + j
                        if (ii < M) and (jj < B):
                            x = T.cast(X[ii, jj], T.float32)
                            e = T.cast(E[ii, jj], T.float32)
                            update = T.float32(gamma) * (-e + _dact(x, activation, T) * c_local[i, j])
                            X[ii, jj] = T.cast(x + update, dtype)
            return main

    return kernel()


@lru_cache(maxsize=256)
def build_weight_update(
    M: int,
    K: int,
    B: int,
    activation: str,
    dtype: str,
    cfg: KernelConfig,
    alpha: float,
    recompute_activation: bool = True,
):
    """Compile W += alpha * E @ A.T, where A is f(X_upper) or a saved activation."""
    _require_tilelang()
    import tilelang
    import tilelang.language as T

    tail = (M % cfg.block_m) != 0 or (K % cfg.block_k) != 0 or (B % cfg.block_n) != 0

    @tilelang.jit(target={"kind": "cuda", "arch": "sm_86"}, compile_flags=_jit_compile_flags())
    def kernel():
        if recompute_activation:
            @T.prim_func
            def main(
                W: T.Tensor((M, K), dtype),
                E: T.Tensor((M, B), dtype),
                X_upper: T.Tensor((K, B), dtype),
            ):
                with T.Kernel(T.ceildiv(K, cfg.block_k), T.ceildiv(M, cfg.block_m), threads=cfg.threads) as (bx, by):
                    e_shared = T.alloc_shared((cfg.block_m, cfg.block_n), dtype)
                    a_raw = T.alloc_shared((cfg.block_k, cfg.block_n), dtype)
                    a_shared = T.alloc_shared((cfg.block_k, cfg.block_n), dtype)
                    c_local = T.alloc_fragment((cfg.block_m, cfg.block_k), T.float32)
                    _decorate_shared(tilelang, T, e_shared, a_shared, cfg.shared_swizzle)
                    if cfg.swizzle:
                        T.use_swizzle(panel_size=cfg.swizzle_panel)
                    T.clear(c_local)
                    for bo in T.Pipelined(T.ceildiv(B, cfg.block_n), num_stages=cfg.num_stages):
                        T.copy(E[by * cfg.block_m, bo * cfg.block_n], e_shared)
                        T.copy(X_upper[bx * cfg.block_k, bo * cfg.block_n], a_raw)
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
                            w_old = T.cast(W[ii, kk], T.float32)
                            W[ii, kk] = T.cast(w_old + T.float32(alpha) * c_local[i, k], dtype)
            return main
        else:
            @T.prim_func
            def main(
                W: T.Tensor((M, K), dtype),
                E: T.Tensor((M, B), dtype),
                A_upper: T.Tensor((K, B), dtype),
            ):
                with T.Kernel(T.ceildiv(K, cfg.block_k), T.ceildiv(M, cfg.block_m), threads=cfg.threads) as (bx, by):
                    e_shared = T.alloc_shared((cfg.block_m, cfg.block_n), dtype)
                    a_shared = T.alloc_shared((cfg.block_k, cfg.block_n), dtype)
                    c_local = T.alloc_fragment((cfg.block_m, cfg.block_k), T.float32)
                    _decorate_shared(tilelang, T, e_shared, a_shared, cfg.shared_swizzle)
                    if cfg.swizzle:
                        T.use_swizzle(panel_size=cfg.swizzle_panel)
                    T.clear(c_local)
                    for bo in T.Pipelined(T.ceildiv(B, cfg.block_n), num_stages=cfg.num_stages):
                        T.copy(E[by * cfg.block_m, bo * cfg.block_n], e_shared)
                        T.copy(A_upper[bx * cfg.block_k, bo * cfg.block_n], a_shared)
                        T.gemm(e_shared, a_shared, c_local, transpose_B=True)
                    for i, k in T.Parallel(cfg.block_m, cfg.block_k):
                        ii = by * cfg.block_m + i
                        kk = bx * cfg.block_k + k
                        if (ii < M) and (kk < K):
                            w_old = T.cast(W[ii, kk], T.float32)
                            W[ii, kk] = T.cast(w_old + T.float32(alpha) * c_local[i, k], dtype)
            return main

    return kernel()


def kernel_call(kernel, *tensors: torch.Tensor) -> None:
    kernel(*tensors)
