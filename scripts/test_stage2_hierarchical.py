from __future__ import annotations

import json
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ipc_tilelang import adaptive
try:
    from ipc_tilelang.hierarchical_policy import HierarchicalPolicy, kernel_signature
except ModuleNotFoundError as exc:
    if "tilelang_kernels" not in str(exc):
        raise
    from dataclasses import dataclass
    import types
    stub = types.ModuleType("ipc_tilelang.tilelang_kernels")
    @dataclass(frozen=True)
    class KernelConfig:
        block_m: int
        block_n: int
        block_k: int
        threads: int
        num_stages: int
        swizzle: bool
        swizzle_panel: int = 8
        shared_swizzle: bool = False
    stub.KernelConfig = KernelConfig
    sys.modules["ipc_tilelang.tilelang_kernels"] = stub
    from ipc_tilelang.hierarchical_policy import HierarchicalPolicy, kernel_signature


def main() -> None:
    cfg1 = {
        "block_m": 32, "block_n": 32, "block_k": 64,
        "threads": 64, "num_stages": 1, "swizzle": True,
        "swizzle_panel": 8, "shared_swizzle": False,
    }
    cfg2 = {
        "block_m": 16, "block_n": 32, "block_k": 32,
        "threads": 128, "num_stages": 1, "swizzle": False,
        "swizzle_panel": 8, "shared_swizzle": False,
    }
    kernels = [
        {"kind": "prediction", "shape": {"M": 10, "K": 64, "B": 128}, "activation": "relu", "dtype": "float16", "config": cfg1},
        {"kind": "prediction", "shape": {"M": 64, "K": 64, "B": 128}, "activation": "relu", "dtype": "float16", "config": cfg1},
    ]
    depth4_dims = [10, 64, 64, 64, 784]
    variant_kernels = [
        {**kernels[0], "config": cfg2},
        kernels[1],
    ]
    sig = kernel_signature(kernels)
    variant_sig = kernel_signature(variant_kernels)
    profile = [{
        "dims": depth4_dims,
        "batch": 128,
        "activation": "relu",
        "dtype": "float16",
        "kernel_signature": variant_sig,
        "selection_source": "rank2:prediction",
        "kernel_entries": variant_kernels,
    }]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "policy.json"
        p.write_text(json.dumps({
            "schema_version": 2,
            "stage": "2.1",
            "kernel_entries": kernels,
            "kernel_profiles": profile,
            "execution_entries": [{
                "dims": depth4_dims,
                "batch": 128,
                "activation": "relu",
                "dtype": "float16",
                "kernel_signature": variant_sig,
                "best": {"grid_z_max_layers": 3, "use_cuda_graph": True, "step_ms": 1.0},
            }],
        }, indent=2), encoding="utf-8")
        policy = HierarchicalPolicy(p)
        assert policy.lookup_kernel(kind="prediction", M=64, K=64, B=128, activation="relu", dtype="float16") is not None
        profiled = policy.lookup_profile_kernel(
            dims=tuple(depth4_dims), batch=128, kind="prediction", M=10, K=64,
            activation="relu", dtype="float16",
        )
        assert profiled is not None
        assert profiled.block_m == 16 and profiled.block_n == 32
        assert policy.selected_grid_z(
            dims=tuple(depth4_dims), batch=128, activation="relu", dtype="float16", kernel_sig=variant_sig, default=2
        ) == 3
        assert policy.selected_use_graph(
            dims=tuple(depth4_dims), batch=128, activation="relu", dtype="float16", kernel_sig=variant_sig, default=False
        )

    assert "v12" in adaptive.AdaptiveFiniteTuner.ENGINE
    assert not Path(__file__).resolve().parents[1].joinpath("src/ipc_tilelang_adaptive").exists()
    print("Stage 2.2 hierarchical policy + locked-v12 namespace test: PASS")


if __name__ == "__main__":
    main()
