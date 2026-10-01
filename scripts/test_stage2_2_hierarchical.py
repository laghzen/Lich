from __future__ import annotations

import json
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# This archive is an overlay: the real Lich repository supplies the runtime modules.
# For the pure policy/top-K test, stub only those runtime-only imports.
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

if "_bootstrap" not in sys.modules:
    boot = types.ModuleType("_bootstrap")
    boot.bootstrap = lambda: None
    sys.modules["_bootstrap"] = boot
if "ipc_tilelang.tilelang_kernels" not in sys.modules:
    tlk = types.ModuleType("ipc_tilelang.tilelang_kernels")
    tlk.KernelConfig = KernelConfig
    sys.modules["ipc_tilelang.tilelang_kernels"] = tlk
if "ipc_tilelang.trainer" not in sys.modules:
    tr = types.ModuleType("ipc_tilelang.trainer")
    tr.IPCConfig = object
    tr.TileLangIPC = object
    sys.modules["ipc_tilelang.trainer"] = tr
if "ipc_tilelang.layer_batching" not in sys.modules:
    lb = types.ModuleType("ipc_tilelang.layer_batching")
    lb.find_internal_square_groups = lambda *args, **kwargs: ()
    sys.modules["ipc_tilelang.layer_batching"] = lb

from hierarchical_autotune import ResultTuningTable, collect_topk_candidates_from_run, _best_measurement_per_config
from ipc_tilelang.hierarchical_policy import HierarchicalPolicy, kernel_signature


def main() -> None:
    cfg1 = KernelConfig(16, 32, 16, 64, 3, False, 8, False)
    cfg2 = KernelConfig(32, 32, 64, 64, 1, True, 8, False)
    cfg3 = KernelConfig(16, 64, 16, 64, 3, False, 8, False)
    rows = [
        {"kind": "prediction", "shape": {"M": 64, "K": 64, "B": 128}, "activation": "relu", "dtype": "float16", "config": asdict(cfg1)},
    ]
    table = ResultTuningTable(rows)
    run = {
        "results": [{
            **rows[0],
            "fresh_observations": [
                {"config": asdict(cfg1), "fidelity": 0, "latency_ms": 0.030, "successful": True},
                {"config": asdict(cfg2), "fidelity": 0, "latency_ms": 0.015, "successful": True},
                {"config": asdict(cfg2), "fidelity": 1, "latency_ms": 0.017, "successful": True},
                {"config": asdict(cfg3), "fidelity": 0, "latency_ms": 0.020, "successful": True},
                {"config": asdict(cfg3), "fidelity": 1, "latency_ms": 0.019, "successful": True},
            ],
        }]
    }
    # Equal latency/fidelity must be deterministically sortable without comparing KernelConfig objects.
    tied = [
        {"config": asdict(cfg1), "fidelity": 1, "latency_ms": 0.020, "successful": True},
        {"config": asdict(cfg2), "fidelity": 1, "latency_ms": 0.020, "successful": True},
    ]
    ranked_tied = _best_measurement_per_config(tied)
    assert len(ranked_tied) == 2

    topk = collect_topk_candidates_from_run(table, run, topk=3)
    assert topk[table.kernel_rows()[0]["kind"], 64, 64, 128, "relu", "float16"] == [cfg2, cfg3, cfg1]

    override = table.copy_with_overrides({(
        "prediction", 64, 64, 128, "relu", "float16"
    ): cfg2})
    sig = kernel_signature(override.kernel_rows())
    assert sig != kernel_signature(table.kernel_rows())

    dims = (10, 64, 64, 64, 784)
    profile = {
        "dims": list(dims), "batch": 128, "activation": "relu", "dtype": "float16",
        "kernel_signature": sig,
        "kernel_entries": override.kernel_rows(),
    }
    payload = {
        "schema_version": 3,
        "stage": "2.2",
        "kernel_profiles": [profile],
        "execution_entries": [{
            "dims": list(dims), "batch": 128, "activation": "relu", "dtype": "float16",
            "kernel_signature": sig,
            "best": {"grid_z_max_layers": 3, "recompute_activation": False, "use_cuda_graph": True, "step_ms": 0.1},
        }],
    }
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "policy.json"
        p.write_text(json.dumps(payload), encoding="utf-8")
        policy = HierarchicalPolicy(p)
        assert policy.selected_grid_z(
            dims=dims, batch=128, activation="relu", dtype="float16", kernel_sig=sig, default=2
        ) == 3
        assert policy.selected_use_graph(
            dims=dims, batch=128, activation="relu", dtype="float16", kernel_sig=sig, default=False
        ) is True
        assert policy.selected_recompute_activation(
            dims=dims, batch=128, activation="relu", dtype="float16", kernel_sig=sig, default=True
        ) is False

    assert not ROOT.joinpath("src/ipc_tilelang_adaptive").exists()
    print("Stage 2.2.1 fresh-topK + joint-policy unit test: PASS")


if __name__ == "__main__":
    main()
