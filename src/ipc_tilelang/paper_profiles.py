from __future__ import annotations

PAPER = {
    "discriminative_efficiency": {
        "hidden": 64,
        "depths": [3, 4, 6],
        "alpha": 1e-4,
        "gamma": 0.5,
        "dataset": "MNIST",
        "note": "Supplement D.1 specifies the 64-wide, depth 3/4/6 setting and alpha/gamma; dataset handling for Fig. 8 is not fully explicit.",
    },
    "mnist_mlp_table1": {
        "dims": [784, 64, 64, 10],
        "reported_ipc_accuracy": 98.54,
        "reported_ipc_std": 0.86,
        "reported_bp_accuracy": 98.26,
        "reported_bp_std": 0.12,
        "reported_pc_accuracy": 98.55,
        "reported_pc_std": 0.14,
        "note": "Main paper reports the MLP on MNIST result; exact training details are not all specified in the paper.",
    },
}
