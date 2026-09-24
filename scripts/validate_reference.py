from __future__ import annotations

import argparse
import torch

from _bootstrap import bootstrap
bootstrap()

from ipc_tilelang.activations import get_activation
from ipc_tilelang.reference import ReferenceState, ipc_step, one_hot


def main():
    p = argparse.ArgumentParser(); p.add_argument("--depth", type=int, default=3); p.add_argument("--hidden", type=int, default=64); p.add_argument("--batch", type=int, default=8); p.add_argument("--steps", type=int, default=5)
    a = p.parse_args()
    torch.manual_seed(0)
    dims = (10,) + (a.hidden,) * a.depth + (28 * 28,)
    dtype = torch.float32; dev = torch.device("cpu")
    x_in = torch.randn((dims[-1], a.batch), dtype=dtype)
    y = one_hot(torch.arange(a.batch) % 10, 10, dtype=dtype)
    w = []
    for n, m in zip(dims[:-1], dims[1:]):
        t = torch.empty((n, m), dtype=dtype)
        torch.nn.init.kaiming_uniform_(t, a=5 ** 0.5); w.append(t)
    x = [torch.zeros((d, a.batch), dtype=dtype) for d in dims]; x[-1].copy_(x_in)
    act = get_activation("relu")
    for l in reversed(range(len(w))):
        pred = w[l] @ act.forward(x[l + 1])
        if l > 0: x[l].copy_(pred)
    x[0].copy_(y)
    st = ReferenceState(x=x, w=w)
    for i in range(a.steps):
        st, m = ipc_step(st, act=act, alpha=1e-4, gamma=0.5, fixed_low=y, fixed_high=x_in)
        print(i, float(m["energy"]), float(m["max_abs_error"]))
    print("Reference iPC trajectory computed successfully.")

if __name__ == "__main__": main()
