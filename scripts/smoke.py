from __future__ import annotations
import argparse
import torch
from _bootstrap import bootstrap
bootstrap()

from ipc_tilelang.tilelang_kernels import KernelConfig, build_prediction_error, build_inference_update, build_weight_update


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=64)
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--B", type=int, default=128)
    p.add_argument("--activation", default="relu")
    a = p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    if torch.cuda.get_device_capability() != (8, 6): raise RuntimeError("Expected SM86")
    dev = torch.device("cuda")
    cfg = KernelConfig(64, 128, 16, 128, 2, True, 8, False)
    x_up = torch.randn((a.K, a.B), device=dev, dtype=torch.float16)
    w = torch.randn((a.M, a.K), device=dev, dtype=torch.float16) * 0.05
    x_lo = torch.randn((a.M, a.B), device=dev, dtype=torch.float16)
    e = torch.empty_like(x_lo)
    build_prediction_error(a.M, a.K, a.B, a.activation, "float16", cfg)(x_up, w, x_lo, e)
    torch.cuda.synchronize()
    ref_e = x_lo - w @ torch.relu(x_up)
    torch.testing.assert_close(e, ref_e, rtol=2e-2, atol=2e-2)

    x = x_lo.clone()
    e_low = torch.randn_like(x_up)
    # W_{l-1} maps M hidden units to K lower units: physical shape [K, M].
    infer_w = torch.randn((a.K, a.M), device=dev, dtype=torch.float16) * 0.05
    build_inference_update(a.M, a.K, a.B, a.activation, "float16", cfg, 0.5)(x, e, infer_w, e_low)
    torch.cuda.synchronize()
    ref_x = x_lo + 0.5 * (-ref_e + (x_lo > 0).to(torch.float16) * (infer_w.T @ e_low))
    torch.testing.assert_close(x, ref_x, rtol=3e-2, atol=3e-2)

    w2 = w.clone(); alpha = 1e-4
    build_weight_update(a.M, a.K, a.B, a.activation, "float16", cfg, alpha, True)(w2, e, x_up)
    torch.cuda.synchronize()
    ref_w = w + alpha * e.float() @ torch.relu(x_up).float().T
    torch.testing.assert_close(w2.float(), ref_w, rtol=2e-2, atol=2e-2)

    # Exercise the actual tail-safe path used by the MNIST paper shape (10-class output).
    m2, k2, b2 = 10, 64, 128
    xu2 = torch.randn((k2, b2), device=dev, dtype=torch.float16)
    wsmall = torch.randn((m2, k2), device=dev, dtype=torch.float16) * 0.05
    xsmall = torch.randn((m2, b2), device=dev, dtype=torch.float16)
    esmall = torch.empty_like(xsmall)
    build_prediction_error(m2, k2, b2, a.activation, "float16", cfg)(xu2, wsmall, xsmall, esmall)
    torch.cuda.synchronize()
    torch.testing.assert_close(esmall, xsmall - wsmall @ torch.relu(xu2), rtol=3e-2, atol=3e-2)

    # Exercise weight-update tail M=10 with non-square dimensions.
    wsmall2 = wsmall.clone()
    ref_wsmall = wsmall.float() + alpha * esmall.float() @ torch.relu(xu2).float().T
    build_weight_update(m2, k2, b2, a.activation, "float16", cfg, alpha, True)(wsmall2, esmall, xu2)
    torch.cuda.synchronize()
    torch.testing.assert_close(wsmall2.float(), ref_wsmall, rtol=2e-2, atol=2e-2)

    w_saved = w.clone()
    a_saved = torch.relu(x_up).contiguous()
    build_weight_update(a.M, a.K, a.B, a.activation, "float16", cfg, alpha, False)(w_saved, e, a_saved)
    torch.cuda.synchronize()
    torch.testing.assert_close(w_saved.float(), ref_w, rtol=2e-2, atol=2e-2)
    print("SM86 TileLang kernel smoke test: PASS (regular + tail-safe + save-A paths)")


if __name__ == "__main__": main()
