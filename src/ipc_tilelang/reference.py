from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import torch
from .activations import ActivationSpec

@dataclass
class ReferenceState:
    x: list[torch.Tensor]
    w: list[torch.Tensor]

def one_hot(labels: torch.Tensor, num_classes: int, *, dtype: torch.dtype) -> torch.Tensor:
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).to(dtype=dtype).T.contiguous()

def init_weights(dims: Sequence[int], *, dtype: torch.dtype, device: torch.device, seed: int = 0) -> list[torch.Tensor]:
    g = torch.Generator(device=device); g.manual_seed(seed)
    out = []
    for n_l, n_u in zip(dims[:-1], dims[1:]):
        w = torch.empty((n_l, n_u), device=device, dtype=dtype)
        torch.nn.init.kaiming_uniform_(w, a=5 ** 0.5, generator=g)
        out.append(w)
    return out

def initialize_state(x_input: torch.Tensor, x_target: torch.Tensor, dims: Sequence[int], *, dtype: torch.dtype, device: torch.device, seed: int = 0) -> ReferenceState:
    if x_input.ndim != 2 or x_target.ndim != 2:
        raise ValueError("Inputs must be [features,batch]")
    w = init_weights(dims, dtype=dtype, device=device, seed=seed)
    x = [torch.zeros((d, x_input.shape[1]), device=device, dtype=dtype) for d in dims]
    x[0].copy_(x_target); x[-1].copy_(x_input)
    # Reasonable fast initialization: hidden states follow a forward prediction from the fixed input.
    for l in reversed(range(len(w))):
        x[l].copy_(w[l] @ torch.zeros_like(x[l + 1])) if l == len(w) - 1 else None
    return ReferenceState(x=x, w=w)

def prediction_errors(x: Sequence[torch.Tensor], w: Sequence[torch.Tensor], act: ActivationSpec):
    a = [None] * len(x); e = [torch.empty_like(xi) for xi in x]
    for l in range(len(w)):
        a[l + 1] = act.forward(x[l + 1])
        e[l].copy_(x[l] - w[l] @ a[l + 1])
    return e, [v if v is not None else x[0] for v in a]

def ipc_step(state: ReferenceState, *, act: ActivationSpec, alpha: float, gamma: float, fixed_low: torch.Tensor, fixed_high: torch.Tensor, update_weights: bool = True):
    L = len(state.w)
    e, a = prediction_errors(state.x, state.w, act)
    # State updates use x/W/e from the same logical time t.
    new_x = [v.clone() for v in state.x]
    for l in range(1, L):
        signal = state.w[l - 1].T @ e[l - 1]
        new_x[l] = state.x[l] + gamma * (-e[l] + act.derivative(state.x[l]) * signal)
    new_x[0] = fixed_low; new_x[L] = fixed_high
    new_w = [v.clone() for v in state.w]
    if update_weights:
        # Algorithm 1 updates x first and theta second; in this implementation the
        # weight update therefore consumes the just-updated x[l+1].
        for l in range(L):
            a_new = act.forward(new_x[l + 1])
            new_w[l] = state.w[l] + alpha * (e[l] @ a_new.T)
    metrics = {"energy": sum((q.float() ** 2).mean() for q in e[:-1]).detach(), "max_abs_error": max(q.float().abs().max() for q in e[:-1]).detach()}
    return ReferenceState(x=new_x, w=new_w), metrics
