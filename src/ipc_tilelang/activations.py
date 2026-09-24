from __future__ import annotations

from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class ActivationSpec:
    name: str
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.name == "relu": return torch.relu(x)
        if self.name == "tanh": return torch.tanh(x)
        if self.name == "sigmoid": return torch.sigmoid(x)
        if self.name == "silu": return torch.nn.functional.silu(x)
        if self.name == "gelu": return torch.nn.functional.gelu(x, approximate="tanh")
        raise ValueError(f"Unsupported activation: {self.name}")
    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        if self.name == "relu": return (x > 0).to(x.dtype)
        if self.name == "tanh":
            y = torch.tanh(x); return 1 - y * y
        if self.name == "sigmoid":
            y = torch.sigmoid(x); return y * (1 - y)
        if self.name == "silu":
            s = torch.sigmoid(x); return s * (1 + x * (1 - s))
        if self.name == "gelu":
            c, k = 0.044715, 0.7978845608028654
            u = k * (x + c * x * x * x); t = torch.tanh(u)
            du = k * (1 + 3 * c * x * x)
            return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * du
        raise ValueError(f"Unsupported activation: {self.name}")

ACTIVATIONS = {n: ActivationSpec(n) for n in ("relu", "tanh", "sigmoid", "silu", "gelu")}

def get_activation(name: str) -> ActivationSpec:
    try: return ACTIVATIONS[name.lower()]
    except KeyError as exc: raise ValueError(f"Unknown activation {name!r}") from exc
