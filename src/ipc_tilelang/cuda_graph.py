from __future__ import annotations

import torch


class IPCGraphRunner:
    """Capture a fixed count of initialized iPC steps.

    TileLang kernels are compiled once; CUDA Graph then removes the Python-side
    launch overhead from repeated hot-step execution. State/error tensor
    addresses remain stable because the trainer reuses its buffers.
    """

    def __init__(self, model, steps: int):
        if steps < 1:
            raise ValueError("steps must be >= 1")
        self.model = model
        self.steps = steps
        self.graph = torch.cuda.CUDAGraph()
        self._captured = False

    def capture(self) -> None:
        torch.cuda.synchronize()
        self.model.precompile()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            for _ in range(self.steps):
                self.model.step_initialized(collect_metrics=False)
        torch.cuda.synchronize()
        self._captured = True

    def replay(self) -> None:
        if not self._captured:
            raise RuntimeError("Graph has not been captured")
        self.graph.replay()
