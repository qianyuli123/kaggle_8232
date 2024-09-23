from typing import Callable
from abc import ABC, abstractmethod

import torch


class Scorer(
    ABC,
    torch.nn.Module,
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
):
    @abstractmethod
    def forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...
