from typing import Generic, TypeVar, Callable

import torch
from abc import ABC, abstractmethod
from collections.abc import Sequence

V = TypeVar("V")


class Encoder(
    ABC,
    Generic[V],
    torch.nn.Module,
):
    @abstractmethod
    def forward(self, batch: Sequence[V]) -> torch.Tensor: ...
