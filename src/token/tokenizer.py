from abc import ABC, abstractmethod
from typing import Callable

import torch


class Tokenizer(ABC, Callable[[str], torch.Tensor]):
    SEP: str
    PAD: str

    def __len__(self) -> int: ...

    @abstractmethod
    def tokenize(self, seq: str) -> list[str]: ...

    @abstractmethod
    def __call__(self, seq: str) -> torch.Tensor: ...
