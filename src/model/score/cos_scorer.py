import torch

from .scorer import Scorer


class CosScorer(Scorer):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.cosine_similarity(x, y)
