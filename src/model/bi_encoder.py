from collections.abc import Sequence
from typing import Callable
import torch

from .encode import Encoder
from .score import Scorer


class BiEncoder(
    torch.nn.Module,
    Callable[[Sequence[str], Sequence[str]], torch.Tensor],
):
    def __init__(
        self,
        q_encoder: Encoder[str],
        d_encoder: Encoder[str],
        scorer: Scorer,
    ) -> None:
        super(BiEncoder, self).__init__()
        self.q_encoder = q_encoder
        self.d_encoder = d_encoder
        self.scorer = scorer

    def forward(
        self,
        q_batch: Sequence[str],
        d_batch: Sequence[str],
    ) -> torch.Tensor:
        assert len(q_batch) == len(
            d_batch
        ), "q_batch and d_batch must have the same length"
        return self.scorer(self.q_encoder(q_batch), self.d_encoder(d_batch))
