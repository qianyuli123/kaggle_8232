import torch
from collections.abc import Sequence

from .encoder import Encoder


class RNNEncoder(Encoder[str]):
    def __init__(self, dim: int, voc_len: int, max_len=128, num_layers=1) -> None:
        super(RNNEncoder, self).__init__()
        self.dim = dim
        self.input_size = max_len
        self.hidden_size = 4 * dim
        self.num_layers = num_layers
        self.embedding = torch.nn.Embedding(num_embeddings=voc_len, embedding_dim=dim)
        self.lstm = torch.nn.LSTM(
            num_layers=self.num_layers,
            input_size=self.dim,
            hidden_size=self.hidden_size,
            bidirectional=True,
        )
        self.fc = torch.nn.Linear(2 * self.hidden_size, dim)

    """
    @params:
    batch: (batch_size, seq_len, dim)

    @returns: (batch_size, dim)
    """

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        seq_len = batch.size(1)
        # (batch_size, seq_len, dim)
        x = self.embedding(batch)

        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(
            2 * self.num_layers,
            seq_len,
            self.hidden_size,
        ).to(batch.device)
        c0 = torch.zeros(
            2 * self.num_layers,
            seq_len,
            self.hidden_size,
        ).to(batch.device)
        # (batch_size, seq_len, 2 * hidden_size)
        y, (hn, cn) = self.lstm(x, (h0, c0))  # 将 x 转换为 3D 张量
        # (batch_size, 2 * hidden_size)
        y = y[:, -1, :]  # 取最后一个时间步的输出
        # (batch_size, dim)
        y = self.fc(y)
        return y
