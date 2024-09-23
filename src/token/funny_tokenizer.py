import torch
from .tokenizer import Tokenizer


class FunnyTokenizer(Tokenizer):
    def __init__(self, max_len=128) -> None:
        self.corpus = dict[str, int]()
        self.max_len = max_len
        self.SEP = "[SEP]"
        self.PAD = "[PAD]"
        self.add(self.SEP)
        self.add(self.PAD)

    def __len__(self) -> int:
        return len(self.corpus)

    def add(self, token: str) -> int:
        token = token.lower()
        if token not in self.corpus:
            self.corpus[token] = len(self.corpus)
        return self.corpus[token]

    def get(self, token: str) -> int:
        token = token.lower()
        return self.corpus.get(token, None)

    def tokenize(self, seq: str) -> list[int]:
        return seq.split()

    def __call__(self, seq: str, padding=True, truncate=True) -> torch.Tensor:
        tokens = self.tokenize(seq)
        token_ids = []
        for token in tokens:
            token_ids.append(self.add(token))
        if truncate:
            token_ids = token_ids[: self.max_len]
        if padding:
            token_ids.extend([self.get(self.PAD)] * (self.max_len - len(token_ids)))
        return torch.tensor(token_ids, dtype=torch.long)
