from collections.abc import Sequence
from typing import Generator
from torch.utils.data import Dataset


from ..data import Question, Answer, Misconception


class AnswerDataset(Dataset[tuple[Answer, Misconception, int]]):
    """
    It is a generator that yields tuples of (answer, misconception, label).

    label: 1 if answer is relevant to misconception, 0 otherwise
    """

    def __init__(
        self,
        questions: Sequence[Question],
        split_ratio: tuple[int, int, int] = (0.8, 0.1, 0.1),
    ) -> None:
        assert all(map(lambda x: x >= 0, split_ratio)), "Split must be positive"
        assert sum(split_ratio) == 1, "Split must sum to 1"

        self.data = list[tuple[Answer, Misconception, int]]()
        for question in questions:
            for answer in question.answers:
                if answer.misconception is not None:
                    self.data.append((answer, answer.misconception, 1))
                    for other in filter(lambda a: a != answer, question.answers):
                        if other.misconception is not None:
                            self.data.append((answer, other.misconception, 0))

        train_ratio, val_ratio, _ = split_ratio
        total_size = len(self.data)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        self.train = self.data[:train_size]
        self.val = self.data[train_size : train_size + val_size]
        self.test = self.data[train_size + val_size :]
