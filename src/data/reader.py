import os
import pandas as pd

from .question import Question, QuestionFactory


class DataReader:
    def __init__(self, dir_path: str) -> None:
        self.dir_path = dir_path
        self.train_data = list[Question]()
        self.test_data = list[Question]()

    def read(self) -> tuple[list[Question], list[Question]]:
        self.train_df = pd.read_csv(os.path.join(self.dir_path, "train.csv"))
        self.test_df = pd.read_csv(os.path.join(self.dir_path, "test.csv"))
        self.misconception_mapping = dict(
            pd.read_csv(os.path.join(self.dir_path, "misconception_mapping.csv")).values
        )
        self.fact = QuestionFactory(misconception_mapping=self.misconception_mapping)
        for id, item in self.train_df.iterrows():
            self.train_data.append(self.fact.create(item))
        for id, item in self.test_df.iterrows():
            self.test_data.append(self.fact.create(item))
        return self.train_data, self.test_data
