from collections.abc import Sequence
from typing import Optional

import numpy as np


class Subject:
    def __init__(self, id: int, name: str) -> None:
        self.id = id
        self.name = name

    def __repr__(self) -> str:
        return f"Subject(id={self.id}, name={self.name})"


class Misconception:
    def __init__(self, id: int, name: str) -> None:
        self.id = id
        self.name = name

    def __repr__(self) -> str:
        return f"Misconception(id={self.id}, name={self.name})"


class Construct:
    def __init__(self, id: int, name: str) -> None:
        self.id = id
        self.name = name

    def __repr__(self) -> str:
        return f"Construct(id={self.id}, name={self.name})"


class Answer:
    def __init__(
        self,
        text: str,
        misconception: Optional[Misconception] = None,
    ) -> None:
        self.text = text
        self.misconception = misconception

    def __repr__(self) -> str:
        return f"Answer(text={self.text}, misconception={self.misconception})"


class Question:
    def __init__(
        self,
        id: int,
        text: str,
        construct: Construct,
        answers: Sequence[Answer],
        subject: Subject,
        answer: int,
    ):
        self.id = id
        self.text = text
        self.construct = construct
        assert len(answers) == 4, "Question must have 4 answers"
        self.answers = answers
        self.subject = subject
        assert answer in range(len(answers)), "Answer must be in range of answers"
        self.answer = answer

    def __repr__(self) -> str:
        return f"Question(id={self.id}, text={self.text}, construct={self.construct}, answers={self.answers}, subject={self.subject}, answer={self.answer})"


class Answer(Answer):
    question: Question


from .cache import Cache


class QuestionFactory:
    def __init__(self, misconception_mapping: dict[int, str]) -> None:
        self.construct_pool = Cache[Construct](Construct)
        self.misconception_pool = Cache[Misconception](Misconception)
        self.subject_pool = Cache[Subject](Subject)
        self.misconception_mapping = misconception_mapping

    def create(self, dic: dict) -> Question:
        for key in (
            "QuestionId",
            "QuestionText",
            "ConstructId",
            "ConstructName",
            "SubjectId",
            "SubjectName",
            "CorrectAnswer",
            "AnswerAText",
            "AnswerBText",
            "AnswerCText",
            "AnswerDText",
        ):
            assert key in dic, f"Missing key {key}"

        question = Question(
            id=dic["QuestionId"],
            text=dic["QuestionText"],
            construct=self.construct_pool.get(
                key=dic["ConstructId"],
                id=dic["ConstructId"],
                name=dic["ConstructName"],
            ),
            subject=self.subject_pool.get(
                key=dic["SubjectId"],
                id=dic["SubjectId"],
                name=dic["SubjectName"],
            ),
            answers=tuple(
                Answer(
                    text=dic[f"Answer{key}Text"],
                    misconception=(
                        self.misconception_pool.get(
                            key=dic[f"Misconception{key}Id"],
                            id=dic[f"Misconception{key}Id"],
                            name=self.misconception_mapping.get(
                                dic[f"Misconception{key}Id"]
                            ),
                        )
                        if f"Misconception{key}Id" in dic
                        and not np.isnan(dic[f"Misconception{key}Id"])
                        else None
                    ),
                )
                for key in ("A", "B", "C", "D")
            ),
            answer=("A", "B", "C", "D").index(dic["CorrectAnswer"]),
        )
        for answer in question.answers:
            answer.question = question
        return question
