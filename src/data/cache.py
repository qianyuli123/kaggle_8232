from typing import TypeVar, Generic, Optional

V = TypeVar("V")


class Cache(Generic[V]):
    def __init__(self, F) -> None:
        self.data: dict[str, V] = {}
        self.F = F

    def __getitem__(self, key: int | str) -> Optional[V]:
        return self.data.get(key)

    def get(self, key: int, *args, **kwargs) -> V:
        if key not in self.data:
            self.data[key] = self.F(*args, **kwargs)
        return self.data[key]
