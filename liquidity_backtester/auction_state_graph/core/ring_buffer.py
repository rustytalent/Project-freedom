"""Small bounded history container."""

from __future__ import annotations

from collections import deque
from typing import Generic, Iterable, Iterator, TypeVar

T = TypeVar("T")


class RingBuffer(Generic[T]):
    def __init__(self, maxlen: int):
        if maxlen <= 0:
            raise ValueError("maxlen must be positive")
        self._items: deque[T] = deque(maxlen=maxlen)

    @property
    def maxlen(self) -> int:
        return int(self._items.maxlen or 0)

    def append(self, item: T) -> None:
        self._items.append(item)

    def extend(self, items: Iterable[T]) -> None:
        for item in items:
            self.append(item)

    def to_list(self) -> list[T]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)
