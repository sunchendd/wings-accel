from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Task:
    pass


class KVStoreBase(ABC):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def cc_store(self) -> object:
        raise NotImplementedError

    @abstractmethod
    def lookup(self, block_ids: list[bytes]) -> list[bool]:
        raise NotImplementedError

    def prefetch(self, block_ids: list[bytes]) -> None:
        return None

    @abstractmethod
    def load_data(
        self,
        block_ids: list[bytes],
        shard_index: list[int],
        dst_addr: list[list[int]] | np.ndarray,
    ) -> Task:
        raise NotImplementedError

    @abstractmethod
    def dump_data(
        self,
        block_ids: list[bytes],
        shard_index: list[int],
        src_addr: list[list[int]] | np.ndarray,
    ) -> Task:
        raise NotImplementedError

    @abstractmethod
    def wait(self, task: Task) -> Any:
        raise NotImplementedError

    @abstractmethod
    def check(self, task: Task) -> Any:
        raise NotImplementedError
