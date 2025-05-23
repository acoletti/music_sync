from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

@dataclass
class BaseInfo:
    """Base class for all info objects."""
    path: Path
    name: str
    size: int

@dataclass
class FolderInfo(BaseInfo):
    """Data class to hold folder information."""
    file_count: int
    files: List[str]

@dataclass
class TrackInfo(BaseInfo):
    """Data class to hold track information."""
    hash: str
    normalized_name: str

class BaseIterator(ABC):
    """Base iterator class for processing items."""
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self._items: List[BaseInfo] = []
        self._current_index = 0

    def __iter__(self) -> 'BaseIterator':
        return self

    def __next__(self) -> BaseInfo:
        if self._current_index >= len(self._items):
            raise StopIteration
        item = self._items[self._current_index]
        self._current_index += 1
        return item

    @abstractmethod
    def load_items(self) -> None:
        """Load all items from the root path."""
        pass 