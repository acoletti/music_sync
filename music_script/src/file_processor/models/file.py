from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from enum import Enum, auto

class FileType(Enum):
    """Enum for different types of files."""
    AUDIO = auto()
    VIDEO = auto()
    DOCUMENT = auto()

@dataclass
class FileInfo:
    """Base class for all file information."""
    path: Path
    name: str
    size: int
    normalized_name: str
    hash: Optional[str] = None 