from pathlib import Path
from dataclasses import dataclass
from typing import List

@dataclass
class FolderInfo:
    """Data class to hold folder information."""
    path: Path
    name: str
    size: int
    file_count: int
    files: List[str] 