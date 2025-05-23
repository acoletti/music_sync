from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrackInfo:
    """Data class to hold track information."""
    path: Path
    name: str
    size: int
    normalized_name: str
    hash: Optional[str] = None 