from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum

class VideoFormat(Enum):
    """Enum for video file formats."""
    MP4 = '.mp4'
    MKV = '.mkv'
    AVI = '.avi'
    MOV = '.mov'
    WMV = '.wmv'

@dataclass
class VideoInfo:
    """Information about a video file."""
    path: Path
    name: str
    size: int
    normalized_name: str
    format: VideoFormat
    hash: Optional[str] = None
    duration: Optional[float] = None  # Duration in seconds
    resolution: Optional[Tuple[int, int]] = None  # (width, height)
    bitrate: Optional[int] = None    # Bitrate in kbps 