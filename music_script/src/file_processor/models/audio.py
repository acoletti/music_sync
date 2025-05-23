from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from enum import Enum

class AudioFormat(Enum):
    """Enum for audio file formats."""
    MP3 = '.mp3'
    FLAC = '.flac'
    WAV = '.wav'
    M4A = '.m4a'
    DSF = '.dsf'

@dataclass
class AudioInfo:
    """Information about an audio file."""
    path: Path
    name: str
    size: int
    normalized_name: str
    format: AudioFormat
    hash: Optional[str] = None
    duration: Optional[float] = None  # Duration in seconds
    bitrate: Optional[int] = None    # Bitrate in kbps
    sample_rate: Optional[int] = None # Sample rate in Hz 