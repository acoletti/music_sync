from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Set
from enum import Enum, auto

class MediaType(Enum):
    """Enum for different types of media files."""
    AUDIO = auto()
    VIDEO = auto()
    DOCUMENT = auto()

class AudioFormat(Enum):
    """Enum for audio file formats."""
    MP3 = '.mp3'
    FLAC = '.flac'
    WAV = '.wav'
    M4A = '.m4a'
    DSF = '.dsf'

class VideoFormat(Enum):
    """Enum for video file formats."""
    MP4 = '.mp4'
    MKV = '.mkv'
    AVI = '.avi'
    MOV = '.mov'
    WMV = '.wmv'

class DocumentFormat(Enum):
    """Enum for document file formats."""
    PDF = '.pdf'
    DOC = '.doc'
    DOCX = '.docx'
    TXT = '.txt'
    RTF = '.rtf'

@dataclass
class MediaInfo:
    """Base class for all media file information."""
    path: Path
    name: str
    size: int
    normalized_name: str
    hash: Optional[str] = field(default=None)

@dataclass
class AudioInfo(MediaInfo):
    """Information about an audio file."""
    duration: Optional[float] = field(default=None)  # Duration in seconds
    bitrate: Optional[int] = field(default=None)    # Bitrate in kbps
    sample_rate: Optional[int] = field(default=None) # Sample rate in Hz
    format: Optional[AudioFormat] = field(default=None)

@dataclass
class VideoInfo(MediaInfo):
    """Information about a video file."""
    duration: Optional[float] = field(default=None)  # Duration in seconds
    resolution: Optional[tuple[int, int]] = field(default=None)  # (width, height)
    bitrate: Optional[int] = field(default=None)    # Bitrate in kbps
    format: Optional[VideoFormat] = field(default=None)

@dataclass
class DocumentInfo(MediaInfo):
    """Information about a document file."""
    page_count: Optional[int] = field(default=None)
    word_count: Optional[int] = field(default=None)
    format: Optional[DocumentFormat] = field(default=None) 