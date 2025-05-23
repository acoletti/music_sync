import hashlib
from pathlib import Path
from typing import List, Optional, Set, Dict, Type
from abc import ABC, abstractmethod
from enum import Enum, auto

from ..models.media import (
    MediaInfo, AudioInfo, VideoInfo, DocumentInfo,
    MediaType, AudioFormat, VideoFormat, DocumentFormat
)
from ..utils.normalization import normalize_track_name

from .file import FileFactory, FileFactoryRegistry
from .audio import DefaultAudioFactory
from .video import DefaultVideoFactory
from .document import DefaultDocumentFactory
from ..models.file import FileType

class MediaType(Enum):
    """Enum for different types of media files."""
    AUDIO = auto()
    VIDEO = auto()
    DOCUMENT = auto()

class MediaFactory(ABC):
    """Base factory interface for creating media info objects."""
    def __init__(self):
        self._supported_formats: Set[str] = set()

    @abstractmethod
    def create(self, path: Path, fast: bool = False) -> Optional[MediaInfo]:
        """Create a new media info object."""
        pass

    def create_batch(self, paths: List[Path], fast: bool = False) -> List[MediaInfo]:
        """Create multiple media info objects."""
        return [
            media_info
            for path in paths
            if (media_info := self.create(path, fast=fast)) is not None
        ]

    def _is_supported_file(self, file_path: Path) -> bool:
        """Check if a file is of a supported format."""
        return file_path.is_file() and file_path.suffix.lower() in self._supported_formats

    def _calculate_file_hash(self, file_path: Path, block_size: int = 65536) -> str:
        """Calculate SHA-256 hash of a file."""
        file_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for block in iter(lambda: f.read(block_size), b''):
                file_hash.update(block)
        return file_hash.hexdigest()

class AudioFactory(MediaFactory):
    """Factory for creating audio file info objects."""
    def __init__(self):
        super().__init__()
        self._supported_formats = {fmt.value for fmt in AudioFormat}

    def create(self, path: Path, fast: bool = False) -> Optional[AudioInfo]:
        """Create a new audio info object."""
        try:
            if not path.exists() or not self._is_supported_file(path):
                return None

            format = AudioFormat(path.suffix.lower())
            return AudioInfo(
                path=path,
                name=path.stem,
                size=path.stat().st_size,
                normalized_name=normalize_track_name(path.stem),
                hash=None if fast else self._calculate_file_hash(path),
                format=format
            )
        except (PermissionError, OSError) as e:
            print(f"Warning: Could not process audio file {path}: {e}")
            return None

class VideoFactory(MediaFactory):
    """Factory for creating video file info objects."""
    def __init__(self):
        super().__init__()
        self._supported_formats = {fmt.value for fmt in VideoFormat}

    def create(self, path: Path, fast: bool = False) -> Optional[VideoInfo]:
        """Create a new video info object."""
        try:
            if not path.exists() or not self._is_supported_file(path):
                return None

            format = VideoFormat(path.suffix.lower())
            return VideoInfo(
                path=path,
                name=path.stem,
                size=path.stat().st_size,
                normalized_name=normalize_track_name(path.stem),
                hash=None if fast else self._calculate_file_hash(path),
                format=format
            )
        except (PermissionError, OSError) as e:
            print(f"Warning: Could not process video file {path}: {e}")
            return None

class DocumentFactory(MediaFactory):
    """Factory for creating document file info objects."""
    def __init__(self):
        super().__init__()
        self._supported_formats = {fmt.value for fmt in DocumentFormat}

    def create(self, path: Path, fast: bool = False) -> Optional[DocumentInfo]:
        """Create a new document info object."""
        try:
            if not path.exists() or not self._is_supported_file(path):
                return None

            format = DocumentFormat(path.suffix.lower())
            return DocumentInfo(
                path=path,
                name=path.stem,
                size=path.stat().st_size,
                normalized_name=normalize_track_name(path.stem),
                hash=None if fast else self._calculate_file_hash(path),
                format=format
            )
        except (PermissionError, OSError) as e:
            print(f"Warning: Could not process document file {path}: {e}")
            return None

class MediaFactoryRegistry(FileFactoryRegistry):
    """Registry for media factories."""
    def __init__(self):
        super().__init__()
        self._factories: Dict[FileType, FileFactory] = {
            FileType.AUDIO: DefaultAudioFactory(),
            FileType.VIDEO: DefaultVideoFactory(),
            FileType.DOCUMENT: DefaultDocumentFactory()
        }

    def get_factory(self, media_type: MediaType):
        """Get the factory for a specific media type."""
        return self._factories[media_type]

    def get_all_factories(self) -> List:
        """Get all registered factories."""
        return list(self._factories.values()) 