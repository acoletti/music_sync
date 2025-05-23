import hashlib
import os
import time
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
        self._max_retries = 3
        self._retry_delay = 2  # Increased delay for network shares

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
        for attempt in range(self._max_retries):
            try:
                if not os.access(file_path, os.R_OK):
                    raise PermissionError(f"No read access to {file_path}")
                return file_path.is_file() and file_path.suffix.lower() in self._supported_formats
            except PermissionError as e:
                if attempt < self._max_retries - 1:
                    print(f"Warning: Permission denied for file {file_path}, retrying... ({attempt + 1}/{self._max_retries})")
                    time.sleep(self._retry_delay)
                    continue
                print(f"Warning: Permission denied for file {file_path} after {self._max_retries} attempts: {e}")
                print("If this is a network share, ensure you have read access")
                return False
            except Exception as e:
                print(f"Warning: Error checking file {file_path}: {e}")
                return False
        return False

    def _get_file_size(self, file_path: Path) -> Optional[int]:
        """Get file size with retry logic."""
        for attempt in range(self._max_retries):
            try:
                if not os.access(file_path, os.R_OK):
                    raise PermissionError(f"No read access to {file_path}")
                return file_path.stat().st_size
            except (PermissionError, OSError) as e:
                if attempt < self._max_retries - 1:
                    print(f"Warning: Error getting file size for {file_path}, retrying... ({attempt + 1}/{self._max_retries})")
                    time.sleep(self._retry_delay)
                    continue
                print(f"Warning: Could not get file size for {file_path}: {e}")
                return None
        return None

    def _calculate_file_hash(self, file_path: Path, block_size: int = 65536) -> Optional[str]:
        """Calculate SHA-256 hash of a file."""
        return FileFactory.calculate_file_hash(file_path, block_size)

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
            file_size = self._get_file_size(path)
            if file_size is None:
                print(f"Warning: Could not get file size for {path}, skipping file")
                return None

            file_hash = None if fast else self._calculate_file_hash(path)
            if not fast and file_hash is None:
                print(f"Warning: Could not calculate hash for {path}, skipping file")
                return None

            return AudioInfo(
                path=path,
                name=path.stem,
                size=file_size,
                normalized_name=normalize_track_name(path.stem),
                hash=file_hash,
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
            file_size = self._get_file_size(path)
            if file_size is None:
                print(f"Warning: Could not get file size for {path}, skipping file")
                return None

            file_hash = None if fast else self._calculate_file_hash(path)
            if not fast and file_hash is None:
                print(f"Warning: Could not calculate hash for {path}, skipping file")
                return None

            return VideoInfo(
                path=path,
                name=path.stem,
                size=file_size,
                normalized_name=normalize_track_name(path.stem),
                hash=file_hash,
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
            file_size = self._get_file_size(path)
            if file_size is None:
                print(f"Warning: Could not get file size for {path}, skipping file")
                return None

            file_hash = None if fast else self._calculate_file_hash(path)
            if not fast and file_hash is None:
                print(f"Warning: Could not calculate hash for {path}, skipping file")
                return None

            return DocumentInfo(
                path=path,
                name=path.stem,
                size=file_size,
                normalized_name=normalize_track_name(path.stem),
                hash=file_hash,
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