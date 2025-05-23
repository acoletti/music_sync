import hashlib
from pathlib import Path
from typing import List, Optional, Set
from abc import ABC, abstractmethod

from ..models.audio import AudioInfo, AudioFormat
from ..models.file import FileInfo, FileType
from .file import FileFactory
from ..utils.normalization import normalize_track_name
from src.file_processor.factories.document import DefaultDocumentFactory

class AudioFactory(FileFactory):
    """Factory interface for creating audio info objects."""
    def __init__(self):
        super().__init__()
        self._supported_formats = {fmt.value for fmt in AudioFormat}

    @abstractmethod
    def create(self, path: Path, fast: bool = False) -> Optional[AudioInfo]:
        """Create a new audio info object."""
        pass

    def create_batch(self, paths: List[Path], fast: bool = False) -> List[AudioInfo]:
        """Create multiple audio info objects."""
        return [
            audio_info
            for path in paths
            if (audio_info := self.create(path, fast=fast)) is not None
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

class DefaultAudioFactory(AudioFactory):
    """Default implementation of AudioFactory."""
    def create(self, path: Path, fast: bool = False) -> Optional[AudioInfo]:
        """Create a new audio info object."""
        if not self._is_supported_file(path):
            return None

        try:
            name = path.stem
            normalized_name = normalize_track_name(name)
            size = path.stat().st_size
            format = AudioFormat(path.suffix.lower())
            hash = None if fast else self._calculate_file_hash(path)

            return AudioInfo(
                path=path,
                name=name,
                size=size,
                normalized_name=normalized_name,
                format=format,
                hash=hash
            )
        except Exception as e:
            print(f"Error creating audio info for {path}: {e}")
            return None 