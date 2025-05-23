import hashlib
from pathlib import Path
from typing import List, Optional, Set, Type
from abc import ABC, abstractmethod

from ..models.file import FileInfo, FileType
from ..utils.normalization import normalize_track_name

class FileFactory(ABC):
    """Base factory interface for creating file info objects."""
    def __init__(self):
        self._supported_formats: Set[str] = set()

    @abstractmethod
    def create(self, path: Path, fast: bool = False) -> Optional[FileInfo]:
        """Create a new file info object."""
        pass

    def create_batch(self, paths: List[Path], fast: bool = False) -> List[FileInfo]:
        """Create multiple file info objects."""
        return [
            file_info
            for path in paths
            if (file_info := self.create(path, fast=fast)) is not None
        ]

    def _is_supported_file(self, file_path: Path) -> bool:
        """Check if a file is of a supported format."""
        return file_path.is_file() and file_path.suffix.lower() in self._supported_formats

    @staticmethod
    def calculate_file_hash(file_path: Path, block_size: int = 65536) -> str:
        """Calculate SHA-256 hash of a file."""
        file_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for block in iter(lambda: f.read(block_size), b''):
                file_hash.update(block)
        return file_hash.hexdigest()

class FileFactoryRegistry:
    """Registry for file factories."""
    def __init__(self):
        self._factories: Dict[FileType, FileFactory] = {}

    def register_factory(self, file_type: FileType, factory: FileFactory) -> None:
        """Register a new factory."""
        self._factories[file_type] = factory

    def get_factory(self, file_type: FileType) -> FileFactory:
        """Get the factory for a specific file type."""
        return self._factories[file_type]

    def get_all_factories(self) -> List[FileFactory]:
        """Get all registered factories."""
        return list(self._factories.values()) 