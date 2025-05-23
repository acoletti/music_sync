import hashlib
import time
import os
from pathlib import Path
from typing import List, Optional, Set, Type, Dict
from abc import ABC, abstractmethod

from ..models.file import FileInfo, FileType
from ..utils.normalization import normalize_track_name

class FileFactory(ABC):
    """Base factory interface for creating file info objects."""
    def __init__(self):
        self._supported_formats: Set[str] = set()
        self._max_retries = 3
        self._retry_delay = 2  # Increased delay for network shares
        self._chunk_size = 65536  # Default chunk size for reading files

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
        for attempt in range(self._max_retries):
            try:
                if not file_path.exists():
                    print(f"Warning: Path does not exist: {file_path}")
                    return False
                if file_path.is_dir():
                    print(f"Warning: Path is a directory, not a file: {file_path}")
                    return False
                if not os.access(file_path, os.R_OK):
                    raise PermissionError(f"No read access to {file_path}")
                return file_path.suffix.lower() in self._supported_formats
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

    @staticmethod
    def calculate_file_hash(file_path: Path, block_size: int = 65536) -> Optional[str]:
        """Calculate SHA-256 hash of a file with improved network share handling."""
        max_retries = 3
        retry_delay = 2  # Increased delay for network shares
        chunk_size = min(block_size, 32768)  # Use smaller chunks for network shares

        # Check if path exists and is a file
        if not file_path.exists():
            print(f"Warning: Cannot calculate hash - path does not exist: {file_path}")
            return None
        if file_path.is_dir():
            print(f"Warning: Cannot calculate hash - path is a directory: {file_path}")
            return None

        for attempt in range(max_retries):
            try:
                file_hash = hashlib.sha256()
                # Check if file exists and is readable before attempting to hash
                if not os.access(file_path, os.R_OK):
                    raise PermissionError(f"No read access to {file_path}")
                
                with open(file_path, 'rb') as f:
                    while True:
                        try:
                            chunk = f.read(chunk_size)
                            if not chunk:
                                break
                            file_hash.update(chunk)
                        except (IOError, OSError) as e:
                            if attempt < max_retries - 1:
                                print(f"Warning: Error reading file chunk, retrying... ({attempt + 1}/{max_retries})")
                                time.sleep(retry_delay)
                                break  # Break inner loop to retry from start
                            raise  # Re-raise if we're out of retries
                return file_hash.hexdigest()
            except PermissionError as e:
                if attempt < max_retries - 1:
                    print(f"Warning: Permission denied while calculating hash for {file_path}, retrying... ({attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                print(f"Warning: Permission denied while calculating hash for {file_path} after {max_retries} attempts: {e}")
                print("If this is a network share, ensure you have read access")
                return None
            except (IOError, OSError) as e:
                if attempt < max_retries - 1:
                    print(f"Warning: Error accessing file {file_path}, retrying... ({attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                print(f"Warning: Error calculating hash for {file_path}: {e}")
                return None
            except Exception as e:
                print(f"Warning: Unexpected error calculating hash for {file_path}: {e}")
                return None
        return None

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