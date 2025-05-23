from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Type, TypeVar, Generic, cast
import re
from functools import reduce
from enum import Enum

from ..models.base import BaseInfo, FolderInfo, TrackInfo
from ..models.audio import AudioInfo
from ..models.document import DocumentInfo
from ..models.video import VideoInfo

InfoType = TypeVar('InfoType', bound=BaseInfo)

class BaseFactory(ABC, Generic[InfoType]):
    """Base factory interface."""
    @abstractmethod
    def create(self, path: Path) -> Optional[InfoType]:
        """Create a new info object."""
        pass

class FolderFactory(BaseFactory):
    """Factory interface for creating folder info objects."""
    @abstractmethod
    def create(self, path: Path) -> Optional[FolderInfo]:
        """Create a new folder info object."""
        pass

    @abstractmethod
    def create_batch(self, paths: List[Path]) -> List[FolderInfo]:
        """Create multiple folder info objects."""
        pass

class TrackFactory(BaseFactory):
    """Factory interface for creating track info objects."""
    @abstractmethod
    def create(self, path: Path) -> Optional[TrackInfo]:
        """Create a new track info object."""
        pass

    @abstractmethod
    def create_batch(self, paths: List[Path]) -> List[TrackInfo]:
        """Create multiple track info objects."""
        pass

class AbstractProcessor(ABC, Generic[InfoType]):
    """Abstract base class for file processors."""
    def __init__(self, factory: BaseFactory[InfoType], file_type_enum: Type[Enum]):
        self._factory = factory
        self._file_type_enum = file_type_enum

    def normalize_name(self, name: str) -> str:
        """Normalize file name by removing common prefixes and suffixes."""
        name = self.remove_numbers(name)
        name = self.remove_disc_numbers(name)
        name = self.remove_common_suffixes(name)
        return name.lower()

    def remove_numbers(self, name: str) -> str:
        """Remove numbers from the beginning of the name."""
        name = re.sub(r'^\d+\s*[-–—]\s*', '', name)
        name = re.sub(r'^\d+\.\s*', '', name)
        return name

    def remove_disc_numbers(self, name: str) -> str:
        """Remove disc/CD numbers from the name."""
        name = re.sub(r'\(disc\s*\d+\)', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\(cd\s*\d+\)', '', name, flags=re.IGNORECASE)
        return name

    def remove_common_suffixes(self, name: str) -> str:
        """Remove common suffixes like (live), (remastered), etc."""
        suffixes = [
            (r'\s*\(live\)$', ''),
            (r'\s*\(remastered\)$', ''),
            (r'\s*\(remaster\)$', '')
        ]
        return reduce(lambda s, pattern_replacement: re.sub(
            pattern_replacement[0], pattern_replacement[1], s, flags=re.IGNORECASE
        ), suffixes, name)

    def _is_relevant_file(self, file_path: Path) -> bool:
        """Check if a file is of the relevant type."""
        try:
            if not file_path.exists():
                print(f"Warning: Path does not exist: {file_path}")
                return False
            if file_path.is_dir():
                print(f"Warning: Path is a directory, not a file: {file_path}")
                return False
            return file_path.suffix.lower() in {fmt.value for fmt in self._file_type_enum}
        except Exception as e:
            print(f"Warning: Error checking file {file_path}: {e}")
            return False

    def process_file(self, file_path: Path) -> Optional[InfoType]:
        """Process a file and return its info using the provided factory."""
        if self._is_relevant_file(file_path):
            return self._factory.create(file_path)
        return None

    def get_files(self, folder_path: Path) -> List[InfoType]:
        """Get all files of the processor's type from a folder."""
        files: List[InfoType] = []
        if not folder_path.is_dir():
            print(f"Warning: Path is not a directory: {folder_path}")
            return files
        try:
            for file_path in folder_path.rglob('*'):
                if self._is_relevant_file(file_path):
                    try:
                        processed_file = self.process_file(file_path)
                        if processed_file:
                            files.append(processed_file)
                    except PermissionError as e:
                        print(f"Warning: Permission denied for file {file_path}: {e}")
                        continue
                    except Exception as e:
                        print(f"Warning: Error processing file {file_path}: {e}")
                        continue
        except PermissionError as e:
            print(f"Warning: Permission denied for folder {folder_path}: {e}")
            print("If this is a network share, ensure you have read access")
        except Exception as e:
            print(f"Warning: Error accessing folder {folder_path}: {e}")

        return files 