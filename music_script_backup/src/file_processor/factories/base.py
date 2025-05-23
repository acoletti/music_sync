from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
import re
from functools import reduce
from typing import Any

from ..models.base import BaseInfo, FolderInfo, TrackInfo

class BaseFactory(ABC):
    """Base factory interface."""
    @abstractmethod
    def create(self, path: Path) -> Optional[BaseInfo]:
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

class AbstractProcessor(ABC):
    """Abstract base class for file processors."""
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

    @abstractmethod
    def process_file(self, file_path: Path) -> Optional[Any]:
        """Process a file and return its info."""
        pass

    @abstractmethod
    def get_files(self, folder_path: Path) -> List[Any]:
        """Get all files of the processor's type from a folder."""
        pass 