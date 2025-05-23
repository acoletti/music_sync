import os
from pathlib import Path
from typing import List, Optional, Set
import time

from .base import FolderFactory
from ..models.folder import FolderInfo

class DefaultFolderFactory(FolderFactory):
    """Default implementation of the folder factory."""
    def __init__(self):
        self._max_retries = 3
        self._retry_delay = 2  # Increased delay for network shares

    def create(self, path: Path, fast: bool = False) -> Optional[FolderInfo]:
        """Create a new folder info object."""
        try:
            if not path.exists():
                print(f"Warning: Path {path} does not exist")
                return None
            if not path.is_dir():
                print(f"Warning: Path {path} is not a directory")
                return None

            # Check read access
            if not os.access(path, os.R_OK):
                print(f"Warning: No read access to directory {path}")
                return None

            if fast:
                return FolderInfo(
                    path=path,
                    name=path.name,
                    size=0,
                    file_count=0,
                    files=[]
                )

            # Get folder contents with retries
            for attempt in range(self._max_retries):
                try:
                    size = self._get_folder_size(path)
                    file_count = self._count_files(path)
                    files = self._get_folder_files(path)
                    return FolderInfo(
                        path=path,
                        name=path.name,
                        size=size,
                        file_count=file_count,
                        files=files
                    )
                except (PermissionError, OSError) as e:
                    if attempt < self._max_retries - 1:
                        print(f"Warning: Error accessing directory {path}, retrying... ({attempt + 1}/{self._max_retries})")
                        time.sleep(self._retry_delay)
                        continue
                    raise

        except (PermissionError, OSError) as e:
            print(f"Warning: Could not process folder {path}: {e}")
            print("If this is a network share, ensure you have read access")
            return None
        except Exception as e:
            print(f"Warning: Unexpected error processing folder {path}: {e}")
            return None

    def create_batch(self, paths: List[Path], fast: bool = False) -> List[FolderInfo]:
        """Create multiple folder info objects."""
        return [
            folder_info
            for path in paths
            if (folder_info := self.create(path, fast=fast)) is not None
        ]

    def _get_folder_size(self, folder_path: Path) -> int:
        """Calculate total size of a folder, excluding symlinks."""
        total = 0
        try:
            for dirpath, _, filenames in os.walk(folder_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if not os.path.islink(fp):
                        try:
                            total += os.path.getsize(fp)
                        except (PermissionError, OSError) as e:
                            print(f"Warning: Could not get size for {fp}: {e}")
                            continue
        except (PermissionError, OSError) as e:
            print(f"Warning: Error walking directory {folder_path}: {e}")
        return total

    def _count_files(self, folder_path: Path) -> int:
        """Count files in a folder (top-level only for speed)."""
        try:
            return sum(1 for entry in os.scandir(folder_path) if entry.is_file())
        except (PermissionError, OSError) as e:
            print(f"Warning: Error counting files in {folder_path}: {e}")
            return 0

    def _get_folder_files(self, folder_path: Path, limit: int = 5) -> List[str]:
        """Get the first N files from a folder (top-level only for speed)."""
        try:
            return [entry.name for entry in os.scandir(folder_path) if entry.is_file()][:limit]
        except (PermissionError, OSError) as e:
            print(f"Warning: Error getting files from {folder_path}: {e}")
            return [] 