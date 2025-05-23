import os
from pathlib import Path
from typing import List, Optional

from .base import FolderFactory
from ..models.folder import FolderInfo

class DefaultFolderFactory(FolderFactory):
    """Default implementation of the folder factory."""
    def create(self, path: Path, fast: bool = False) -> Optional[FolderInfo]:
        """Create a new folder info object."""
        try:
            if not path.exists():
                print(f"Warning: Path {path} does not exist")
                return None
            if fast:
                return FolderInfo(
                    path=path,
                    name=path.name,
                    size=0,
                    file_count=0,
                    files=[]
                )
            return FolderInfo(
                path=path,
                name=path.name,
                size=self._get_folder_size(path),
                file_count=self._count_files(path),
                files=self._get_folder_files(path)
            )
        except (PermissionError, OSError) as e:
            print(f"Warning: Could not process folder {path}: {e}")
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
        for dirpath, _, filenames in os.walk(folder_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    try:
                        total += os.path.getsize(fp)
                    except Exception:
                        pass
        return total

    def _count_files(self, folder_path: Path) -> int:
        """Count files in a folder (top-level only for speed)."""
        try:
            return sum(1 for entry in os.scandir(folder_path) if entry.is_file())
        except Exception:
            return 0

    def _get_folder_files(self, folder_path: Path, limit: int = 5) -> List[str]:
        """Get the first N files from a folder (top-level only for speed)."""
        try:
            return [entry.name for entry in os.scandir(folder_path) if entry.is_file()][:limit]
        except Exception:
            return [] 