import argparse
from pathlib import Path
from typing import List, Dict, Set, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base import Command
from ..factories.media import MediaFactoryRegistry
from ..models.file import FileInfo, FileType

class PreviewCommand(Command):
    """Command for previewing changes."""
    def __init__(self, source_dir: Path, target_dir: Optional[Path] = None, fast: bool = False, file_type: str = "all"):
        super().__init__()
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.fast = fast
        self.file_type = file_type
        self.factory_registry = MediaFactoryRegistry()

    def execute(self) -> None:
        """Execute the preview command."""
        print(f"\nPreviewing changes for {self.file_type} files in: {self.source_dir}")
        if self.target_dir:
            print(f"Target directory: {self.target_dir}")
        print(f"Fast mode: {self.fast}\n")

        # Get all files from the source directory
        files = self._get_all_files(self.source_dir)
        
        # Process files by type
        if self.file_type == "all":
            file_types = [FileType.AUDIO, FileType.VIDEO, FileType.DOCUMENT]
        else:
            file_types = [FileType[self.file_type.upper()]]

        for file_type in file_types:
            factory = self.factory_registry.get_factory(file_type)
            file_infos = factory.create_batch(files, fast=self.fast)
            
            if file_infos:
                print(f"\nFound {len(file_infos)} {file_type.name.lower()} files:")
                for file_info in file_infos:
                    print(f"  - {file_info.name} ({file_info.size:,} bytes)")

    def _get_all_files(self, directory: Path) -> List[Path]:
        """Get all files in the directory and its subdirectories."""
        return [
            path for path in directory.rglob("*")
            if path.is_file()
        ]

    def add_to_parser(self, parser):
        pass 