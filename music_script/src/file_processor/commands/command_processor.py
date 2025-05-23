from pathlib import Path
from typing import List, Optional

from ..factories.media import MediaFactoryRegistry
from ..models.file import FileInfo, FileType
from .base import Command

class CommandProcessor(Command):
    """Command to process and clean up files."""
    def __init__(self, source_dir: Path, target_dir: Optional[Path] = None):
        super().__init__()
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.factory_registry = MediaFactoryRegistry()

    def execute(self) -> None:
        """Execute the file processing command."""
        # Get all files from the source directory
        files = self._get_all_files(self.source_dir)
        
        # Process files by type
        for file_type in FileType:
            factory = self.factory_registry.get_factory(file_type)
            file_infos = factory.create_batch(files)
            
            if file_infos:
                print(f"\nProcessing {file_type.value} files:")
                self._process_files(file_infos)

    def _get_all_files(self, directory: Path) -> List[Path]:
        """Get all files in the directory and its subdirectories."""
        return [
            path for path in directory.rglob("*")
            if path.is_file()
        ]

    def _process_files(self, file_infos: List[FileInfo]) -> None:
        """Process a list of file info objects."""
        # TODO: Implement file processing logic
        pass 