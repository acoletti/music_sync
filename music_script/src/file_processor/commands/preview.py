import argparse
from pathlib import Path
from typing import List

from .base import Command
from ..factories.media import MediaFactoryRegistry
from ..models.file import FileType

class PreviewCommand(Command):
    """Command for previewing changes."""
    def __init__(self):
        super().__init__()
        self.factory_registry = MediaFactoryRegistry()

    def execute(self, args: argparse.Namespace) -> None:
        """Execute the preview command."""
        source_dir = args.source_dir
        target_dir = args.target_dir
        fast = args.fast
        file_type_str = args.file_type

        print(f"\nPreviewing changes for {file_type_str} files in: {source_dir}")
        if target_dir:
            print(f"Target directory: {target_dir}")
        print(f"Fast mode: {fast}\n")

        files = self._get_all_files(source_dir)
        
        if file_type_str == "all":
            process_file_types = [FileType.AUDIO, FileType.VIDEO, FileType.DOCUMENT]
        else:
            try:
                process_file_types = [FileType[file_type_str.upper()]]
            except KeyError:
                print(f"Error: Invalid file type '{file_type_str}'. Choices are 'all', 'audio', 'video', 'document'.")
                return

        for ft in process_file_types:
            try:
                factory = self.factory_registry.get_factory(ft)
            except KeyError: 
                print(f"Warning: No factory registered for file type {ft.name}. Skipping.")
                continue
            
            file_infos = factory.create_batch(files, fast=fast)
            
            if file_infos:
                print(f"\nFound {len(file_infos)} {ft.name.lower()} files:")
                for file_info in file_infos:
                    print(f"  - {file_info.name} ({getattr(file_info, 'size', 0):,} bytes)")

    def _get_all_files(self, directory: Path) -> List[Path]:
        """Get all files in the directory and its subdirectories."""
        return [
            path for path in directory.rglob("*")
            if path.is_file()
        ]

    def add_to_parser(self, parser: argparse.ArgumentParser) -> None:
        """Add command-specific arguments to the parser."""
        parser.add_argument(
            "--source-dir",
            type=Path,
            required=True,
            help="The source directory to scan for files."
        )
        parser.add_argument(
            "--target-dir",
            type=Path,
            help="The target directory (optional, for future use or context)."
        )
        parser.add_argument(
            "--fast",
            action="store_true",
            help="Enable fast mode (e.g., skip hash calculation)."
        )
        parser.add_argument(
            "--file-type",
            type=str,
            default="all",
            choices=["all", "audio", "video", "document"],
            help="Type of files to preview (all, audio, video, document)."
        ) 