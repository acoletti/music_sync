from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass

from ..media_processor import MediaProcessor
from ..models.media import MediaInfo, MediaType
from ..models.file import FileType
from .base import Command

@dataclass
class ProcessingResult:
    """Results of processing files."""
    processed_files: int = 0
    cleaned_files: int = 0
    duplicates: Dict[str, List[MediaInfo]] = None

class CommandProcessor(Command):
    """Command to process and clean up files."""
    def __init__(self, source_dir: Path, target_dir: Optional[Path] = None, media_type: Optional[MediaType] = None):
        super().__init__()
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.media_processor = MediaProcessor(media_type)

    def execute(self) -> ProcessingResult:
        """Execute the file processing command."""
        print(f"\nProcessing files in {self.source_dir}")
        if self.target_dir:
            print(f"Target directory: {self.target_dir}")
        
        # Get all media files
        files = self.media_processor.get_files(self.source_dir)
        if not files:
            print("No media files found.")
            return ProcessingResult()
        
        print(f"\nFound {len(files)} media files")
        
        # Find duplicates
        duplicates = self.media_processor.get_duplicates(files)
        if not duplicates:
            print("No duplicates found.")
            return ProcessingResult(processed_files=len(files))
        
        print(f"\nFound {len(duplicates)} groups of duplicate files")
        
        # Process duplicates
        cleaned_count = 0
        for name, group in duplicates.items():
            print(f"\nProcessing group: {name}")
            print(f"Found {len(group)} duplicate files")
            
            # Sort by size (largest first)
            group.sort(key=lambda x: x.size, reverse=True)
            
            # Keep the largest file
            keep_file = group[0]
            print(f"Keeping: {keep_file.path} ({keep_file.size} bytes)")
            
            # Remove others
            for file in group[1:]:
                try:
                    if self.target_dir:
                        # Move to target directory
                        target_path = self.target_dir / file.path.name
                        file.path.rename(target_path)
                        print(f"Moved: {file.path} -> {target_path}")
                    else:
                        # Delete file
                        file.path.unlink()
                        print(f"Deleted: {file.path}")
                    cleaned_count += 1
                except Exception as e:
                    print(f"Error processing {file.path}: {e}")
        
        return ProcessingResult(
            processed_files=len(files),
            cleaned_files=cleaned_count,
            duplicates=duplicates
        ) 