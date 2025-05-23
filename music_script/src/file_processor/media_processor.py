from pathlib import Path
from typing import List, Optional, Dict, Type
from .factories.base import AbstractProcessor
from .factories.media import MediaFactoryRegistry
from .models.media import MediaInfo, MediaType
from .models.file import FileType

class MediaProcessor(AbstractProcessor[MediaInfo]):
    """Generic processor for handling any type of media files."""
    def __init__(self, media_type: Optional[MediaType] = None):
        self.factory_registry = MediaFactoryRegistry()
        self.media_type = media_type
        super().__init__(
            factory=self.factory_registry.get_factory(media_type) if media_type else None,
            file_type_enum=None  # Will be set by specific factory
        )

    def get_files(self, directory: Path) -> List[MediaInfo]:
        """Get all media files from a directory."""
        if self.media_type:
            # Use specific factory if media type is specified
            return super().get_files(directory)
        
        # Otherwise, use all factories
        all_files: List[MediaInfo] = []
        for factory in self.factory_registry.get_all_factories():
            self.factory = factory
            all_files.extend(super().get_files(directory))
        return all_files

    def process_file(self, file_path: Path, fast: bool = False) -> Optional[MediaInfo]:
        """Process a single media file."""
        if not self.media_type:
            # Try each factory until one succeeds
            for factory in self.factory_registry.get_all_factories():
                self.factory = factory
                if result := super().process_file(file_path, fast):
                    return result
            return None
        return super().process_file(file_path, fast)

    def get_duplicates(self, files: List[MediaInfo], fast: bool = False) -> Dict[str, List[MediaInfo]]:
        """Find duplicate media files based on content."""
        duplicates: Dict[str, List[MediaInfo]] = {}
        
        # Group by normalized name first
        by_name: Dict[str, List[MediaInfo]] = {}
        for file in files:
            by_name.setdefault(file.normalized_name, []).append(file)
        
        # Check each group for actual duplicates
        for name, group in by_name.items():
            if len(group) > 1:
                # Further check by size and hash
                by_size: Dict[int, List[MediaInfo]] = {}
                for file in group:
                    by_size.setdefault(file.size, []).append(file)
                
                for size, size_group in by_size.items():
                    if len(size_group) > 1:
                        if fast:
                            # In fast mode, consider files with same size as duplicates
                            duplicates[name] = size_group
                        else:
                            # In normal mode, check hashes
                            by_hash: Dict[str, List[MediaInfo]] = {}
                            for file in size_group:
                                if file.hash:
                                    by_hash.setdefault(file.hash, []).append(file)
                            
                            # Add groups with multiple files sharing the same hash
                            for hash_group in by_hash.values():
                                if len(hash_group) > 1:
                                    duplicates[name] = hash_group
        
        return duplicates 