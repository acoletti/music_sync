#!/usr/bin/env python3
import os
import re
from pathlib import Path
from collections import defaultdict
import shutil
from difflib import SequenceMatcher
import unicodedata
from tqdm import tqdm
import hashlib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial, reduce
import argparse
import json
import time
from datetime import datetime
from dataclasses import dataclass
from typing import List, Set, Dict, Tuple, Iterator, Optional, Any, NamedTuple, Union, Callable
from contextlib import contextmanager
from abc import ABC, abstractmethod
from src.file_processor.models.audio import AudioFormat, AudioInfo
from src.file_processor.factories.audio import DefaultAudioFactory
from src.file_processor.audio_processor import AudioProcessor
from src.file_processor.document_processor import DocumentProcessor
from src.file_processor.video_processor import VideoProcessor
from src.file_processor.models.document import DocumentInfo
from src.file_processor.models.video import VideoInfo
from src.file_processor.factories.document import DefaultDocumentFactory
from src.file_processor.factories.video import DefaultVideoFactory
from src.file_processor.factories.file import FileFactory
from src.file_processor.commands.preview import PreviewCommand

@dataclass
class FolderInfo:
    """Data class to hold folder information."""
    path: Path
    name: str
    size: int
    file_count: int
    files: List[str]

@dataclass
class TrackInfo:
    """Data class to hold track information."""
    path: Path
    name: str
    size: int
    normalized_name: str
    hash: Optional[str] = None

@dataclass
class ProcessingStats:
    """Data class to hold processing statistics."""
    total_processed: int = 0
    total_cleaned: int = 0
    last_run: Optional[str] = None

@dataclass
class ProcessingResult:
    """Data class to hold processing results."""
    processed_groups: List[List[Path]]
    processed_count: int
    cleaned_count: int

class FolderIterator:
    """Iterator class for processing folders."""
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self._folders: List[FolderInfo] = []
        self._current_index = 0

    def __iter__(self) -> 'FolderIterator':
        return self

    def __next__(self) -> FolderInfo:
        if self._current_index >= len(self._folders):
            raise StopIteration
        folder = self._folders[self._current_index]
        self._current_index += 1
        return folder

    def load_folders(self) -> None:
        """Load all folders from the root path."""
        try:
            self._folders = [
                folder_info
                for item in os.listdir(self.root_path)
                if self._is_valid_folder(item)
                if (folder_info := self._create_folder_info(item)) is not None
            ]
        except (PermissionError, OSError) as e:
            print(f"Error accessing root path {self.root_path}: {e}")
            self._folders = []

    def _is_valid_folder(self, item: str) -> bool:
        """Check if an item is a valid folder."""
        return (self.root_path / item).is_dir()

    def _create_folder_info(self, item: str) -> Optional[FolderInfo]:
        """Create a FolderInfo object for a given item."""
        try:
            full_path = self.root_path / item
            if not full_path.exists():
                print(f"Warning: Path {full_path} does not exist")
                return None
            return FolderInfo(
                path=full_path,
                name=item,
                size=get_folder_size(full_path),
                file_count=count_files_in_folder(full_path),
                files=self._get_folder_files(full_path)
            )
        except (PermissionError, OSError) as e:
            print(f"Warning: Could not process folder {item}: {e}")
            return None

    def _get_folder_files(self, folder_path: Path) -> List[str]:
        """Get the first 5 files from a folder."""
        try:
            return [
                f.name for f in folder_path.rglob('*')
                if f.is_file()
            ][:5]
        except (PermissionError, OSError) as e:
            print(f"Warning: Could not access files in {folder_path}: {e}")
            return []

class TrackIterator:
    """Iterator class for processing tracks."""
    def __init__(self, folder_paths: List[Path]):
        self.folder_paths = folder_paths
        self._tracks: List[AudioInfo] = []
        self._current_index = 0
        self._audio_processor = AudioProcessor()

    def __iter__(self) -> 'TrackIterator':
        return self

    def __next__(self) -> AudioInfo:
        if self._current_index >= len(self._tracks):
            raise StopIteration
        track = self._tracks[self._current_index]
        self._current_index += 1
        return track

    def load_tracks(self) -> None:
        """Load all tracks from the given folders."""
        self._tracks = [
            track_info
            for folder_path in self.folder_paths
            for track_info in self._audio_processor.get_files(folder_path)
        ]

def get_folder_files(folder_path: Path, limit: int = 5) -> List[str]:
    """Get a list of files from a folder up to a limit."""
    return [
        f.name for f in folder_path.rglob('*')
        if f.is_file()
    ][:limit]

def create_track_info(file_path: Path) -> TrackInfo:
    """Create a TrackInfo object for a given file."""
    # Instantiate AudioProcessor to use its normalize_name method
    audio_processor = AudioProcessor() 
    return TrackInfo(
        path=file_path,
        name=file_path.stem,
        size=get_file_size(file_path),
        normalized_name=audio_processor.normalize_name(file_path.stem) # Use instance method
    )

@contextmanager
def process_folder_chunk(folder_chunk: List[Tuple[str, Path]], 
                        executor: ProcessPoolExecutor) -> Iterator[List[Tuple[Path, Path]]]:
    """Context manager for processing folder chunks."""
    try:
        futures = [
            executor.submit(is_likely_same_content, p1, p2) 
            for p1, p2 in folder_chunk
        ]
        results = [
            (p1, p2)
            for future, (p1, p2) in zip(futures, folder_chunk)
            if future.result()
        ]
        yield results
    finally:
        pass

class FileProcessorConfig:
    """Configuration manager for file processing."""
    def __init__(self, file_root: Path):
        self.file_root = file_root
        self.config_dir = self.file_root / '.file_processor'
        self.progress_file = self.config_dir / 'progress.json'
        self.config_file = self.config_dir / 'config.json'
        self.processed_folders: Set[str] = set()
        self.last_run: Optional[str] = None
        self.ensure_config_dir()
        self.load_progress()

    def ensure_config_dir(self) -> None:
        """Create the configuration directory if it doesn't exist."""
        try:
            self.config_dir.mkdir(exist_ok=True)
            print(f"Configuration directory created at: {self.config_dir}")
            
            if not self.config_file.exists():
                self.save_config(ProcessingStats())
                print("Configuration file initialized.")
        except Exception as e:
            print(f"Error creating configuration directory: {e}")
            raise

    def load_progress(self) -> None:
        """Load the progress from the progress file."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    self.processed_folders = set(data.get('processed_folders', []))
                    self.last_run = data.get('last_run')
                print(f"Loaded progress for {len(self.processed_folders)} processed folders.")
            except Exception as e:
                print(f"Error loading progress file: {e}")
                self.processed_folders = set()
        else:
            print("No progress file found. Starting fresh.")
            self.processed_folders = set()

    def save_progress(self) -> None:
        """Save the current progress to the progress file."""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump({
                    'processed_folders': list(self.processed_folders),
                    'last_run': datetime.now().isoformat()
                }, f, indent=2)
            print(f"Progress saved: {len(self.processed_folders)} folders processed.")
        except Exception as e:
            print(f"Error saving progress file: {e}")

    def save_config(self, config_data: ProcessingStats) -> None:
        """Save configuration data."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config_data.__dict__, f, indent=2)
            print("Configuration saved successfully.")
        except Exception as e:
            print(f"Error saving configuration: {e}")
            # Decide if we want to raise here or handle

    def load_config(self) -> ProcessingStats:
        """Load configuration data."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    return ProcessingStats(**data)
            except Exception as e:
                print(f"Error loading configuration file: {e}. Returning default.")
        return ProcessingStats()

    def is_folder_processed(self, folder_path: Path) -> bool:
        """Check if a folder has already been processed."""
        return str(folder_path) in self.processed_folders

    def mark_folder_processed(self, folder_path: Path) -> None:
        """Mark a folder as processed."""
        self.processed_folders.add(str(folder_path))
        self.save_progress()

    def update_stats(self, cleaned_count: int = 0) -> None:
        """Update processing statistics."""
        config = self.load_config()
        config.last_run = datetime.now().isoformat()
        # Assuming total_processed is managed elsewhere or by number of groups
        # config.total_processed = len(self.processed_folders) # This might not be right if reset
        config.total_cleaned += cleaned_count
        self.save_config(config)

def normalize_string(s: str) -> str:
    """Normalize string by removing special characters and common words."""
    s = convert_to_ascii(s)
    s = remove_special_chars(s)
    s = remove_common_words(s)
    return s

def convert_to_ascii(s: str) -> str:
    """Convert string to ASCII, removing accents and special characters."""
    return unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('ASCII')

def remove_special_chars(s: str) -> str:
    """Remove special characters from string."""
    return re.sub(r'[^a-z0-9\s]', '', s.lower())

def remove_common_words(s: str) -> str:
    """Remove common words that might differ between versions."""
    return re.sub(r'\b(disc|cd|disc\s*\d+|cd\s*\d+)\b', '', s)

def are_strings_similar(a: str, b: str, threshold: float = 0.85) -> bool:
    """Compare two strings and return True if they are similar enough."""
    a = normalize_string(a)
    b = normalize_string(b)
    
    if is_one_contained_in_other(a, b):
        return True
        
    if are_strings_identical(a, b):
        return True
        
    return are_strings_similar_by_ratio(a, b, threshold)

def is_one_contained_in_other(a: str, b: str) -> bool:
    """Check if one string is contained within the other."""
    return a in b or b in a

def are_strings_identical(a: str, b: str) -> bool:
    """Check if strings are identical after normalization."""
    return a == b

def are_strings_similar_by_ratio(a: str, b: str, threshold: float) -> bool:
    """Check if strings are similar by ratio and acceptable difference."""
    ratio = SequenceMatcher(None, a, b).ratio()
    if ratio > threshold:
        diff = set(a.split()) ^ set(b.split())
        return is_acceptable_difference(diff)
    return False

def is_acceptable_difference(diff: Set[str]) -> bool:
    """Check if the difference between strings is acceptable."""
    return all(
        re.match(r'^\d+$', word) or 
        word in ['remastered', 'remaster', 'deluxe', 'edition'] 
        for word in diff
    )

def get_base_folder_name(folder_name: str) -> str:
    """Remove (1), (1)(1), etc. from folder names."""
    return re.sub(r'\s*\(\d+\)(?:\s*\(\d+\))?$', '', folder_name)

def count_files_in_folder(folder_path: Path) -> int:
    """Count files in a folder."""
    return len([
        f for f in Path(folder_path).rglob('*') 
        if f.is_file()
    ])

def get_file_size(file_path: Path) -> int:
    """Get file size in bytes."""
    return os.path.getsize(file_path)

def get_folder_size(folder_path: Path) -> int:
    """Calculate total size of a folder, excluding symlinks."""
    return sum(
        os.path.getsize(fp)
        for dirpath, _, filenames in os.walk(folder_path)
        for f in filenames
        if not os.path.islink(fp := os.path.join(dirpath, f))
    )

def get_audio_files(folder_path: Path) -> List[str]:
    """Get list of audio files in a folder using the abstract factory."""
    audio_factory = DefaultAudioFactory()
    audio_extensions = {fmt.value for fmt in AudioFormat}
    return [
        file
        for file in os.listdir(folder_path)
        if os.path.splitext(file)[1].lower() in audio_extensions
    ]

def process_file_chunk(file_paths: List[Path], chunk_size: int = 65536) -> List[Tuple[Path, str]]:
    """Process a chunk of files to calculate their hashes."""
    return [
        (file_path, file_hash)
        for file_path in file_paths
        if (file_hash := calculate_file_hash_safe(file_path, chunk_size)) is not None
    ]

def calculate_file_hash_safe(file_path: Path, chunk_size: int) -> Optional[str]:
    """Safely calculate file hash with error handling."""
    try:
        return FileFactory.calculate_file_hash(file_path, chunk_size)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_folder_chunk(chunk: List[FolderInfo], 
                        folders: List[Tuple[str, Path]], 
                        chunk_size: int, 
                        processed: Set[Path], 
                        executor: ProcessPoolExecutor, 
                        pbar: tqdm) -> List[List[Path]]:
    """Process a chunk of folders and return any similar groups found."""
    return [
        group
        for folder_info in chunk
        if folder_info.path not in processed
        if (group := process_single_folder(folder_info.name, folder_info.path, folders, chunk_size, processed, executor, pbar))
    ]

def process_single_folder(name1: str, 
                         path1: Path, 
                         folders: List[Tuple[str, Path]], 
                         chunk_size: int, 
                         processed: Set[Path], 
                         executor: ProcessPoolExecutor, 
                         pbar: tqdm) -> Optional[List[Path]]:
    """Process a single folder and return a group if similar folders are found."""
    current_group = [path1]
    processed.add(path1)
    
    similar_folders = find_similar_folders_for_path(path1, folders[chunk_size + 1:], processed, pbar)
    if not similar_folders:
        return None
        
    batch_results = process_folder_batch(similar_folders, executor)
    current_group.extend(get_new_folders_from_batch(batch_results, processed))
    
    return current_group if len(current_group) > 1 else None

def find_similar_folders_for_path(path1: Path, 
                                remaining_folders: List[FolderInfo], 
                                processed: Set[Path], 
                                pbar: tqdm) -> List[Tuple[Path, Path]]:
    """Find similar folders for a given path."""
    similar_folders = []
    for folder_info in remaining_folders:
        path2 = folder_info.path
        if path2 in processed:
            continue
        if similar(path1, path2):
            similar_folders.append((path1, path2))
        pbar.update(1)
    return similar_folders

def get_new_folders_from_batch(batch_results: List[Tuple[Path, Path]], 
                             processed: Set[Path]) -> List[Path]:
    """Get new folders from batch results that haven't been processed."""
    new_folders = []
    for _, path2 in batch_results:
        if path2 not in processed:
            new_folders.append(path2)
            processed.add(path2)
    return new_folders

def is_likely_same_content(path1: Path, path2: Path) -> bool:
    """Verify if two folders likely contain the same content."""
    files1 = get_folder_files(path1)[:5]
    files2 = get_folder_files(path2)[:5]
    
    if set(files1) & set(files2):
        return True
    
    size1 = get_folder_size(path1)
    size2 = get_folder_size(path2)
    count1 = count_files_in_folder(path1)
    count2 = count_files_in_folder(path2)
    
    if max(size1, size2) == 0 or max(count1, count2) == 0:
        return False

    size_ratio = min(size1, size2) / max(size1, size2)
    count_ratio = min(count1, count2) / max(count1, count2)
    
    return size_ratio > 0.8 and count_ratio > 0.8

def similar(path1: Path, path2: Path) -> bool:
    """Determine if two folders are similar."""
    return is_likely_same_content(path1, path2)

def find_similar_folders(root_path: Path) -> List[List[Path]]:
    """Find groups of similar folders in the given root path."""
    print("\n=== Scanning for similar folders ===")
    
    folder_iterator = FolderIterator(root_path)
    folder_iterator.load_folders()
    
    if not folder_iterator._folders:
        print("No folders found to analyze.")
        return []
        
    print(f"\nFound {len(folder_iterator._folders)} folders to analyze")
    print("\n=== Analyzing folder similarities ===")
    
    num_workers, chunk_size = calculate_worker_settings(len(folder_iterator._folders))
    folder_chunks = create_folder_chunks(folder_iterator._folders, chunk_size)
    total_comparisons = calculate_total_comparisons(folder_iterator._folders, chunk_size)
    
    print(f"\nProcessing folders using {num_workers} workers...")
    similar_groups = []
    processed = set()
    
    with tqdm(total=total_comparisons, desc="Comparing folders") as pbar:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for chunk in folder_chunks:
                results = process_folder_chunk(chunk, folder_iterator._folders, chunk_size, processed, executor, pbar)
                for group in results:
                    similar_groups.append(group)
                pbar.update(len(chunk))
    
    print(f"\nFound {len(similar_groups)} groups of similar folders")
    return similar_groups

def find_duplicate_tracks_across_folders(folder_group: List[Path]) -> Dict[str, List[Path]]:
    """Find duplicate tracks across a group of folders."""
    track_iterator = TrackIterator(folder_group)
    track_iterator.load_tracks()

    tracks_by_name: Dict[str, List[Path]] = defaultdict(list)
    for track_info in track_iterator:
        # Assuming track_info has a normalized_name and path attribute
        tracks_by_name[track_info.normalized_name].append(track_info.path)

    duplicate_tracks: Dict[str, List[Path]] = {}
    for name, paths in tracks_by_name.items():
        if len(paths) > 1:
            # Further check if these are actual duplicates (e.g. by hash or size)
            # For now, just matching by name
            # We might want to ensure they are from different original folders if folder_group can contain subfolders of a processed unit
            unique_parent_folders = {p.parent for p in paths}
            if len(unique_parent_folders) > 1: # Ensure paths are from different main folders in the group
                 duplicate_tracks[name] = paths
    return duplicate_tracks

def preview_changes(root_path: Path) -> List[str]:
    """Preview changes that would be made to the music folder."""
    print("\n=== Starting preview of changes ===")
    changes = []
    
    similar_folder_groups = find_similar_folders(root_path)
    
    for group_idx, folder_group in enumerate(similar_folder_groups, 1):
        print(f"\nProcessing group {group_idx} of {len(similar_folder_groups)}")
        folder_changes = []
        folder_changes.append(f"\nSimilar folders found:")
        for folder in folder_group:
            folder_changes.append(f"  - {folder}")
        
        folder_sizes = [(path, count_files_in_folder(path)) for path in folder_group]
        folder_sizes.sort(key=lambda x: x[1], reverse=True)
        
        keep_folder = folder_sizes[0][0]
        folder_changes.append(f"\nWill keep folder: {keep_folder} with {folder_sizes[0][1]} files")
        
        duplicate_tracks = find_duplicate_tracks_across_folders(folder_group)
        
        if duplicate_tracks:
            folder_changes.append("\nDuplicate tracks found:")
            for track_name, file_paths in duplicate_tracks.items():
                folder_changes.append(f"\n  Track: {track_name}")
                file_paths.sort(key=get_file_size, reverse=True)
                
                keep_file = file_paths[0]
                folder_changes.append(f"    Will keep: {keep_file} ({get_file_size(keep_file)} bytes)")
                
                for file_path in file_paths[1:]:
                    folder_changes.append(f"    Will remove: {file_path} ({get_file_size(file_path)} bytes)")
            
            for folder_path, file_count in folder_sizes[1:]:
                if count_files_in_folder(folder_path) == 0:
                    folder_changes.append(f"\n  Will remove empty folder: {folder_path}")
        
        changes.extend(folder_changes)
    
    print("\n=== Preview complete ===")
    return changes

def get_music_folder_path(args: argparse.Namespace) -> Optional[Path]:
    """Get the music folder path from arguments or default location."""
    if args.path:
        music_folder = Path(args.path).resolve()
        if not music_folder.exists():
            print(f"Error: Path '{music_folder}' does not exist.")
            return None
        if not music_folder.is_dir():
            print(f"Error: Path '{music_folder}' is not a directory.")
            return None
        return music_folder
    
    d_drive = Path('/mnt/d')
    if not d_drive.exists():
        print("D: drive not found. Please make sure it's properly mounted or use --path to specify a different location.")
        return None

    for item in os.listdir(d_drive):
        if item.lower() == 'music_backup':
            music_folder = d_drive / item / 'music'
            if music_folder.exists():
                return music_folder

    print("Music folder not found in D: drive. Please use --path to specify the music folder location.")
    return None

class Command(ABC):
    """Base class for command pattern."""
    @abstractmethod
    def add_to_parser(self, parser: argparse.ArgumentParser) -> None:
        """Add command arguments to the parser."""
        pass

    @abstractmethod
    def execute(self, args: argparse.Namespace) -> None:
        """Execute the command."""
        pass

class CleanupCommand(Command):
    """Command for cleaning up music files."""
    def add_to_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('-y', '--yes', action='store_true', 
                          help='Automatically answer yes to all prompts')
        parser.add_argument('--reset', action='store_true',
                          help='Reset progress and start fresh')
        parser.add_argument('--path', type=str,
                          help='Path to the music folder (absolute or relative)')

    def execute(self, args: argparse.Namespace) -> None:
        print("=== Starting music folder cleanup ===")
        
        music_folder = get_music_folder_path(args)
        if not music_folder:
            return

        print(f"Using music folder: {music_folder}")
        result = process_music_folder(music_folder, args)
        display_processing_results(result)

class CommandFactory:
    """Factory for creating command objects."""
    _commands: Dict[str, Command] = {
        'cleanup': CleanupCommand(),
        'preview': PreviewCommand()
    }

    @classmethod
    def get_command(cls, command_name: str) -> Optional[Command]:
        """Get a command by name."""
        return cls._commands.get(command_name)

    @classmethod
    def register_command(cls, name: str, command: Command) -> None:
        """Register a new command."""
        cls._commands[name] = command

def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all commands."""
    parser = argparse.ArgumentParser(description='Music folder cleanup and management tool.')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    for command_name, command in CommandFactory._commands.items():
        subparser = subparsers.add_parser(command_name, help=f'{command_name} command')
        command.add_to_parser(subparser)
    
    return parser

def main() -> None:
    """Main function to run the music cleanup script."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
        
    command = CommandFactory.get_command(args.command)
    if command:
        command.execute(args)
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()

def process_music_folder(music_folder: Path, args: argparse.Namespace) -> ProcessingResult:
    """Process the music folder and return results."""
    config = FileProcessorConfig(music_folder)
    
    if args.reset:
        reset_processing_progress(config)
    
    audio_processor = AudioProcessor()
    document_processor = DocumentProcessor()
    video_processor = VideoProcessor()
    
    similar_groups = find_similar_folders(music_folder)
    if not similar_groups:
        print("No similar folders found!")
        return ProcessingResult([], 0, 0)
    
    return process_similar_groups(similar_groups, config, args.yes)

def reset_processing_progress(config: FileProcessorConfig) -> None:
    """Reset the processing progress."""
    print("\nResetting progress...")
    config.processed_folders = set()
    config.save_progress()
    print("Progress reset complete.")

def process_similar_groups(similar_groups: List[List[Path]],
                         config: FileProcessorConfig,
                         auto_yes: bool) -> ProcessingResult:
    """Process groups of similar folders."""
    print(f"\nFound {len(similar_groups)} groups of similar folders")
    
    processed_count = 0
    cleaned_count = 0
    
    for group_idx, folder_group in enumerate(similar_groups, 1):
        print(f"\n=== Processing group {group_idx} of {len(similar_groups)} ===")
        result = process_and_clean_group(folder_group, config, auto_yes)
        processed_count += result.processed_count
        cleaned_count += result.cleaned_count
    
    return ProcessingResult(similar_groups, processed_count, cleaned_count)

def display_processing_results(result: ProcessingResult) -> None:
    """Display the results of processing."""
    print("\n=== Cleanup Statistics ===")
    print(f"Total folders processed: {result.processed_count}")
    print(f"Total items cleaned: {result.cleaned_count}")
    print("\n=== Cleanup complete ===")

def calculate_worker_settings(folder_count: int) -> Tuple[int, int]:
    """Calculate optimal number of workers and chunk size."""
    num_cores = multiprocessing.cpu_count()
    num_workers = min(num_cores * 4, 60)  # Cap at 60 workers for Windows
    chunk_size = max(1, folder_count // (num_workers * 8))
    return num_workers, chunk_size

def create_folder_chunks(folders: List[FolderInfo], chunk_size: int) -> List[List[FolderInfo]]:
    """Split FolderInfo objects into chunks for parallel processing."""
    return [
        folders[i:i + chunk_size]
        for i in range(0, len(folders), chunk_size)
    ]

def calculate_total_comparisons(folders: List[FolderInfo], chunk_size: int) -> int:
    """Calculate total number of folder comparisons needed."""
    n = len(folders)
    return n * (n - 1) // 2

def process_folder_batch(folder_pairs: list[tuple[Path, Path]], executor: ProcessPoolExecutor) -> list[tuple[Path, Path]]:
    """Process a batch of folder pairs in parallel and return those that are likely the same content."""
    futures = [executor.submit(is_likely_same_content, p1, p2) for p1, p2 in folder_pairs]
    results = []
    for future, (p1, p2) in zip(futures, folder_pairs):
        try:
            if future.result():
                results.append((p1, p2))
        except Exception as e:
            print(f"Error processing folder pair {p1}, {p2}: {e}")
    return results

def process_and_clean_group(folder_group: List[Path], config: FileProcessorConfig, auto_yes: bool) -> ProcessingResult:
    """Process and clean a group of similar folders."""
    processed_count = 0
    cleaned_count = 0
    
    # Sort folders by number of files (keep the one with most files)
    folder_sizes = [(path, count_files_in_folder(path)) for path in folder_group]
    folder_sizes.sort(key=lambda x: x[1], reverse=True)
    
    keep_folder = folder_sizes[0][0]
    print(f"\nKeeping folder: {keep_folder} with {folder_sizes[0][1]} files")
    
    # Find duplicate tracks across folders
    duplicate_tracks = find_duplicate_tracks_across_folders(folder_group)
    
    if duplicate_tracks:
        print("\nDuplicate tracks found:")
        for track_name, file_paths in duplicate_tracks.items():
            print(f"\n  Track: {track_name}")
            file_paths.sort(key=get_file_size, reverse=True)
            
            keep_file = file_paths[0]
            print(f"    Keeping: {keep_file} ({get_file_size(keep_file)} bytes)")
            
            for file_path in file_paths[1:]:
                if auto_yes or input(f"    Remove {file_path} ({get_file_size(file_path)} bytes)? [y/N] ").lower() == 'y':
                    try:
                        file_path.unlink()
                        print(f"    Removed: {file_path}")
                        cleaned_count += 1
                    except Exception as e:
                        print(f"    Error removing {file_path}: {e}")
    
    # Remove empty folders
    for folder_path, file_count in folder_sizes[1:]:
        if count_files_in_folder(folder_path) == 0:
            if auto_yes or input(f"\nRemove empty folder {folder_path}? [y/N] ").lower() == 'y':
                try:
                    # folder_path.rmdir() # Old method, only for empty dirs
                    shutil.rmtree(folder_path) # Use shutil.rmtree to remove recursively
                    print(f"Removed folder: {folder_path}") # Updated message
                    cleaned_count += 1
                except Exception as e:
                    print(f"Error removing folder {folder_path}: {e}")
    
    # After removing individual files, re-check folders that were not initially empty
    # and remove them if they have become empty.
    for folder_path, _ in folder_sizes[1:]: # Iterate through all non-kept folders
        if folder_path.exists() and count_files_in_folder(folder_path) == 0:
            # No need to ask again if auto_yes was true, but for safety, 
            # if not auto_yes, this prompts for folders that became empty after track deletions.
            # Consider if this second prompt is desired or if initial prompt for non-empty folders is enough.
            # For now, let's assume if it's empty now, it's safe to remove if confirmed.
            if auto_yes or input(f"\nRemove folder {folder_path} (now empty after track deletions)? [y/N] ").lower() == 'y':
                try:
                    shutil.rmtree(folder_path)
                    print(f"Removed folder (became empty): {folder_path}")
                    cleaned_count += 1 # This might double count if already counted as an "empty folder" before track deletion
                                      # The logic for cleaned_count for folders needs review if precise counts are critical.
                                      # However, for functionality, this ensures removal.
                except Exception as e:
                    print(f"Error removing folder (became empty) {folder_path}: {e}")

    processed_count = len(folder_group)
    return ProcessingResult([folder_group], processed_count, cleaned_count)

if __name__ == "__main__":
    main() 