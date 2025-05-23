from abc import ABC, abstractmethod
import argparse
from pathlib import Path
from typing import Optional
import os

from ..builders.progress import ProgressManager

class Command(ABC):
    """Base class for all commands."""
    def __init__(self):
        self.progress_manager = ProgressManager()

    @abstractmethod
    def add_to_parser(self, parser: argparse.ArgumentParser) -> None:
        """Add command arguments to the parser."""
        pass

    @abstractmethod
    def execute(self, args: argparse.Namespace) -> None:
        """Execute the command."""
        pass

    def get_music_folder_path(self, args: argparse.Namespace) -> Optional[Path]:
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