import argparse
from pathlib import Path

from .commands.preview import PreviewCommand
from .commands.command_processor import CommandProcessor

def main():
    parser = argparse.ArgumentParser(
        description="A tool for processing and organizing files by type (audio, video, documents).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all file types in a directory
  python -m src.file_processor /path/to/source --filetype all

  # Process only audio files
  python -m src.file_processor /path/to/source --filetype audio

  # Preview changes for video files
  python -m src.file_processor /path/to/source --filetype video --preview

  # Fast processing of documents
  python -m src.file_processor /path/to/source --filetype document --fast
        """
    )
    parser.add_argument("source_dir", type=str, help="Source directory to process")
    parser.add_argument("--target-dir", type=str, help="Target directory for processed files")
    parser.add_argument("--preview", action="store_true", help="Preview changes without making them")
    parser.add_argument("--fast", action="store_true", help="Skip file counting and use fast processing")
    parser.add_argument("--filetype", type=str, choices=["audio", "video", "document", "all"], 
                       default="all", help="Type of files to process (default: all)")

    args = parser.parse_args()
    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir) if args.target_dir else None

    if not source_dir.exists():
        print(f"Error: Source directory '{source_dir}' does not exist")
        return

    if args.preview:
        command = PreviewCommand(source_dir, target_dir, fast=args.fast, file_type=args.filetype)
    else:
        command = CommandProcessor(source_dir, target_dir, file_type=args.filetype)

    command.execute()

if __name__ == "__main__":
    main() 