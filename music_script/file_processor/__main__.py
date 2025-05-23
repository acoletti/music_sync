import argparse
from pathlib import Path

from .commands.preview import PreviewCommand
from .commands.command_processor import CommandProcessor

def main():
    parser = argparse.ArgumentParser(description="File processing and cleanup tool")
    parser.add_argument("source_dir", type=str, help="Source directory to process")
    parser.add_argument("--target-dir", type=str, help="Target directory for processed files")
    parser.add_argument("--preview", action="store_true", help="Preview changes without making them")
    parser.add_argument("--fast", action="store_true", help="Skip file counting and use fast processing")

    args = parser.parse_args()
    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir) if args.target_dir else None

    if not source_dir.exists():
        print(f"Error: Source directory '{source_dir}' does not exist")
        return

    if args.preview:
        command = PreviewCommand(source_dir, target_dir, fast=args.fast)
    else:
        command = CommandProcessor(source_dir, target_dir)

    command.execute()

if __name__ == "__main__":
    main() 