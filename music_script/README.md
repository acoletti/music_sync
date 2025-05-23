# Music Cleanup Tool

A Python tool for cleaning up and organizing music files. This tool helps identify and remove duplicate music files, organize folders, and maintain a clean music library.

## Features

- Scan music folders for duplicates
- Preview changes before making them
- Clean up duplicate files and empty folders
- Progress tracking and resumable operations
- Support for various audio formats (MP3, FLAC, WAV, DSF)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/music-cleanup.git
cd music-cleanup
```

2. Install the package:
```bash
pip install -e .
```

## Usage

### Preview Changes

To preview changes that would be made to your music folder:

```bash
music-cleanup preview --path "/path/to/music/folder"
```

### Clean Up Music Files

To clean up your music folder:

```bash
music-cleanup cleanup --path "/path/to/music/folder"
```

Additional options:
- `-y, --yes`: Automatically answer yes to all prompts
- `--reset`: Reset progress and start fresh

## Development

The project uses a modular architecture with the following components:

- `commands/`: Command implementations for different operations
- `factories/`: Factory classes for creating domain objects
- `models/`: Data models and domain objects
- `builders/`: Builder classes for constructing complex objects
- `utils/`: Utility functions and helpers

## License

This project is licensed under the MIT License - see the LICENSE file for details. 