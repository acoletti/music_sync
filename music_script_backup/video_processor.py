from pathlib import Path
from typing import List, Optional
from src.file_processor.factories.base import AbstractProcessor
from src.file_processor.models.video import VideoInfo, VideoFormat
from src.file_processor.factories.video import DefaultVideoFactory

class VideoProcessor(AbstractProcessor):
    def __init__(self):
        self._video_factory = DefaultVideoFactory()

    def process_file(self, file_path: Path) -> Optional[VideoInfo]:
        return self._video_factory.create(file_path)

    def get_files(self, folder_path: Path) -> List[VideoInfo]:
        return [
            self.process_file(file_path)
            for file_path in folder_path.rglob('*')
            if self._is_video_file(file_path)
        ]

    def _is_video_file(self, file_path: Path) -> bool:
        return file_path.is_file() and file_path.suffix.lower() in {fmt.value for fmt in VideoFormat} 