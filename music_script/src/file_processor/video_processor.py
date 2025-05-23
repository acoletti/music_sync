from pathlib import Path
from typing import List, Optional
from .factories.base import AbstractProcessor
from .models.video import VideoInfo, VideoFormat
from .factories.video import DefaultVideoFactory

class VideoProcessor(AbstractProcessor[VideoInfo]):
    def __init__(self):
        super().__init__(factory=DefaultVideoFactory(), file_type_enum=VideoFormat)

    # get_files and process_file are now inherited from AbstractProcessor
    # _is_video_file is effectively replaced by _is_relevant_file in AbstractProcessor 