from pathlib import Path
from typing import List, Optional
from .factories.base import AbstractProcessor
from .models.audio import AudioInfo, AudioFormat
from .factories.audio import DefaultAudioFactory

class AudioProcessor(AbstractProcessor[AudioInfo]):
    def __init__(self):
        super().__init__(factory=DefaultAudioFactory(), file_type_enum=AudioFormat)

    # get_files and process_file are now inherited from AbstractProcessor
    # _is_audio_file is effectively replaced by _is_relevant_file in AbstractProcessor 