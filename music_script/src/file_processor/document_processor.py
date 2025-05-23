from pathlib import Path
from typing import List, Optional
from .factories.base import AbstractProcessor
from .models.document import DocumentInfo, DocumentFormat
from .factories.document import DefaultDocumentFactory

class DocumentProcessor(AbstractProcessor[DocumentInfo]):
    def __init__(self):
        super().__init__(factory=DefaultDocumentFactory(), file_type_enum=DocumentFormat)

    # get_files and process_file are now inherited from AbstractProcessor
    # _is_document_file is effectively replaced by _is_relevant_file in AbstractProcessor 