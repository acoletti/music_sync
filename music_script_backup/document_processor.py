from pathlib import Path
from typing import List, Optional
from src.file_processor.factories.base import AbstractProcessor
from src.file_processor.models.document import DocumentInfo, DocumentFormat
from src.file_processor.factories.document import DefaultDocumentFactory

class DocumentProcessor(AbstractProcessor):
    def __init__(self):
        self._document_factory = DefaultDocumentFactory()

    def process_file(self, file_path: Path) -> Optional[DocumentInfo]:
        return self._document_factory.create(file_path)

    def get_files(self, folder_path: Path) -> List[DocumentInfo]:
        return [
            self.process_file(file_path)
            for file_path in folder_path.rglob('*')
            if self._is_document_file(file_path)
        ]

    def _is_document_file(self, file_path: Path) -> bool:
        return file_path.is_file() and file_path.suffix.lower() in {fmt.value for fmt in DocumentFormat} 