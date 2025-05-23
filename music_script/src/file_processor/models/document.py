from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from enum import Enum

class DocumentFormat(Enum):
    """Enum for document file formats."""
    PDF = '.pdf'
    DOC = '.doc'
    DOCX = '.docx'
    TXT = '.txt'
    RTF = '.rtf'

@dataclass
class DocumentInfo:
    """Information about a document file."""
    path: Path
    name: str
    size: int
    normalized_name: str
    format: DocumentFormat
    hash: Optional[str] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None 