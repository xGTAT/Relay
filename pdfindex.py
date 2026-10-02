"""PDF text extraction and chunking for course material."""

import io
import re

from pypdf import PdfReader


def extract_pages(data: bytes) -> list[str]:
    """Text of each page, in order. Pages with no extractable text come back empty."""
    reader = PdfReader(io.BytesIO(data))
    return [(page.extract_text() or "") for page in reader.pages]


def chunk_pages(pages: list[str], size: int = 1200, overlap: int = 150) -> list[tuple[int, str]]:
    """Split page text into overlapping chunks. Returns (page_number, text), pages numbered from 1."""
    chunks: list[tuple[int, str]] = []
    for number, raw in enumerate(pages, start=1):
        text = re.sub(r"\s+", " ", raw).strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(len(text), start + size)
            chunks.append((number, text[start:end]))
            if end == len(text):
                break
            start = end - overlap
    return chunks
