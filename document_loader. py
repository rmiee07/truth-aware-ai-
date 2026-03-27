"""
document_loader.py
==================
Handles loading documents from various formats (PDF, TXT, DOCX, web URL)
and splitting them into overlapping chunks for embedding.

Supported sources:
  - Plain text (.txt)
  - PDF files (.pdf)            → requires: pip install pymupdf
  - Word documents (.docx)      → requires: pip install python-docx
  - Web URLs                    → requires: pip install requests beautifulsoup4
  - Raw strings (for quick testing)
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional
from config import CHUNKING


# ---------------------------------------------------------------------------
# Data structure for a single document chunk
# ---------------------------------------------------------------------------
@dataclass
class DocumentChunk:
    """Represents one piece of a larger document after chunking."""
    text: str
    source: str             # file path or URL the chunk came from
    chunk_id: int           # position of this chunk within its source document
    metadata: dict = field(default_factory=dict)  # any extra info (page no., title, etc.)

    def __repr__(self):
        preview = self.text[:80].replace("\n", " ")
        return f"<Chunk id={self.chunk_id} src='{self.source}' | '{preview}...'>"


# ---------------------------------------------------------------------------
# Text cleaner
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Normalize whitespace and remove common OCR/PDF artefacts.
    Does NOT remove meaningful punctuation or sentence structure.
    """
    text = re.sub(r'\s+', ' ', text)           # collapse multiple spaces / newlines
    text = re.sub(r'[^\x00-\x7F]+', ' ', text) # strip non-ASCII (optional — comment out for multilingual)
    text = text.strip()
    return text


# ---------------------------------------------------------------------------
# Core chunker
# ---------------------------------------------------------------------------
def chunk_text(
    text: str,
    source: str,
    chunk_size: int  = CHUNKING.chunk_size,
    overlap: int     = CHUNKING.chunk_overlap,
    min_len: int     = CHUNKING.min_chunk_len,
) -> List[DocumentChunk]:
    """
    Split `text` into overlapping character-level chunks.

    Strategy: slide a window of `chunk_size` characters across the text,
    stepping by (chunk_size - overlap) each time. This ensures adjacent
    chunks share some context, preventing facts from being cut at boundaries.

    Args:
        text       : full document text (pre-cleaned)
        source     : file path / URL label for provenance tracking
        chunk_size : maximum characters per chunk
        overlap    : characters shared between consecutive chunks
        min_len    : discard chunks shorter than this (usually headers/footers)

    Returns:
        List of DocumentChunk objects
    """
    text = clean_text(text)
    if not text:
        return []

    chunks: List[DocumentChunk] = []
    step = max(1, chunk_size - overlap)
    chunk_id = 0

    for start in range(0, len(text), step):
        end = min(start + chunk_size, len(text))
        fragment = text[start:end].strip()

        if len(fragment) < min_len:
            continue  # skip tiny trailing fragments

        chunks.append(DocumentChunk(
            text=fragment,
            source=source,
            chunk_id=chunk_id,
        ))
        chunk_id += 1

        if end == len(text):
            break  # reached the end

    return chunks


# ---------------------------------------------------------------------------
# Format-specific loaders
# ---------------------------------------------------------------------------

def load_txt(file_path: str) -> str:
    """Load a plain text file."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def load_pdf(file_path: str) -> str:
    """
    Load a PDF using PyMuPDF (fitz).
    Install: pip install pymupdf
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")

    doc = fitz.open(file_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text("text")
        if page_text.strip():
            pages.append(f"[Page {page_num}]\n{page_text}")
    doc.close()
    return "\n".join(pages)


def load_docx(file_path: str) -> str:
    """
    Load a Word document using python-docx.
    Install: pip install python-docx
    """
    try:
        from docx import Document
    except ImportError:
        raise ImportError("python-docx not installed. Run: pip install python-docx")

    doc = Document(file_path)
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(paragraphs)


def load_url(url: str) -> str:
    """
    Fetch and parse a web page as plain text.
    Install: pip install requests beautifulsoup4
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("Missing packages. Run: pip install requests beautifulsoup4")

    headers = {"User-Agent": "Mozilla/5.0 (research bot)"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove scripts, styles, nav, footer — keep main content
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    return soup.get_text(separator=" ")


# ---------------------------------------------------------------------------
# High-level: load any source → list of chunks
# ---------------------------------------------------------------------------

def load_and_chunk(source: str) -> List[DocumentChunk]:
    """
    Dispatch to the correct loader based on file extension or URL scheme,
    then chunk the resulting text.

    Args:
        source : file path (str) or URL (starts with http/https)
                 or a plain string prefixed with "text:" for raw input

    Returns:
        List[DocumentChunk]
    """
    source = source.strip()

    # --- Raw string input (for testing / demos) ---
    if source.startswith("text:"):
        raw_text = source[5:]  # strip the "text:" prefix
        return chunk_text(raw_text, source="raw_input")

    # --- URL ---
    if source.startswith("http://") or source.startswith("https://"):
        print(f"[Loader] Fetching URL: {source}")
        raw_text = load_url(source)
        return chunk_text(raw_text, source=source)

    # --- File ---
    if not os.path.isfile(source):
        raise FileNotFoundError(f"File not found: {source}")

    ext = os.path.splitext(source)[-1].lower()
    print(f"[Loader] Loading file: {source}  (format: {ext})")

    if ext == ".txt":
        raw_text = load_txt(source)
    elif ext == ".pdf":
        raw_text = load_pdf(source)
    elif ext in (".docx", ".doc"):
        raw_text = load_docx(source)
    else:
        # Fallback: try reading as plain text
        print(f"[Loader] Unknown extension '{ext}', attempting plain text read.")
        raw_text = load_txt(source)

    return chunk_text(raw_text, source=source)


def load_multiple(sources: List[str]) -> List[DocumentChunk]:
    """
    Load and chunk multiple sources.
    All chunks are returned as a flat list (each chunk retains its source label).
    """
    all_chunks: List[DocumentChunk] = []
    for src in sources:
        try:
            chunks = load_and_chunk(src)
            all_chunks.extend(chunks)
            print(f"[Loader] '{src}' → {len(chunks)} chunks")
        except Exception as e:
            print(f"[Loader] ERROR loading '{src}': {e}")
    print(f"[Loader] Total chunks: {len(all_chunks)}")
    return all_chunks


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Test with raw text input
    sample = (
        "text:Retrieval-Augmented Generation (RAG) combines a retrieval system "
        "with a generative language model. The retriever fetches relevant documents "
        "from a knowledge base using vector similarity search. These documents are "
        "then passed as context to the language model, which generates a grounded response. "
        "This reduces hallucinations significantly compared to vanilla LLM inference."
    )
    chunks = load_and_chunk(sample)
    for c in chunks:
        print(c)
