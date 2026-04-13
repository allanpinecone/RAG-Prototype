"""
Ingestion pipeline: extracts text from PDFs in the Documents directory,
chunks them with overlap, and upserts into Pinecone using integrated
inference (llama-text-embed-v2 @ 2048 dims).
"""

import os
import re
import hashlib
import time
from pathlib import Path

import fitz  # PyMuPDF
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "vdr-copilot")
DOCS_DIR = Path(__file__).resolve().parent.parent / "Documents"

CHUNK_SIZE = 1000  # chars
CHUNK_OVERLAP = 200
BATCH_SIZE = 50


def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """Extract text page-by-page from a PDF, returning list of {page, text}."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        if text.strip():
            pages.append({"page": page_num + 1, "text": text.strip()})
    doc.close()
    return pages


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E£€¥%°±²³µ·¼½¾]", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks at sentence boundaries when possible."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            boundary = text.rfind(". ", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if len(c) > 50]


def derive_fund_name(pdf_path: Path) -> str:
    """Derive a human-readable fund/document name from the file path."""
    name = pdf_path.stem
    name = name.replace("-", " ").replace("_", " ")
    name = re.sub(r"\b(en|gb|uk|eur|usd|gbp|pdf)\b", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+", " ", name).strip()
    return name or pdf_path.stem


def derive_fund_category(pdf_path: Path) -> str:
    """Categorise documents by type based on filename patterns."""
    fname = pdf_path.name.lower()
    if "prospectus" in fname:
        return "Prospectus"
    if "kiid" in fname or "kid" in fname:
        return "KIID / KID"
    if "factsheet" in fname:
        return "Factsheet"
    if "annual" in fname:
        return "Annual Report"
    if "interim" in fname:
        return "Interim Report"
    if "sdr" in fname or "cfd" in fname:
        return "SDR / CFD"
    if "instrument" in fname:
        return "Legal / Instrument"
    if "information_document" in fname or "information-document" in fname:
        return "Information Document"
    return "Fund Document"


def collect_pdfs(directory: Path) -> list[Path]:
    """Recursively collect all PDF files."""
    return sorted(directory.rglob("*.pdf")) + sorted(directory.rglob("*.PDF"))


def make_record_id(pdf_path: Path, chunk_idx: int) -> str:
    path_hash = hashlib.md5(str(pdf_path).encode()).hexdigest()[:10]
    return f"{path_hash}_chunk{chunk_idx:04d}"


def create_index(pc: Pinecone):
    """Create the Pinecone index with integrated llama-text-embed-v2 if it doesn't exist."""
    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME in existing:
        print(f"Index '{INDEX_NAME}' already exists, skipping creation.")
        return

    print(f"Creating index '{INDEX_NAME}' with llama-text-embed-v2 @ 2048 dims...")
    pc.create_index_for_model(
        name=INDEX_NAME,
        cloud="aws",
        region="us-east-1",
        embed={
            "model": "llama-text-embed-v2",
            "metric": "cosine",
            "dimension": 2048,
            "field_map": {"text": "chunk_text"},
        },
    )

    while True:
        desc = pc.describe_index(INDEX_NAME)
        if desc.status.ready:
            break
        print("  Waiting for index to be ready...")
        time.sleep(3)
    print("  Index is ready.")


def ingest():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    create_index(pc)
    index = pc.Index(INDEX_NAME)

    pdfs = collect_pdfs(DOCS_DIR)
    print(f"\nFound {len(pdfs)} PDF files to process.\n")

    total_chunks = 0
    for pdf_path in pdfs:
        rel_path = pdf_path.relative_to(DOCS_DIR)
        fund_name = derive_fund_name(pdf_path)
        category = derive_fund_category(pdf_path)
        parent_folder = str(rel_path.parent) if str(rel_path.parent) != "." else "Root"

        print(f"Processing: {rel_path}")
        pages = extract_text_from_pdf(pdf_path)
        if not pages:
            print(f"  Skipped (no extractable text)")
            continue

        records = []
        chunk_idx = 0
        for page_info in pages:
            cleaned = clean_text(page_info["text"])
            chunks = chunk_text(cleaned)
            for chunk in chunks:
                record = {
                    "_id": make_record_id(pdf_path, chunk_idx),
                    "chunk_text": chunk,
                    "fund_name": fund_name,
                    "category": category,
                    "source_file": str(rel_path),
                    "page_number": page_info["page"],
                    "folder": parent_folder,
                }
                records.append(record)
                chunk_idx += 1

        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i : i + BATCH_SIZE]
            index.upsert_records(namespace="funds", records=batch)
            time.sleep(0.5)

        total_chunks += chunk_idx
        print(f"  -> {chunk_idx} chunks from {len(pages)} pages")

    print(f"\nIngestion complete. {total_chunks} total chunks upserted across {len(pdfs)} files.")


if __name__ == "__main__":
    ingest()
