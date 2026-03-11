"""
ingest.py — Standalone document ingestion script
Run this ONCE to pre-load documents into ChromaDB before starting the server.
Place documents in the /docs folder (same folder as backend).
Supports: .txt, .docx, .pdf (requires: pip install pymupdf)
"""

import os
import io
import hashlib
import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from docx import Document

# ==========================================
# Logging
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ==========================================
# Paths — looks for /docs in same folder as this script
# ==========================================
SCRIPT_DIR   = Path(__file__).resolve().parent
docs_path    = SCRIPT_DIR / "docs"
persist_path = SCRIPT_DIR / "chroma_db"

logger.info(f"Documents folder : {docs_path}")
logger.info(f"ChromaDB folder  : {persist_path}")

if not docs_path.exists():
    docs_path.mkdir(parents=True)
    raise FileNotFoundError(
        f"Created docs folder at: {docs_path}\n"
        f"Please put your documents (.txt, .docx, .pdf) there and run again."
    )

# ==========================================
# ChromaDB — wipe and rebuild
# ==========================================
client = chromadb.Client(
    Settings(persist_directory=str(persist_path), is_persistent=True)
)

try:
    client.delete_collection("rag_docs")
    logger.info("Old collection deleted — starting fresh.")
except Exception:
    pass

collection = client.get_or_create_collection("rag_docs")

# ==========================================
# Embedding Model
# ==========================================
logger.info("Loading embedding model all-MiniLM-L6-v2...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ==========================================
# Utility Functions
# ==========================================
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

def extract_text(file_path: Path) -> str:
    """
    Extract text from file.
    Reads raw bytes first so all file types are handled correctly.
    """
    suffix = file_path.suffix.lower()
    content_bytes = file_path.read_bytes()

    if suffix == ".txt":
        return content_bytes.decode("utf-8", errors="ignore").strip()

    elif suffix == ".docx":
        # docx is zip-based — use BytesIO so we don't need a real file path
        doc = Document(io.BytesIO(content_bytes))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    elif suffix == ".pdf":
        try:
            import fitz  # PyMuPDF — pip install pymupdf
            text_parts = []
            with fitz.open(stream=content_bytes, filetype="pdf") as pdf:
                for page in pdf:
                    text_parts.append(page.get_text())
            return "\n".join(text_parts).strip()
        except ImportError:
            logger.warning("PyMuPDF not installed. Run: pip install pymupdf")
            return ""

    logger.warning(f"Unsupported file type: {file_path.name}")
    return ""

def generate_hash(content_bytes: bytes) -> str:
    return hashlib.md5(content_bytes).hexdigest()

# ==========================================
# Process Documents
# ==========================================
SUPPORTED = {".txt", ".docx", ".pdf"}

all_documents: list = []
all_metadatas: list = []
all_ids:       list = []

files = [f for f in sorted(docs_path.iterdir()) if f.suffix.lower() in SUPPORTED and f.is_file()]

if not files:
    raise ValueError(
        f"No supported documents (.txt, .docx, .pdf) found in:\n  {docs_path}\n"
        f"Please add files and run again."
    )

logger.info(f"Found {len(files)} document(s) to process.")

for file_path in files:
    logger.info(f"\n--- Processing: {file_path.name} ---")

    content_bytes = file_path.read_bytes()
    text = extract_text(file_path)

    if not text.strip():
        logger.warning(f"  SKIPPED: Empty or unreadable — {file_path.name}")
        continue

    file_hash = generate_hash(content_bytes)
    chunks    = chunk_text(text)

    logger.info(f"  Size      : {len(content_bytes)} bytes")
    logger.info(f"  Characters: {len(text)}")
    logger.info(f"  Chunks    : {len(chunks)}")
    logger.info(f"  Hash      : {file_hash[:8]}...")
    logger.info(f"  Preview   : {text[:150].strip()!r}")

    for i, chunk in enumerate(chunks):
        all_documents.append(chunk)
        all_metadatas.append({"source": file_path.name, "hash": file_hash})
        all_ids.append(f"{file_path.name}_{i}")

# ==========================================
# Embed & Store
# ==========================================
if not all_documents:
    raise ValueError("No valid text extracted from any document!")

logger.info(f"\nGenerating embeddings for {len(all_documents)} chunks...")
embeddings = model.encode(all_documents, show_progress_bar=True).tolist()

logger.info("Storing in ChromaDB...")
BATCH_SIZE = 256
for i in range(0, len(all_documents), BATCH_SIZE):
    collection.add(
        documents=all_documents[i:i + BATCH_SIZE],
        embeddings=embeddings[i:i + BATCH_SIZE],
        metadatas=all_metadatas[i:i + BATCH_SIZE],
        ids=all_ids[i:i + BATCH_SIZE],
    )
    logger.info(f"  Stored batch {i // BATCH_SIZE + 1}/{(len(all_documents) - 1) // BATCH_SIZE + 1}")

logger.info(f"\n=== DONE ===")
logger.info(f"  {len(all_documents)} chunks from {len(files)} file(s) stored in ChromaDB.")
logger.info(f"  You can now start the server: uvicorn main:app --reload")
