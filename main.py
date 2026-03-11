from dotenv import load_dotenv
load_dotenv("keys.env")

import os
import io
import sqlite3
import requests
import chromadb
from collections import defaultdict
from chromadb.config import Settings
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from docx import Document
import shutil
import hashlib
import logging
from typing import Optional
from datetime import datetime

# ==========================================
# Logging
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rag_api.log")
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# FastAPI App
# ==========================================
app = FastAPI(title="RAG Chatbot API", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "RAG Chatbot API is running", "version": "5.0.0"}

# ==========================================
# Paths
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
persist_path = os.path.join(PROJECT_ROOT, "chroma_db")
docs_path    = os.path.join(PROJECT_ROOT, "docs")
backup_path  = os.path.join(PROJECT_ROOT, "backup_documents")

for p in [docs_path, backup_path]:
    os.makedirs(p, exist_ok=True)

# ==========================================
# API Keys
# ==========================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_KEY   = os.getenv("HF_API_KEY")

# ==========================================
# Globals
# ==========================================
model: Optional[SentenceTransformer] = None
collection = None

# ==========================================
# SQLite
# ==========================================
DB_PATH = os.path.join(PROJECT_ROOT, "chat_history.db")
conn    = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor  = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS history (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    question  TEXT NOT NULL,
    answer    TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

try:
    cursor.execute("ALTER TABLE history ADD COLUMN sources TEXT")
    conn.commit()
except sqlite3.OperationalError:
    pass  # column already exists

def save_history(question: str, answer: str, sources: str = ""):
    cursor.execute(
        "INSERT INTO history (question, answer, sources) VALUES (?, ?, ?)",
        (question, answer, sources)
    )
    conn.commit()

# ==========================================
# Startup
# ==========================================
@app.on_event("startup")
def startup_event():
    global model, collection
    logger.info("Loading embedding model all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.Client(Settings(persist_directory=persist_path, is_persistent=True))
    collection = client.get_or_create_collection("rag_docs")
    if GROQ_API_KEY:
        logger.info("LLM: Groq / llama-3.1-8b-instant")
    elif HF_API_KEY:
        logger.info("LLM: HuggingFace / Mistral-7B-Instruct-v0.2")
    else:
        logger.warning("No API key found! Set GROQ_API_KEY in keys.env")
    logger.info("Server Ready!")

# ==========================================
# Utilities
# ==========================================
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

def extract_text_from_bytes(content_bytes: bytes, filename: str) -> str:
    """
    Extract plain text from uploaded file bytes.
    Handles .txt, .docx, .pdf correctly without writing temp files.
    """
    try:
        fname = filename.lower()

        if fname.endswith(".txt"):
            return content_bytes.decode("utf-8", errors="ignore").strip()

        elif fname.endswith(".docx"):
            # docx is a zip-based format — pass bytes via BytesIO
            doc = Document(io.BytesIO(content_bytes))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

        elif fname.endswith(".pdf"):
            try:
                import fitz  # PyMuPDF
                parts = []
                with fitz.open(stream=content_bytes, filetype="pdf") as pdf:
                    for page in pdf:
                        parts.append(page.get_text())
                return "\n".join(parts).strip()
            except ImportError:
                logger.warning("PyMuPDF not installed. Install with: pip install pymupdf")
                return ""

    except Exception as e:
        logger.error(f"Text extraction failed for {filename}: {e}")
    return ""

def extract_text_from_path(file_path: str, filename: str) -> str:
    """Extract text from a file already saved to disk (used for restore)."""
    try:
        with open(file_path, "rb") as f:
            content_bytes = f.read()
        return extract_text_from_bytes(content_bytes, filename)
    except Exception as e:
        logger.error(f"Could not read file {file_path}: {e}")
        return ""

def generate_hash(content: bytes) -> str:
    return hashlib.md5(content).hexdigest()

def embed_and_store(text: str, filename: str, file_hash: str) -> int:
    if not text.strip():
        return 0
    chunks = chunk_text(text)
    if not chunks:
        return 0
    embeddings = model.encode(chunks).tolist()
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"source": filename, "hash": file_hash} for _ in chunks],
        ids=[f"{filename}_{i}" for i in range(len(chunks))]
    )
    return len(chunks)

def get_file_info(file_path: str, filename: str, location: str) -> dict:
    stat = os.stat(file_path)
    info = {
        "filename":   filename,
        "size_bytes": stat.st_size,
        "size_kb":    round(stat.st_size / 1024, 1),
        "modified":   datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "location":   location,
        "extension":  os.path.splitext(filename)[1].lower()
    }
    # add chunk count from ChromaDB
    try:
        result = collection.get(where={"source": filename})
        info["chunks"] = len(result.get("ids", []))
    except Exception:
        info["chunks"] = 0
    return info

# ==========================================
# LLM — Groq
# ==========================================
def call_groq(context: str, question: str) -> str:
    system_prompt = (
        "You are a document question-answering assistant. "
        "Answer ONLY using the CONTEXT provided. Never use outside knowledge.\n"
        "RULES:\n"
        "1. If the context contains the answer, give a DETAILED response covering all relevant points. "
        "Write at least 300 words. Use numbered lists or paragraphs to organize clearly.\n"
        "2. If the context does NOT contain enough information, say exactly: "
        "'The uploaded documents do not contain information about this topic.'\n"
        "3. Never fabricate facts, names, dates, or statistics not present in the context.\n"
        "4. Do NOT use your training knowledge — only what is in the CONTEXT."
    )
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"CONTEXT:\n\n{context}\n\nQUESTION: {question}\n\nDetailed answer based only on the context above:"}
        ],
        "max_tokens": 2048,
        "temperature": 0.2
    }
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers, json=payload, timeout=60
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

# ==========================================
# LLM — HuggingFace
# ==========================================
def call_huggingface(context: str, question: str) -> str:
    prompt = (
        f"[INST] You are a document QA assistant. Answer using ONLY the context. "
        f"Give a detailed answer of at least 300 words. "
        f"If the context doesn't contain the answer, say so.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nDetailed answer: [/INST]"
    )
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 1500, "temperature": 0.2, "return_full_text": False}
    }
    resp = requests.post(
        "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
        headers=headers, json=payload, timeout=90
    )
    resp.raise_for_status()
    result = resp.json()
    answer = (result[0] if isinstance(result, list) else result).get("generated_text", "").strip()
    if not answer:
        raise ValueError("Empty response from model.")
    return answer

def call_llm(context: str, question: str) -> str:
    if GROQ_API_KEY:
        return call_groq(context, question)
    elif HF_API_KEY:
        return call_huggingface(context, question)
    raise EnvironmentError("No API key configured in keys.env")

# ==========================================
# Request Model
# ==========================================
class QueryModel(BaseModel):
    question: str

# ==========================================
# CHAT Endpoint
# ==========================================
@app.post("/chat")
def chat(query: QueryModel):
    question = query.question.strip()

    if not GROQ_API_KEY and not HF_API_KEY:
        return {"answer": "No API key configured. Add GROQ_API_KEY to keys.env"}

    # Guard: how many chunks are in DB?
    try:
        total_in_db = collection.count()
    except Exception:
        total_in_db = 0

    if total_in_db == 0:
        return {"answer": "No documents have been uploaded yet. Please upload a document using the File Manager first."}

    # n_results must not exceed total docs in collection — else ChromaDB crashes
    n_fetch = min(10, total_in_db)

    # Vector search
    question_embedding = model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_fetch,
        include=["documents", "metadatas", "distances"]
    )

    retrieved_docs = results.get("documents", [[]])[0]
    metadatas_list = results.get("metadatas",  [[]])[0]
    distances      = results.get("distances",   [[]])[0]

    # Log query header
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"  QUESTION : {question}")
    logger.info(f"  DB TOTAL : {total_in_db} chunks  |  FETCHED : {len(retrieved_docs)}")
    logger.info("=" * 70)

    # Similarity threshold — LOW on purpose (dominant-file logic handles noise)
    THRESHOLD = 0.10

    passed_docs = []
    passed_meta = []
    passed_sims = []

    for i, (doc, meta, dist) in enumerate(zip(retrieved_docs, metadatas_list, distances)):
        sim = round(1 - float(dist), 4)
        tag = "PASS" if sim >= THRESHOLD else "SKIP"
        logger.info(f"  [{i+1}] {tag}  sim={sim}  file={meta['source']}")
        logger.info(f"        {doc[:180].strip()}{'...' if len(doc) > 180 else ''}")
        if sim >= THRESHOLD:
            passed_docs.append(doc)
            passed_meta.append(meta)
            passed_sims.append(sim)

    logger.info(f"  PASSED: {len(passed_docs)}/{len(retrieved_docs)}")

    if not passed_docs:
        logger.info("  No relevant chunks found.")
        logger.info("=" * 70)
        return {
            "answer": (
                "I could not find relevant information about this topic in the uploaded documents. "
                "Please ensure you have uploaded a document related to your question."
            )
        }

    # Score per file — pick dominant file if it holds > 55% of score
    file_scores = defaultdict(float)
    file_chunks = defaultdict(list)
    for doc, meta, sim in zip(passed_docs, passed_meta, passed_sims):
        src = meta["source"]
        file_scores[src] += sim
        file_chunks[src].append((sim, doc))

    total_score = sum(file_scores.values())
    top_file    = max(file_scores, key=file_scores.get)
    top_ratio   = file_scores[top_file] / total_score if total_score > 0 else 0

    logger.info("  FILE SCORES:")
    for fname, fscore in sorted(file_scores.items(), key=lambda x: -x[1]):
        pct = round((fscore / total_score) * 100) if total_score > 0 else 0
        logger.info(f"    {pct:3d}%  {fname}")

    if top_ratio >= 0.55 and len(file_scores) > 1:
        # Clear dominant — use only that file's best chunks
        logger.info(f"  DOMINANT FILE ({round(top_ratio*100)}%): {top_file}")
        best       = sorted(file_chunks[top_file], reverse=True)[:5]
        final_docs = [c[1] for c in best]
        sources    = [top_file]
    else:
        # Mixed / single file — take best chunks overall
        all_sorted = sorted(zip(passed_sims, passed_docs, passed_meta), reverse=True)[:5]
        final_docs = [c[1] for c in all_sorted]
        sources    = list({c[2]["source"] for c in all_sorted})
        logger.info(f"  USING: top chunks from all files")

    # Cap each chunk to avoid token overflow
    context = "\n\n---\n\n".join(doc[:3000] for doc in final_docs)

    logger.info(f"  CONTEXT: {len(final_docs)} chunks, {len(context)} chars, sources={sources}")
    logger.info("=" * 70)

    # Call LLM
    try:
        answer = call_llm(context, question)
        logger.info(f"  ANSWER ({len(answer)} chars): {answer[:120]}...")
    except requests.exceptions.Timeout:
        return {"answer": "Request timed out. Please try again."}
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else 0
        msgs = {
            401: "API key invalid or expired. Check your keys.env file.",
            400: "Bad request to API. Restart the server.",
            429: "Rate limit reached. Wait a moment and try again.",
            503: "Model is loading. Wait 20 seconds and try again."
        }
        return {"answer": msgs.get(status, f"API error ({status}). Please try again.")}
    except requests.exceptions.ConnectionError:
        return {"answer": "No internet connection. Check your network."}
    except requests.exceptions.RequestException as e:
        logger.error(f"LLM error: {e}")
        return {"answer": "Could not reach the language model. Check your API key."}
    except (KeyError, IndexError, ValueError) as e:
        logger.error(f"Parse error: {e}")
        return {"answer": "Unexpected response from model. Try again."}
    except EnvironmentError as e:
        return {"answer": str(e)}

    save_history(question, answer, ", ".join(sources))
    return {"answer": answer, "sources": sources}

# ==========================================
# Upload Document
# ==========================================
@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...), replace: bool = Query(False)):
    allowed = {".txt", ".docx", ".pdf"}
    _, ext = os.path.splitext(file.filename)
    if ext.lower() not in allowed:
        return {"error": f"Unsupported file type '{ext}'. Allowed: .txt .docx .pdf", "type": "unsupported"}

    # Read raw bytes — always safe for any file type
    content_bytes = await file.read()
    file_hash     = generate_hash(content_bytes)

    # Check for duplicates
    existing_name    = collection.get(where={"source": file.filename})
    existing_content = collection.get(where={"hash": file_hash})

    if existing_name["ids"] and not replace:
        return {"error": "A file with this name already exists.", "type": "same_name", "action_required": True}
    if existing_content["ids"] and not replace:
        return {"error": "A file with identical content already exists.", "type": "same_content", "action_required": True}

    # Remove old embeddings if replacing
    if replace:
        try:
            collection.delete(where={"source": file.filename})
            collection.delete(where={"hash": file_hash})
        except Exception as e:
            logger.warning(f"Pre-replace delete warning: {e}")

    # Save file to disk
    dest = os.path.join(docs_path, file.filename)
    with open(dest, "wb") as f:
        f.write(content_bytes)

    # Extract text from bytes (correct for all file types)
    text = extract_text_from_bytes(content_bytes, file.filename)

    if not text.strip():
        return {"error": f"Could not extract text from '{file.filename}'. File may be empty or corrupt.", "type": "extraction_failed"}

    count = embed_and_store(text, file.filename, file_hash)
    logger.info(f"Uploaded '{file.filename}' — {count} chunks embedded.")
    return {"message": f"File '{file.filename}' uploaded successfully! ({count} chunks embedded)"}

# ==========================================
# List Documents
# ==========================================
@app.get("/list-documents")
def list_documents():
    active_files, backup_files = [], []

    for filename in sorted(os.listdir(docs_path)):
        if filename.startswith("."):
            continue
        fp = os.path.join(docs_path, filename)
        if os.path.isfile(fp):
            active_files.append(get_file_info(fp, filename, "active"))

    for filename in sorted(os.listdir(backup_path)):
        if filename.startswith("."):
            continue
        fp = os.path.join(backup_path, filename)
        if os.path.isfile(fp):
            backup_files.append(get_file_info(fp, filename, "backup"))

    return {
        "active":  active_files,
        "backups": backup_files,
        "counts":  {"active": len(active_files), "backup": len(backup_files)}
    }

# ==========================================
# Delete (active -> backup)
# ==========================================
@app.delete("/files/{filename}")
def delete_file(filename: str):
    file_path = os.path.join(docs_path, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found in active documents.")

    backup_dest = os.path.join(backup_path, filename)
    if os.path.exists(backup_dest):
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext   = os.path.splitext(filename)
        backup_dest = os.path.join(backup_path, f"{name}_{timestamp}{ext}")

    shutil.move(file_path, backup_dest)

    deleted_chunks = 0
    try:
        existing = collection.get(where={"source": filename})
        deleted_chunks = len(existing.get("ids", []))
        if deleted_chunks > 0:
            collection.delete(where={"source": filename})
    except Exception as e:
        logger.warning(f"ChromaDB delete warning: {e}")

    logger.info(f"Deleted '{filename}' -> backup. Removed {deleted_chunks} chunks.")
    return {
        "message":        f"'{filename}' moved to backup.",
        "backup_name":    os.path.basename(backup_dest),
        "chunks_removed": deleted_chunks
    }

# ==========================================
# Restore (backup -> active)
# ==========================================
@app.post("/files/{filename}/restore")
def restore_file(filename: str):
    backup_file = os.path.join(backup_path, filename)
    if not os.path.exists(backup_file):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found in backup.")

    restore_dest = os.path.join(docs_path, filename)
    if os.path.exists(restore_dest):
        raise HTTPException(status_code=409, detail=f"'{filename}' already exists in active documents. Delete it first.")

    shutil.move(backup_file, restore_dest)

    text = extract_text_from_path(restore_dest, filename)
    if not text.strip():
        shutil.move(restore_dest, backup_file)
        raise HTTPException(status_code=422, detail=f"Could not extract text from '{filename}'. Moved back to backup.")

    with open(restore_dest, "rb") as f:
        content_bytes = f.read()
    file_hash = generate_hash(content_bytes)

    try:
        collection.delete(where={"source": filename})
    except Exception:
        pass

    count = embed_and_store(text, filename, file_hash)
    logger.info(f"Restored '{filename}' — {count} chunks embedded.")
    return {"message": f"'{filename}' restored and re-embedded.", "chunks_embedded": count}

# ==========================================
# Permanent Delete
# ==========================================
@app.delete("/files/{filename}/permanent")
def permanent_delete(filename: str):
    backup_file = os.path.join(backup_path, filename)
    if not os.path.exists(backup_file):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found in backup.")
    os.remove(backup_file)
    logger.info(f"Permanently deleted '{filename}'.")
    return {"message": f"'{filename}' permanently deleted."}

# ==========================================
# File Info
# ==========================================
@app.get("/files/{filename}/info")
def file_info(filename: str):
    active_path = os.path.join(docs_path, filename)
    backup_file = os.path.join(backup_path, filename)
    if os.path.exists(active_path):
        return get_file_info(active_path, filename, "active")
    elif os.path.exists(backup_file):
        return get_file_info(backup_file, filename, "backup")
    raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")

# ==========================================
# History
# ==========================================
@app.get("/history")
def get_history(limit: int = Query(50, ge=1, le=500)):
    rows = cursor.execute(
        "SELECT id, question, answer, sources, timestamp FROM history ORDER BY id DESC LIMIT ?",
        (limit,)
    ).fetchall()
    return [{"id": r[0], "question": r[1], "answer": r[2], "sources": r[3], "timestamp": r[4]} for r in rows]

@app.delete("/clear-history")
def clear_history():
    cursor.execute("DELETE FROM history")
    conn.commit()
    return {"message": "Chat history cleared."}

@app.delete("/delete-history/{history_id}")
def delete_history_item(history_id: int):
    cursor.execute("DELETE FROM history WHERE id = ?", (history_id,))
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="History item not found.")
    conn.commit()
    return {"message": f"History item {history_id} deleted."}
