import os
import sys
import faiss
import pickle
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer

# -------- Text extractors --------
import PyPDF2
from docx import Document
from PIL import Image
import pytesseract


# ===============================
# CONFIG
# ===============================
VECTOR_INDEX_FILE = "vectors.index"
CHUNKS_FILE = "chunks.pkl"
DOCUMENTS_FILE = "documents.pkl"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# If an output directory is provided via CLI, we'll override the paths to
# write per-project files into that directory. The CLI entry below will set
# `OUT_DIR` when invoked as `python pdf_to_vectors.py <file> <out_dir>`.
OUT_DIR = None


# ===============================
# LOAD EMBEDDING MODEL
# ===============================
print("🔄 Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
EMBEDDING_DIM = embedder.get_sentence_embedding_dimension()


# ===============================
# TEXT EXTRACTION FUNCTIONS
# ===============================
def extract_text_from_pdf(path):
    reader = PyPDF2.PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        pages.append({"page": i + 1, "text": page.extract_text() or ""})
    return pages


def extract_text_from_docx(path):
    doc = Document(path)
    text = "\n".join(p.text for p in doc.paragraphs)
    return [{"page": None, "text": text}]


def extract_text_from_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [{"page": None, "text": f.read()}]


def extract_text_from_image(path):
    image = Image.open(path)
    text = pytesseract.image_to_string(image)
    return [{"page": None, "text": text}]


# ===============================
# MAIN INGESTION FUNCTION
# ===============================
def ingest_document(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    ext = os.path.splitext(file_path)[1].lower()
    doc_id = os.path.basename(file_path)

    print(f"\n📄 Ingesting: {doc_id}")

    # -------- Extract text --------
    if ext == ".pdf":
        pages = extract_text_from_pdf(file_path)
    elif ext == ".docx":
        pages = extract_text_from_docx(file_path)
    elif ext == ".txt":
        pages = extract_text_from_txt(file_path)
    elif ext in [".png", ".jpg", ".jpeg"]:
        pages = extract_text_from_image(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # -------- Chunking --------
    chunks = []
    metadata = []

    for p in pages:
        text = p["text"]
        page_no = p["page"]

        for start in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk = text[start:start + CHUNK_SIZE].strip()
            if chunk:
                chunks.append(chunk)
                metadata.append({
                    "doc_id": doc_id,
                    "page": page_no
                })

    print(f"✂️ Created {len(chunks)} chunks")

    # -------- Embeddings --------
    print("🧠 Creating embeddings...")
    embeddings = embedder.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # ===============================
    # LOAD OR CREATE FAISS INDEX
    # ===============================
    # allow OUT_DIR override
    vector_file = VECTOR_INDEX_FILE
    chunks_file = CHUNKS_FILE
    documents_file = DOCUMENTS_FILE
    if OUT_DIR:
        os.makedirs(OUT_DIR, exist_ok=True)
        vector_file = os.path.join(OUT_DIR, "vectors.index")
        chunks_file = os.path.join(OUT_DIR, "chunks.pkl")
        documents_file = os.path.join(OUT_DIR, "documents.pkl")

    if os.path.exists(vector_file):
        index = faiss.read_index(vector_file)
        if index.d != EMBEDDING_DIM:
            raise ValueError("Embedding dimension mismatch")
    else:
        index = faiss.IndexFlatIP(EMBEDDING_DIM)

    index.add(embeddings.astype("float32"))
    faiss.write_index(index, vector_file)

    # ===============================
    # LOAD OR APPEND CHUNKS
    # ===============================
    all_chunks = []
    all_metadata = []
    total_pages = 0

    if os.path.exists(chunks_file):
        with open(chunks_file, "rb") as f:
            old = pickle.load(f)
            all_chunks = old["chunks"]
            all_metadata = old["metadata"]
            total_pages = old.get("total_pages", 0)

    all_chunks.extend(chunks)
    all_metadata.extend(metadata)
    total_pages += len(pages)

    with open(chunks_file, "wb") as f:
        pickle.dump({
            "chunks": all_chunks,
            "metadata": all_metadata,
            "total_pages": total_pages,
            "embedding_dim": EMBEDDING_DIM
        }, f)

    # ===============================
    # UPDATE DOCUMENT REGISTRY
    # ===============================
    documents = []
    if os.path.exists(documents_file):
        with open(documents_file, "rb") as f:
            documents = pickle.load(f)

    documents = [d for d in documents if d["id"] != doc_id]
    documents.append({
        "id": doc_id,
        "path": os.path.abspath(file_path),
        "chunks": len(chunks),
        "indexed_at": datetime.utcnow().isoformat()
    })

    with open(documents_file, "wb") as f:
        pickle.dump(documents, f)

    # Also write a human-readable meta.json next to the pickle for convenience
    try:
        meta_json = documents_file.rsplit('.', 1)[0] + '.json'
        import json
        with open(meta_json, 'w', encoding='utf-8') as jf:
            json.dump(documents, jf, indent=2)
    except Exception:
        pass

    print("✅ Ingestion complete")


# ===============================
# CLI ENTRY
# ===============================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_to_vectors.py <file> [out_dir]")
        sys.exit(1)

    # If a second argument is provided, treat it as the output directory
    # where we write project-specific vectors/chunks/documents files.
    if len(sys.argv) >= 3:
        OUT_DIR = sys.argv[2]

    for file in sys.argv[1:2]:
        ingest_document(file)
