import faiss
import pickle
import numpy as np
import requests
import os
import re
import subprocess
import sys
from sentence_transformers import SentenceTransformer


# ===============================
# CONFIG
# ===============================
VECTOR_INDEX_FILE = "vectors.index"
CHUNKS_FILE = "chunks.pkl"
DOCUMENTS_FILE = "documents.pkl"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"

TOP_K = 3

# Per-project data directory
RAG_DATA_DIR = "rag_data"
# active project id (subfolder under RAG_DATA_DIR)
ACTIVE_PROJECT = "default"


def project_paths(project_id=None):
    if project_id is None:
        project_id = ACTIVE_PROJECT
    base = os.path.join(RAG_DATA_DIR, project_id)
    return {
        "base": base,
        "vectors": os.path.join(base, "vectors.index"),
        "chunks": os.path.join(base, "chunks.pkl"),
        "documents_pkl": os.path.join(base, "documents.pkl"),
        "meta_json": os.path.join(base, "meta.json"),
    }


# ===============================
# LOAD EMBEDDING MODEL
# ===============================
print("🔄 Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)


# ===============================
# LLM CALL
# ===============================
def generate_answer(context, question, conversation_history=None):
    history = ""
    if conversation_history:
        history = "\n".join(f"{r}: {t}" for r, t in conversation_history)

    if context:
        prompt = f"""
You are an AI assistant answering questions using ONLY the provided context
from multiple uploaded documents. Mention document names and page numbers
when relevant.

Context:
{context}

Conversation:
{history}

Question:
{question}

Answer:
"""
    else:
        # When no document context is provided, allow the model to answer
        # from its internal knowledge. Conversation history may be included
        # to preserve conversational flow/pronoun resolution only — it must
        # not be used to determine whether factual information exists.
        prompt = f"""
You are an AI assistant. Use your internal knowledge to answer the question.
If conversation text is provided, use it only to preserve flow and pronoun
references; do not treat earlier chat messages as a source of factual data
or as evidence that a document contains the requested information.

Conversation:
{history}

Question:
{question}

Answer:
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )
    except Exception as e:
        print(f"❌ LLM request failed: {e}")
        return ""

    # Try to parse the response robustly — different local LLM endpoints
    # can return variants of keys. Try common structures and fall back to
    # raw text for debugging.
    try:
        j = response.json()
    except Exception:
        # not JSON — return raw text
        text = response.text or ""
        if not text:
            print("❌ Empty response from LLM (non-JSON)")
        return text

    # Common simple key
    if isinstance(j, dict):
        for key in ("response", "result", "text", "output", "answer"):
            if key in j and isinstance(j[key], str) and j[key].strip():
                return j[key].strip()

        # Check for choices/message style
        if "choices" in j and isinstance(j["choices"], list):
            parts = []
            for c in j["choices"]:
                if isinstance(c, dict):
                    if "message" in c and isinstance(c["message"], dict):
                        msg = c["message"].get("content") or c["message"].get("text")
                        if isinstance(msg, str):
                            parts.append(msg)
                    if "content" in c and isinstance(c["content"], list):
                        for item in c["content"]:
                            if isinstance(item, dict) and "text" in item:
                                parts.append(item.get("text", ""))
            if parts:
                return "".join(parts).strip()

    # nothing found — print debug and return empty string
    print("❗ Unexpected LLM response structure:")
    try:
        import json as _json
        print(_json.dumps(j, indent=2))
    except Exception:
        print(str(j))

    return ""


# ===============================
# SYSTEM-LEVEL QUESTIONS
# ===============================
def handle_system_query(question):
    q = question.lower()
    paths = project_paths()
    # prefer JSON meta if present, else fallback to pickle documents
    docs = None
    if os.path.exists(paths["meta_json"]):
        try:
            import json
            with open(paths["meta_json"], "r", encoding="utf-8") as f:
                docs = json.load(f)
        except Exception:
            docs = None
    if docs is None and os.path.exists(paths["documents_pkl"]):
        try:
            with open(paths["documents_pkl"], "rb") as f:
                docs = pickle.load(f)
        except Exception:
            docs = None

    if not docs:
        return None

    if "how many" in q and "document" in q:
        return f"You have uploaded {len(docs)} document(s)."

    if any(k in q for k in [
        "list documents", "uploaded documents", "which documents",
        "show documents", "what documents"
    ]):
        lines = []
        for d in docs:
            lines.append(
                f"- {d['id']} | chunks: {d['chunks']} | indexed at {d['indexed_at']}"
            )
        return "Uploaded documents:\n" + "\n".join(lines)

    return None


# ===============================
# SMALL-TALK DETECTION
# ===============================
def is_small_talk(question: str) -> bool:
    """Return True if the question/input looks like small-talk (greetings,
    acknowledgements, casual messages) that should NOT trigger document
    retrieval.

    Uses simple heuristics and a short phrase set to avoid false positives.
    """
    if not question:
        return False

    q = question.lower().strip()
    # remove punctuation for matching
    q_clean = re.sub(r"[^\w\s]", "", q)
    tokens = q_clean.split()

    # common small-talk tokens/phrases
    small_tokens = {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank",
        "thankyou",
        "thx",
        "ty",
        "ok",
        "okay",
        "good",
        "great",
        "nice",
        "cool",
        "bye",
        "goodbye",
        "see",
        "ya",
    }

    # direct phrase checks
    if q_clean in ("thank you", "thanks a lot", "see you", "see ya"):
        return True

    # single-token greetings/ackspyr
    if len(tokens) == 1 and tokens[0] in small_tokens:
        return True

    # very short inputs that contain only small-talk words
    if len(tokens) <= 3 and all(t in small_tokens or t in ("you",) for t in tokens):
        return True

    return False


def is_general_knowledge(question: str) -> bool:
    """Heuristic check for general-knowledge / concept / conversational queries
    that can be answered from the LLM's internal knowledge rather than by
    retrieving uploaded documents. Returns True when the question appears to
    ask about facts, concepts, explanations, or casual conversation and does
    NOT explicitly reference documents or uploaded files.

    NOTE: This function no longer forces bypass of retrieval when a vector
    index exists; the main handler will prefer the index for informational
    queries when available. This function is only used when no index exists.
    """
    if not question:
        return False

    q = question.lower().strip()
    # phrases that usually indicate general knowledge / concept explanation
    general_phrases = [
        "what is",
        "who is",
        "when is",
        "where is",
        "why",
        "how",
        "explain",
        "define",
        "describe",
        "tell me about",
        "give me an example",
        "difference between",
        "show me",
        "joke",
        "quote",
        "translate",
    ]

    # tokens that indicate the user is explicitly asking about the uploaded documents
    doc_tokens = [
        "document",
        "documents",
        "file",
        "files",
        "uploaded",
        "index",
        "page",
        "pdf",
        "docx",
        "in the document",
        "in my files",
        "in these documents",
    ]

    # If question contains any doc-specific token, prefer retrieval
    for t in doc_tokens:
        if t in q:
            return False

    # If it starts with or contains a general-knowledge phrase, treat as general
    for p in general_phrases:
        if p in q:
            return True

    # Short casual inputs that are not document-references are general chat
    if len(q.split()) <= 5 and not any(t in q for t in doc_tokens):
        # avoid misclassifying numeric ids or short codes
        if re.search(r"\d{4,}", q):
            return False
        return True

    return False


def is_document_query(question: str) -> bool:
    """Heuristic to detect questions that are explicitly requesting
    information expected to be found in uploaded documents (IDs, register
    numbers, patient names, page-specific facts, document references, etc.).
    These should always trigger retrieval first.
    """
    if not question:
        return False

    q = question.lower()


    # Strong indicators: explicit document/file mention
    doc_indicators = [
        "document",
        "documents",
        "file",
        "files",
        "uploaded",
        "pdf",
        "docx",
        "indexed",
        "index",
        "in the document",
        "in the file",
        "find in",
    ]
    for token in doc_indicators:
        if token in q:
            return True

    # Objects/fields that are commonly document-dependent
    doc_fields = [
        "allerg",  # allergy/allergies
        "allergy",
        "allergies",
        "name",
        "names",
        "id",
        "ids",
        "register",
        "registration",
        "register number",
        "register no",
        "patient",
        "patients",
        "record",
        "records",
        "attribute",
        "attributes",
        "mrn",
        "ssn",
        "passport",
        "invoice",
        "order number",
        "serial number",
        "dob",
        "date of birth",
        "address",
        "phone",
        "phone number",
        "email",
    ]

    # Action verbs which, when combined with doc_fields, should force retrieval
    action_verbs = [
        "list",
        "retrieve",
        "identify",
        "extract",
        "find",
        "lookup",
        "look up",
        "show",
        "get",
        "what is",
        "what's",
        "who is",
        "which",
        "give me",
        "provide",
    ]

    # If the question contains any of the doc_fields together with action verbs,
    # treat it as document-dependent. This enforces the rule that requests to
    # list/retrieve/identify/extract fields must trigger retrieval.
    for field in doc_fields:
        if field in q:
            for verb in action_verbs:
                if verb in q:
                    return True

    # Also match common factual request patterns explicitly referencing fields
    if re.search(r"\b(list|retrieve|identify|extract|find|show|get|lookup)\b.*\b(name|names|id|ids|allerg|allergy|allergies|record|records|patient|patients|attribute|attributes|mrn|ssn|dob)\b", q):
        return True

    # Regex: explicit field requests like "what is the <field> for <entity>"
    if re.search(r"\bwhat (?:is|'?s) the\b.*\b(id|number|name|mrn|ssn|invoice|order|passport|dob|allergy)\b", q):
        return True

    # Fallback: if the question clearly mentions document pages or indexes
    if re.search(r"\b(page|page number|section|paragraph)\b", q):
        return True

    return False


# ===============================
# RAG QUESTION HANDLER
# ===============================
def ask_question(question, conversation_history=None):
    # system-level shortcut
    sys_answer = handle_system_query(question)
    if sys_answer:
        return sys_answer

    # Evaluate document-level intent first (document-dependent queries must
    # always trigger retrieval when an index exists).
    doc_query = is_document_query(question)

    # Small-talk bypass: do not trigger retrieval for greetings/acks
    if is_small_talk(question) and not doc_query:
        return generate_answer(context=None, question=question, conversation_history=conversation_history)

    paths = project_paths()
    index_exists = os.path.exists(paths["vectors"]) and os.path.exists(paths["chunks"])

    # If this is explicitly a document-dependent query, force retrieval.
    if doc_query:
        if not index_exists:
            return "The requested information may be document-dependent, but no uploaded documents are indexed for this project."

        # load index + data
        index = faiss.read_index(paths["vectors"])
        with open(paths["chunks"], "rb") as f:
            data = pickle.load(f)

        chunks = data.get("chunks", [])
        metadata = data.get("metadata", [])

        if len(chunks) == 0:
            return "The requested information is not present in the uploaded documents."

        # embed query
        query_vector = embedder.encode(question, normalize_embeddings=True).reshape(1, -1)

        # search
        k = min(TOP_K, len(chunks))
        scores, indices = index.search(query_vector.astype("float32"), k)

        # If nothing retrieved, explicitly report not present
        if len(indices[0]) == 0 or all(i < 0 for i in indices[0]):
            return "The requested information is not present in the uploaded documents."

        try:
            top_score = max(scores[0]) if len(scores) and len(scores[0]) else 0.0
        except Exception:
            top_score = 0.0

        LOW_CONFIDENCE_THRESHOLD = 0.05
        if top_score < LOW_CONFIDENCE_THRESHOLD:
            return "The requested information is not present in the uploaded documents."

        # build context
        context_parts = []
        print("\n🔍 Retrieved Chunks:")
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(chunks):
                continue
            meta = metadata[idx]
            doc = meta.get("doc_id", "unknown")
            page = meta.get("page")

            label = f"[{doc}"
            if page is not None:
                label += f" | Page {page}"
            label += "]"

            print(f"  {rank}. Score={score:.3f} {label}")
            context_parts.append(f"{label} {chunks[idx]}")

        context = "\n\n".join(context_parts)
        # Use conversation history only to maintain flow/pronouns, not to decide
        # whether the answer exists.
        return generate_answer(context, question, conversation_history)

    # If not explicitly document-dependent:
    # - If an index exists, prefer retrieval-first for factual/informational queries.
    # - Small-talk already bypassed above.
    if index_exists:
        # load index + data
        index = faiss.read_index(paths["vectors"])
        with open(paths["chunks"], "rb") as f:
            data = pickle.load(f)

        chunks = data.get("chunks", [])
        metadata = data.get("metadata", [])

        if len(chunks) == 0:
            return "The requested information is not present in the uploaded documents."

        # embed query
        query_vector = embedder.encode(question, normalize_embeddings=True).reshape(1, -1)

        # search
        k = min(TOP_K, len(chunks))
        scores, indices = index.search(query_vector.astype("float32"), k)

        if len(indices[0]) == 0 or all(i < 0 for i in indices[0]):
            return "The requested information is not present in the uploaded documents."

        try:
            top_score = max(scores[0]) if len(scores) and len(scores[0]) else 0.0
        except Exception:
            top_score = 0.0

        LOW_CONFIDENCE_THRESHOLD = 0.05
        if top_score < LOW_CONFIDENCE_THRESHOLD:
            return "The requested information is not present in the uploaded documents."

        # build context
        context_parts = []
        print("\n🔍 Retrieved Chunks:")
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(chunks):
                continue
            meta = metadata[idx]
            doc = meta.get("doc_id", "unknown")
            page = meta.get("page")

            label = f"[{doc}"
            if page is not None:
                label += f" | Page {page}"
            label += "]"

            print(f"  {rank}. Score={score:.3f} {label}")
            context_parts.append(f"{label} {chunks[idx]}")

        context = "\n\n".join(context_parts)
        return generate_answer(context, question, conversation_history)

    # No index exists: allow general-knowledge questions to be answered by LLM,
    # but inform user when a query could be document-dependent.
    if is_general_knowledge(question):
        return generate_answer(context=None, question=question, conversation_history=conversation_history)

    return "The requested information may be document-dependent, but no uploaded documents are indexed for this project."


# ===============================
# DOCUMENT INGESTOR CALLER
# ===============================
def run_document_ingestor(file_path):
    import urllib.parse

    p = file_path.strip().strip('"').strip("'")
    if p.startswith("file://"):
        parsed = urllib.parse.urlparse(p)
        p = urllib.parse.unquote(parsed.path)
        if os.name == "nt" and p.startswith("/"):
            p = p.lstrip("/")

    if not os.path.exists(p):
        print(f"❌ File not found: {p}")
        return

    try:
        # ensure we call the correct ingestor script (pdf_to_vectors.py) using an absolute path
        script_path = os.path.join(os.path.dirname(__file__), "pdf_to_vectors.py")
        if not os.path.exists(script_path):
            raise FileNotFoundError(script_path)

        print(f"🔁 Indexing document: {p}")
        # pass the target project directory as the second argument so the
        # ingestor writes per-project files under rag_data/<project>/
        proj = project_paths()["base"]
        os.makedirs(proj, exist_ok=True)
        subprocess.run(
            [sys.executable, script_path, p, proj],
            check=True
        )
        print("✅ Indexing completed.")
    except subprocess.CalledProcessError:
        print("❌ Failed to index document.")
    except FileNotFoundError:
        print("❌ pdf_to_vectors.py not found.")


# ===============================
# MAIN LOOP
# ===============================
def main():
    global ACTIVE_PROJECT

    conversation_history = []

    def print_status():
        print("\n" + "=" * 60)
        proj_paths = project_paths()
        print(f"🔖 Active project: {ACTIVE_PROJECT} (folder: {proj_paths['base']})")
        if os.path.exists(proj_paths["chunks"]):
            with open(proj_paths["chunks"], "rb") as f:
                data = pickle.load(f)
            print("🤖 MULTI-DOCUMENT RAG SYSTEM READY")
            print(f"🧩 Total chunks: {len(data['chunks'])}")
            print(f"🧠 Embedding dim: {data['embedding_dim']}")
        else:
            print("🤖 CHAT MODE (no documents indexed)")
        print("Commands: upload | clear | exit | projects | use <project>")
        print("=" * 60)

    print_status()

    while True:
        user_input = input("\n❓ Your input: ").strip()

        if user_input.lower() in ["exit", "quit", "bye", "q"]:
            print("👋 Goodbye!")
            break

        if user_input.lower() == "clear":
            conversation_history.clear()
            print("🧹 Conversation cleared.")
            continue

        if user_input.lower() == "upload":
            path = input("Enter file path (PDF/DOCX/TXT/Image): ").strip()
            if path:
                # choose project
                proj = input(f"Project id (folder under {RAG_DATA_DIR}) [default: {ACTIVE_PROJECT}]: ").strip()
                if proj:
                    # switch active project for this operation
                    ACTIVE_PROJECT = proj

                # Ask whether to isolate this project's data (delete existing files)
                iso = input("Isolate this upload (delete existing indexed documents in project)? [y/N]: ").strip().lower()
                proj_paths = project_paths()
                if iso in ("y", "yes"):
                    try:
                        if os.path.exists(proj_paths["base"]):
                            import shutil
                            shutil.rmtree(proj_paths["base"])
                            print(f"🗑️ Removed project folder {proj_paths['base']} to isolate new upload.")
                    except Exception as e:
                        print(f"⚠️ Failed to remove project folder: {e}")

                run_document_ingestor(path)
                print_status()
            continue

        if user_input.lower() == "projects":
            # list project folders
            if not os.path.exists(RAG_DATA_DIR):
                print("No projects found.")
            else:
                for name in sorted(os.listdir(RAG_DATA_DIR)):
                    path = os.path.join(RAG_DATA_DIR, name)
                    if os.path.isdir(path):
                        print(f"- {name}")
            continue

        if user_input.lower().startswith("use "):
            # switch active project: `use project_001`
            try:
                newp = user_input.split(None, 1)[1].strip()
            except Exception:
                newp = ""
            if newp:
                ACTIVE_PROJECT = newp
                print(f"Active project set to: {ACTIVE_PROJECT}")
            continue

        if not user_input:
            print("⚠️ Please enter a question.")
            continue

        conversation_history.append(("User", user_input))

        answer = ask_question(user_input, conversation_history)

        if not answer:
            answer = generate_answer(
                context=None,
                question=user_input,
                conversation_history=conversation_history
            )

        conversation_history.append(("Assistant", answer))

        print("\n🤖 Answer:")
        print(answer)


if __name__ == "__main__":
    main()
