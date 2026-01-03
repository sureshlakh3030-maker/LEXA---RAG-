import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Get embedding dimension from the model
_embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
EMBEDDING_DIM = _embedder.get_sentence_embedding_dimension()

def rebuild_index_for_project(project_dir: str, doc_id_to_remove: str = None):
    """
    Rebuild the FAISS index for a project, optionally removing a specific document.
    
    Args:
        project_dir: Directory containing the project's index files
        doc_id_to_remove: Document ID to remove (if None, rebuilds all)
    """
    chunks_file = os.path.join(project_dir, "chunks.pkl")
    vectors_file = os.path.join(project_dir, "vectors.index")
    documents_file = os.path.join(project_dir, "documents.pkl")
    
    if not os.path.exists(chunks_file):
        return False
    
    # Load chunks and metadata
    with open(chunks_file, "rb") as f:
        data = pickle.load(f)
    
    all_chunks = data.get("chunks", [])
    all_metadata = data.get("metadata", [])
    
    # Filter out chunks from the document to remove
    if doc_id_to_remove:
        filtered_chunks = []
        filtered_metadata = []
        
        for chunk, meta in zip(all_chunks, all_metadata):
            if meta.get("doc_id") != doc_id_to_remove:
                filtered_chunks.append(chunk)
                filtered_metadata.append(meta)
        
        all_chunks = filtered_chunks
        all_metadata = filtered_metadata
    
    if len(all_chunks) == 0:
        # No chunks left, remove index files
        if os.path.exists(vectors_file):
            os.remove(vectors_file)
        if os.path.exists(chunks_file):
            os.remove(chunks_file)
        if os.path.exists(documents_file):
            # Update documents list to remove the deleted document
            with open(documents_file, "rb") as f:
                documents = pickle.load(f)
            documents = [d for d in documents if d["id"] != doc_id_to_remove]
            if documents:
                with open(documents_file, "wb") as f:
                    pickle.dump(documents, f)
                # Update JSON too
                try:
                    meta_json = documents_file.rsplit('.', 1)[0] + '.json'
                    import json
                    with open(meta_json, 'w', encoding='utf-8') as jf:
                        json.dump(documents, jf, indent=2)
                except Exception:
                    pass
            else:
                os.remove(documents_file)
                try:
                    meta_json = documents_file.rsplit('.', 1)[0] + '.json'
                    if os.path.exists(meta_json):
                        os.remove(meta_json)
                except Exception:
                    pass
        return True
    
    # Re-embed all remaining chunks
    print(f"🔄 Re-embedding {len(all_chunks)} chunks...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = embedder.encode(
        all_chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    
    # Create new index
    embedding_dim = embedder.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings.astype("float32"))
    
    # Save new index
    faiss.write_index(index, vectors_file)
    
    # Update chunks file
    with open(chunks_file, "wb") as f:
        pickle.dump({
            "chunks": all_chunks,
            "metadata": filtered_metadata if doc_id_to_remove else all_metadata,
            "total_pages": data.get("total_pages", 0),
            "embedding_dim": embedding_dim
        }, f)
    
    # Update documents list
    if doc_id_to_remove and os.path.exists(documents_file):
        with open(documents_file, "rb") as f:
            documents = pickle.load(f)
        documents = [d for d in documents if d["id"] != doc_id_to_remove]
        with open(documents_file, "wb") as f:
            pickle.dump(documents, f)
        # Update JSON too
        try:
            meta_json = documents_file.rsplit('.', 1)[0] + '.json'
            import json
            with open(meta_json, 'w', encoding='utf-8') as jf:
                json.dump(documents, jf, indent=2)
        except Exception:
            pass
    
    print("✅ Re-indexing complete")
    return True

def delete_document_from_project(project_dir: str, doc_id: str) -> bool:
    """
    Delete a document from a project by removing its chunks and rebuilding the index.
    
    Args:
        project_dir: Directory containing the project's index files
        doc_id: Document ID to delete
    
    Returns:
        True if successful, False otherwise
    """
    try:
        return rebuild_index_for_project(project_dir, doc_id)
    except Exception as e:
        print(f"❌ Error deleting document: {e}")
        return False

