from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import sys
import subprocess
import shutil

# Import functions from existing Python files
sys.path.insert(0, os.path.dirname(__file__))
from ask_questions import (
    ask_question,
    project_paths,
    run_document_ingestor,
    ACTIVE_PROJECT,
    RAG_DATA_DIR
)
import pdf_to_vectors
import database
import reindex_utils

app = FastAPI(title="RAG API Server")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QuestionRequest(BaseModel):
    question: str
    conversation_history: Optional[List[dict]] = []
    project_id: Optional[str] = None
    conversation_id: Optional[int] = None

class MessageRequest(BaseModel):
    conversation_id: int
    role: str
    content: str

class ProjectSwitchRequest(BaseModel):
    project_id: str

class ProjectCreateRequest(BaseModel):
    project_id: str

# Global active project
_current_project = "default"

# Monkey patch to update ACTIVE_PROJECT in ask_questions module
def set_active_project(project_id):
    global _current_project
    _current_project = project_id
    import ask_questions
    ask_questions.ACTIVE_PROJECT = project_id

@app.get("/")
def read_root():
    return {"message": "RAG API Server is running"}

@app.post("/api/ask")
async def ask(request: QuestionRequest):
    """Ask a question to the RAG system"""
    try:
        global _current_project
        project_to_use = request.project_id or _current_project
        set_active_project(project_to_use)
        
        # Get or create conversation
        conversation_id = request.conversation_id
        if not conversation_id:
            conversation_id = database.get_active_conversation(project_to_use)
        
        # Save user message
        database.add_message(conversation_id, "User", request.question)
        
        # Get conversation history from database or use provided
        history = None
        if request.conversation_history:
            history = [(h.get("role", "User"), h.get("content", "")) 
                      for h in request.conversation_history]
        else:
            # Load from database
            history = database.get_conversation_history(conversation_id)
            # Remove the last message (the current question) as it's already included
            if history:
                history = history[:-1]
        
        answer = ask_question(request.question, history)

        # Treat empty strings or falsy answers as a failure to produce an answer
        if not answer:
            answer = "I couldn't find an answer. Please make sure documents are indexed or the LLM service is reachable."
        
        # Save assistant message
        database.add_message(conversation_id, "Assistant", answer)
        
        return {
            "answer": answer, 
            "success": True,
            "conversation_id": conversation_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    project_id: Optional[str] = Form(default=None),
    isolate: Optional[str] = Form(default="false")
):
    """Upload and index a document"""
    try:
        global _current_project
        project_to_use = project_id or _current_project
        set_active_project(project_to_use)
        
        # Convert isolate string to boolean
        isolate_bool = isolate.lower() == "true" if isolate else False
        
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Handle project isolation
        if isolate_bool:
            proj_paths = project_paths(project_to_use)
            if os.path.exists(proj_paths["base"]):
                shutil.rmtree(proj_paths["base"])
                os.makedirs(proj_paths["base"], exist_ok=True)
        
        # Index the document to the project directory
        proj_paths = project_paths(project_to_use)
        os.makedirs(proj_paths["base"], exist_ok=True)
        
        # Temporarily set OUT_DIR in pdf_to_vectors module to use the project directory
        original_out_dir = pdf_to_vectors.OUT_DIR
        pdf_to_vectors.OUT_DIR = proj_paths["base"]
        try:
            pdf_to_vectors.ingest_document(temp_path)
        finally:
            # Restore original OUT_DIR
            pdf_to_vectors.OUT_DIR = original_out_dir
        
        # Clean up temp file
        os.remove(temp_path)
        
        return {"message": "File uploaded and indexed successfully", "success": True}
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/projects")
async def list_projects():
    """List all projects"""
    try:
        if not os.path.exists(RAG_DATA_DIR):
            return {"projects": [], "success": True}
        
        projects = []
        for name in sorted(os.listdir(RAG_DATA_DIR)):
            path = os.path.join(RAG_DATA_DIR, name)
            if os.path.isdir(path):
                proj_paths = project_paths(name)
                chunks_count = 0
                docs_count = 0
                
                if os.path.exists(proj_paths["chunks"]):
                    import pickle
                    with open(proj_paths["chunks"], "rb") as f:
                        data = pickle.load(f)
                        chunks_count = len(data.get("chunks", []))
                
                if os.path.exists(proj_paths["documents_pkl"]):
                    import pickle
                    with open(proj_paths["documents_pkl"], "rb") as f:
                        docs = pickle.load(f)
                        docs_count = len(docs)
                
                projects.append({
                    "id": name,
                    "chunks_count": chunks_count,
                    "documents_count": docs_count
                })
        
        return {"projects": projects, "success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/projects/switch")
async def switch_project(request: ProjectSwitchRequest):
    """Switch active project"""
    try:
        set_active_project(request.project_id)
        return {"message": f"Switched to project: {_current_project}", "success": True, "project_id": _current_project}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/projects/{project_id}/documents")
async def list_documents(project_id: str):
    """List documents in a project"""
    try:
        proj_paths = project_paths(project_id)
        documents = []
        
        # Try JSON first
        if os.path.exists(proj_paths["meta_json"]):
            import json
            with open(proj_paths["meta_json"], "r", encoding="utf-8") as f:
                documents = json.load(f)
        elif os.path.exists(proj_paths["documents_pkl"]):
            import pickle
            with open(proj_paths["documents_pkl"], "rb") as f:
                documents = pickle.load(f)
        
        return {"documents": documents, "success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/projects/{project_id}/stats")
async def get_project_stats(project_id: str):
    """Get statistics for a project"""
    try:
        proj_paths = project_paths(project_id)
        stats = {
            "chunks_count": 0,
            "documents_count": 0,
            "embedding_dim": 0,
            "has_index": False
        }
        
        if os.path.exists(proj_paths["chunks"]):
            import pickle
            with open(proj_paths["chunks"], "rb") as f:
                data = pickle.load(f)
                stats["chunks_count"] = len(data.get("chunks", []))
                stats["embedding_dim"] = data.get("embedding_dim", 0)
                stats["has_index"] = True
        
        if os.path.exists(proj_paths["documents_pkl"]):
            import pickle
            with open(proj_paths["documents_pkl"], "rb") as f:
                docs = pickle.load(f)
                stats["documents_count"] = len(docs)
        
        return {"stats": stats, "success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/projects/{project_id}/documents/{document_id}")
async def delete_document(project_id: str, document_id: str):
    """Delete a document from a project and re-index remaining documents"""
    try:
        proj_paths = project_paths(project_id)
        
        if not os.path.exists(proj_paths["base"]):
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Delete the document and re-index
        success = reindex_utils.delete_document_from_project(proj_paths["base"], document_id)
        
        if success:
            return {
                "message": f"Document '{document_id}' deleted and index rebuilt successfully",
                "success": True
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to delete document and re-index")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Chat/Database endpoints
@app.get("/api/projects/{project_id}/conversations")
async def get_conversations(project_id: str):
    """Get all conversations for a project"""
    try:
        conversations = database.get_all_conversations(project_id)
        return {"conversations": conversations, "success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversations/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: int):
    """Get all messages for a conversation"""
    try:
        conversation = database.get_conversation_by_id(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        messages = database.get_messages(conversation_id)
        return {
            "messages": messages,
            "conversation": conversation,
            "success": True
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/conversations")
async def create_conversation_endpoint(request: dict):
    """Create a new conversation"""
    try:
        project_id = request.get("project_id", "default")
        title = request.get("title")
        conversation_id = database.create_conversation(project_id, title)
        return {"conversation_id": conversation_id, "success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/projects/{project_id}/active-conversation")
async def get_active_conversation_endpoint(project_id: str):
    """Get or create the active conversation for a project"""
    try:
        conversation_id = database.get_active_conversation(project_id)
        messages = database.get_messages(conversation_id)
        conversation = database.get_conversation_by_id(conversation_id)
        return {
            "conversation_id": conversation_id,
            "messages": messages,
            "conversation": conversation,
            "success": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation_endpoint(conversation_id: int):
    """Delete a conversation"""
    try:
        success = database.delete_conversation(conversation_id)
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/conversations/{conversation_id}/clear")
async def clear_conversation_endpoint(conversation_id: int):
    """Clear all messages from a conversation"""
    try:
        success = database.clear_conversation(conversation_id)
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

