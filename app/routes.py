"""
FastAPI route definitions for the AI Chatbot and RAG Engine.
Provides REST and SSE endpoints for:
- Chat completions (General & RAG)
- Real-time token streaming
- PDF Document upload & indexing
- Session & history management
- API Key verification & settings
"""
import os
import shutil
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import (
    UPLOADS_DIR,
    SAMPLE_DATA_DIR,
    ENV_FILE,
    DEFAULT_CHAT_MODEL,
    DEFAULT_TOP_K,
    get_gemini_api_key
)
from app.gemini_service import gemini_service
from app.rag_engine import rag_pipeline
from app import database

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

# ==========================================
# Pydantic Request / Response Schemas
# ==========================================

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Active conversation session ID")
    message: str = Field(..., min_length=1, description="User prompt")
    mode: str = Field("rag", description="Chat mode: 'rag' or 'general'")
    document_name: Optional[str] = Field(None, description="Optional target document for RAG")
    model: Optional[str] = Field(DEFAULT_CHAT_MODEL, description="Gemini model name")
    temperature: float = Field(0.7, ge=0.0, le=1.0, description="Creativity temperature")
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=10, description="Number of context chunks to retrieve")

class CreateSessionRequest(BaseModel):
    title: Optional[str] = "New Chat"
    active_document: Optional[str] = None

class UpdateSessionRequest(BaseModel):
    title: Optional[str] = None
    active_document: Optional[str] = None

class KeyVerificationRequest(BaseModel):
    api_key: str

class SaveKeyRequest(BaseModel):
    api_key: str

class RAGPreviewRequest(BaseModel):
    query: str
    document_name: Optional[str] = None
    top_k: int = 4

# ==========================================
# Health & Settings Endpoints
# ==========================================

@router.get("/health")
def health_check():
    """Returns application health and Gemini API key configuration status."""
    has_key = gemini_service.is_configured()
    # Mask key for privacy
    raw_key = gemini_service.api_key
    masked = f"{raw_key[:4]}...{raw_key[-4:]}" if len(raw_key) > 8 else ("Configured" if has_key else "Missing")
    
    return {
        "status": "healthy",
        "api_key_configured": has_key,
        "masked_key": masked,
        "active_chat_model": DEFAULT_CHAT_MODEL,
        "indexed_documents_count": len(rag_pipeline.stores)
    }

@router.post("/settings/verify-key")
def verify_api_key(req: KeyVerificationRequest):
    """Tests if a candidate Gemini API key is valid."""
    result = gemini_service.validate_api_key(req.api_key.strip())
    return result

@router.post("/settings/save-key")
def save_api_key(req: SaveKeyRequest):
    """Saves the provided API key to the .env file and updates the in-memory client."""
    new_key = req.api_key.strip()
    if not new_key:
        raise HTTPException(status_code=400, detail="API key cannot be empty.")
    
    # Verify key first
    validation = gemini_service.validate_api_key(new_key)
    if not validation["valid"]:
        raise HTTPException(status_code=400, detail=f"Invalid API Key: {validation['message']}")

    # Write to .env
    lines = []
    if ENV_FILE.exists():
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
    
    key_found = False
    new_lines = []
    for line in lines:
        if line.strip().startswith("GEMINI_API_KEY="):
            new_lines.append(f"GEMINI_API_KEY={new_key}\n")
            key_found = True
        else:
            new_lines.append(line)
    if not key_found:
        new_lines.append(f"GEMINI_API_KEY={new_key}\n")

    with open(ENV_FILE, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    # Update runtime environment and client
    os.environ["GEMINI_API_KEY"] = new_key
    gemini_service._explicit_key = new_key
    gemini_service._client = None  # Force re-init

    return {"success": True, "message": "API Key saved and activated successfully."}

# ==========================================
# Session Management Endpoints
# ==========================================

@router.get("/sessions")
def list_sessions():
    """Retrieve all chat sessions."""
    return database.get_all_sessions()

@router.post("/sessions")
def create_session(req: CreateSessionRequest):
    """Create a new conversation session."""
    session = database.create_session(
        title=req.title or "New Chat",
        active_document=req.active_document
    )
    return session

@router.get("/sessions/{session_id}")
def get_session(session_id: str):
    """Retrieve a session with all its messages."""
    session = database.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    messages = database.get_session_messages(session_id)
    return {"session": session, "messages": messages}

@router.patch("/sessions/{session_id}")
def update_session(session_id: str, req: UpdateSessionRequest):
    """Update session title or attached document."""
    session = database.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    database.update_session(
        session_id=session_id,
        title=req.title,
        active_document=req.active_document
    )
    return {"success": True}

@router.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """Delete a conversation session."""
    database.delete_session(session_id)
    return {"success": True}

# ==========================================
# Document & RAG Ingestion Endpoints
# ==========================================

@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document, extract text, chunk, and index into the vector store.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files (.pdf) are supported.")

    dest_path = UPLOADS_DIR / file.filename
    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_size = dest_path.stat().st_size
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    try:
        # Run RAG indexing pipeline
        index_summary = rag_pipeline.index_pdf(dest_path)
        
        # Register in SQLite database
        doc_record = database.register_document(
            filename=file.filename,
            filepath=str(dest_path),
            file_size_bytes=file_size,
            total_pages=index_summary["total_pages"],
            total_chunks=index_summary["total_chunks"]
        )
        return {
            "success": True,
            "document": doc_record,
            "message": f"Document '{file.filename}' processed successfully: {index_summary['total_pages']} pages, {index_summary['total_chunks']} chunks indexed."
        }
    except Exception as e:
        logger.error(f"Error indexing document {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error indexing PDF: {str(e)}")

@router.post("/documents/load-sample")
def load_sample_document():
    """
    Loads and indexes the pre-built sample 'Data_Science_Internship_Handbook.pdf'.
    Provides an immediate demonstration without requiring manual upload.
    """
    sample_file = SAMPLE_DATA_DIR / "Data_Science_Internship_Handbook.pdf"
    if not sample_file.exists():
        # Generate on the fly if needed
        from sample_data.generate_sample import generate_handbook
        generate_handbook(sample_file)

    target_path = UPLOADS_DIR / sample_file.name
    shutil.copy(sample_file, target_path)
    file_size = target_path.stat().st_size

    index_summary = rag_pipeline.index_pdf(target_path)
    doc_record = database.register_document(
        filename=sample_file.name,
        filepath=str(target_path),
        file_size_bytes=file_size,
        total_pages=index_summary["total_pages"],
        total_chunks=index_summary["total_chunks"]
    )
    return {
        "success": True,
        "document": doc_record,
        "message": "Sample Data Science Internship Handbook loaded and indexed successfully!"
    }

@router.get("/documents")
def list_documents():
    """Lists all uploaded documents."""
    return database.get_all_documents()

@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    """Deletes a document from the database, disk, and vector memory."""
    doc = database.delete_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    
    filename = doc["filename"]
    rag_pipeline.delete_document_index(filename)
    
    filepath = Path(doc["filepath"])
    if filepath.exists():
        filepath.unlink()
        
    return {"success": True, "message": f"Document '{filename}' deleted."}

@router.post("/rag/preview")
def preview_rag_retrieval(req: RAGPreviewRequest):
    """
    Diagnostic tool for interns: inspects top-K retrieved chunks and similarity scores
    for any query without generating a completion.
    """
    results = rag_pipeline.retrieve(req.query, filename=req.document_name, top_k=req.top_k)
    return {
        "query": req.query,
        "retrieved_count": len(results),
        "results": [
            {
                "chunk_id": r.chunk.chunk_id,
                "filename": r.chunk.filename,
                "page": r.chunk.page,
                "score": r.score,
                "method": r.retrieval_method,
                "text": r.chunk.text
            }
            for r in results
        ]
    }

# ==========================================
# Chat Generation Endpoints
# ==========================================

@router.post("/chat")
def chat(req: ChatRequest):
    """
    Main chat endpoint supporting both General Conversation and Document RAG.
    Maintains multi-turn context and saves turns to SQLite.
    """
    session = database.get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    user_query = req.message.strip()
    sources = []
    effective_prompt = user_query
    system_instruction = (
        "You are an intelligent, articulate Data Science and AI Assistant. "
        "Provide clear, informative, well-structured responses formatted in Markdown."
    )

    # 1. RAG Context Retrieval if requested
    if req.mode == "rag":
        target_doc = req.document_name or session.get("active_document")
        retrieved_chunks = rag_pipeline.retrieve(user_query, filename=target_doc, top_k=req.top_k)
        
        if retrieved_chunks:
            effective_prompt, sources = rag_pipeline.build_augmented_prompt(user_query, retrieved_chunks)
            system_instruction = (
                "You are an expert AI Document Assistant. "
                "Synthesize answers based on the provided document excerpts. "
                "Include clear citations like [Page X] or [Source Y] when referencing specific facts. "
                "If the context does not contain the answer, politely state that the document does not cover it."
            )
        else:
            # If no document indexed or no matches, clarify in the prompt
            effective_prompt = (
                f"Note: No relevant document excerpts were retrieved. "
                f"Please answer the user's question with general knowledge and note that no specific document context was available.\n\n"
                f"Question: {user_query}"
            )

    # 2. Retrieve multi-turn conversation history from SQLite
    history_records = database.get_session_messages(req.session_id)
    # Convert previous messages into [{'role': 'user'|'model', 'content': '...'}]
    # Keep last 6 turns for optimal context window balance
    recent_history = []
    for msg in history_records[-6:]:
        role = "user" if msg["role"] == "user" else "model"
        recent_history.append({"role": role, "content": msg["content"]})

    # Save user message to database
    database.add_message(
        session_id=req.session_id,
        role="user",
        content=user_query
    )

    # 3. Call Gemini Model
    model_name = req.model or DEFAULT_CHAT_MODEL
    try:
        assistant_reply = gemini_service.generate_chat_response(
            user_message=effective_prompt,
            conversation_history=recent_history,
            system_instruction=system_instruction,
            model=model_name,
            temperature=req.temperature
        )
    except Exception as e:
        logger.error(f"Gemini generation error: {e}")
        # Provide graceful error message
        if not gemini_service.is_configured():
            assistant_reply = (
                "⚠️ **Gemini API Key Required**\n\n"
                "To generate responses, please configure your Google Gemini API key. "
                "Click the **Settings (⚙️)** icon in the top header, enter your key, and click **Save**."
            )
        else:
            assistant_reply = f"❌ **Error generating response:** {str(e)}"

    # Save assistant message to database
    saved_msg = database.add_message(
        session_id=req.session_id,
        role="assistant",
        content=assistant_reply,
        sources=sources,
        model_used=model_name
    )

    # Auto-generate a title if this was the first turn and title is "New Chat"
    if session.get("title") == "New Chat" and len(history_records) == 0:
        clean_title = user_query[:32] + ("..." if len(user_query) > 32 else "")
        database.update_session(session_id=req.session_id, title=clean_title)

    return {
        "message": saved_msg,
        "sources": sources,
        "model_used": model_name
    }

@router.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """
    Streams tokens in real time via Server-Sent Events (SSE).
    Followed by a final JSON metadata event containing source citations.
    """
    session = database.get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    user_query = req.message.strip()
    sources = []
    effective_prompt = user_query
    system_instruction = "You are an expert Data Science and AI Assistant. Provide helpful, accurate Markdown responses."

    if req.mode == "rag":
        target_doc = req.document_name or session.get("active_document")
        retrieved_chunks = rag_pipeline.retrieve(user_query, filename=target_doc, top_k=req.top_k)
        if retrieved_chunks:
            effective_prompt, sources = rag_pipeline.build_augmented_prompt(user_query, retrieved_chunks)
            system_instruction = (
                "You are an expert AI Document Assistant. "
                "Base your response strictly on the provided document excerpts and cite source pages."
            )

    history_records = database.get_session_messages(req.session_id)
    recent_history = [
        {"role": "user" if m["role"] == "user" else "model", "content": m["content"]}
        for m in history_records[-6:]
    ]

    # Save user message
    database.add_message(session_id=req.session_id, role="user", content=user_query)

    def event_generator():
        accumulated_text = ""
        model_name = req.model or DEFAULT_CHAT_MODEL
        try:
            for token in gemini_service.generate_chat_stream(
                user_message=effective_prompt,
                conversation_history=recent_history,
                system_instruction=system_instruction,
                model=model_name,
                temperature=req.temperature
            ):
                accumulated_text += token
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            err_msg = f"\n\n[Error during streaming: {str(e)}]"
            accumulated_text += err_msg
            yield f"data: {json.dumps({'token': err_msg})}\n\n"

        # Save assistant message to database once stream completes
        saved = database.add_message(
            session_id=req.session_id,
            role="assistant",
            content=accumulated_text,
            sources=sources,
            model_used=model_name
        )

        # Send final metadata event
        meta = {
            "done": True,
            "message_id": saved["id"],
            "sources": sources,
            "model_used": model_name
        }
        yield f"data: {json.dumps(meta)}\n\n"

        # Auto-update title
        if session.get("title") == "New Chat" and len(history_records) == 0:
            clean_title = user_query[:32] + ("..." if len(user_query) > 32 else "")
            database.update_session(session_id=req.session_id, title=clean_title)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
