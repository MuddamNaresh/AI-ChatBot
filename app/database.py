"""
Database persistence layer using SQLite.
Stores chat sessions, message histories, and document metadata.
Clean, reliable, and thread-safe for local development.
"""
import sqlite3
import json
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
from app.config import DB_PATH

from contextlib import contextmanager

def get_utc_now() -> str:
    """Return ISO formatted UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()

@contextmanager
def get_connection():
    """Create a managed SQLite connection that automatically commits and closes."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    """Initialize database tables on application startup."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # 1. Sessions table: tracks individual conversation threads
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                active_document TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 2. Messages table: tracks chat turns (User / Assistant)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT,
                model_used TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
        """)
        
        # 3. Documents table: tracks uploaded PDF documents and indexing status
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                file_size_bytes INTEGER DEFAULT 0,
                total_pages INTEGER DEFAULT 0,
                total_chunks INTEGER DEFAULT 0,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()

# ==========================================
# Session Management Methods
# ==========================================

def create_session(title: str = "New Chat", active_document: Optional[str] = None) -> Dict[str, Any]:
    """Create and return a new conversation session."""
    session_id = str(uuid.uuid4())
    now = get_utc_now()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (id, title, active_document, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, title, active_document, now, now)
        )
        conn.commit()
    return {
        "id": session_id,
        "title": title,
        "active_document": active_document,
        "created_at": now,
        "updated_at": now
    }

def get_all_sessions() -> List[Dict[str, Any]]:
    """Retrieve all sessions ordered by most recently updated."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions ORDER BY updated_at DESC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single session by its unique ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

def update_session(session_id: str, title: Optional[str] = None, active_document: Optional[str] = None):
    """Update title or active document for a session."""
    now = get_utc_now()
    with get_connection() as conn:
        cursor = conn.cursor()
        if title is not None and active_document is not None:
            cursor.execute(
                "UPDATE sessions SET title = ?, active_document = ?, updated_at = ? WHERE id = ?",
                (title, active_document, now, session_id)
            )
        elif title is not None:
            cursor.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, session_id)
            )
        elif active_document is not None:
            cursor.execute(
                "UPDATE sessions SET active_document = ?, updated_at = ? WHERE id = ?",
                (active_document, now, session_id)
            )
        conn.commit()

def delete_session(session_id: str):
    """Delete a conversation session and all its associated messages."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()

# ==========================================
# Message Management Methods
# ==========================================

def add_message(
    session_id: str,
    role: str,
    content: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    model_used: Optional[str] = None
) -> Dict[str, Any]:
    """Add a chat message (user or assistant) to a session."""
    message_id = str(uuid.uuid4())
    sources_str = json.dumps(sources) if sources else None
    now = get_utc_now()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO messages (id, session_id, role, content, sources, model_used, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (message_id, session_id, role, content, sources_str, model_used, now)
        )
        # Touch session's updated_at
        cursor.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
        conn.commit()
    
    return {
        "id": message_id,
        "session_id": session_id,
        "role": role,
        "content": content,
        "sources": sources,
        "model_used": model_used,
        "created_at": now
    }

def get_session_messages(session_id: str) -> List[Dict[str, Any]]:
    """Retrieve all messages for a session in chronological order."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,)
        )
        rows = cursor.fetchall()
        result = []
        for row in rows:
            item = dict(row)
            if item.get("sources"):
                try:
                    item["sources"] = json.loads(item["sources"])
                except Exception:
                    item["sources"] = []
            else:
                item["sources"] = []
            result.append(item)
        return result

# ==========================================
# Document Registry Methods
# ==========================================

def register_document(
    filename: str,
    filepath: str,
    file_size_bytes: int,
    total_pages: int,
    total_chunks: int
) -> Dict[str, Any]:
    """Record an uploaded and indexed document in the database."""
    doc_id = str(uuid.uuid4())
    now = get_utc_now()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO documents (id, filename, filepath, file_size_bytes, total_pages, total_chunks, uploaded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (doc_id, filename, filepath, file_size_bytes, total_pages, total_chunks, now)
        )
        conn.commit()
    return {
        "id": doc_id,
        "filename": filename,
        "filepath": filepath,
        "file_size_bytes": file_size_bytes,
        "total_pages": total_pages,
        "total_chunks": total_chunks,
        "uploaded_at": now
    }

def get_all_documents() -> List[Dict[str, Any]]:
    """Get list of all registered documents."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents ORDER BY uploaded_at DESC")
        return [dict(row) for row in cursor.fetchall()]

def delete_document(doc_id: str) -> Optional[Dict[str, Any]]:
    """Delete a document record by ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        row = cursor.fetchone()
        if not row:
            return None
        doc_data = dict(row)
        cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
        return doc_data
