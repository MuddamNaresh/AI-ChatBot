"""
FastAPI Application Entry Point.
Initializes the application, sets up middleware, static assets,
API routes, and startup lifecycle hooks.
"""
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import APP_NAME, APP_VERSION, BASE_DIR, SAMPLE_DATA_DIR, UPLOADS_DIR
from app.database import init_db
from app.routes import router as api_router
from app.rag_engine import rag_pipeline

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("auragen_app")

# Initialize FastAPI App
app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="A complete full-stack AI-powered Chatbot with Google Gemini 3.8 Flash and PDF RAG."
)

# Enable CORS for cross-origin local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(api_router)

# Mount Static Files (Frontend UI)
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.on_event("startup")
async def on_startup():
    """Initializes SQLite database and pre-loads sample document if none exists."""
    logger.info("Initializing SQLite database...")
    init_db()
    
    from app.database import get_all_documents, register_document
    docs_in_db = get_all_documents()
    
    # Pre-index sample handbook if database has no documents yet
    if len(docs_in_db) == 0:
        sample_file = SAMPLE_DATA_DIR / "Data_Science_Internship_Handbook.pdf"
        if not sample_file.exists():
            from sample_data.generate_sample import generate_handbook
            generate_handbook(sample_file)
            
        logger.info("Pre-indexing sample Data Science Handbook for immediate demonstration...")
        dest = UPLOADS_DIR / sample_file.name
        if not dest.exists():
            import shutil
            shutil.copy(sample_file, dest)
        try:
            summary = rag_pipeline.index_pdf(dest)
            register_document(
                filename=sample_file.name,
                filepath=str(dest),
                file_size_bytes=dest.stat().st_size,
                total_pages=summary["total_pages"],
                total_chunks=summary["total_chunks"]
            )
            logger.info("Sample handbook successfully indexed on startup!")
        except Exception as e:
            logger.warning(f"Failed to auto-index sample document: {e}")

@app.get("/")
def serve_index():
    """Serves the Single Page Application HTML."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return {"message": "Frontend index.html is being prepared."}
    return FileResponse(str(index_path))
