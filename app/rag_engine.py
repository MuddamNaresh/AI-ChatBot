"""
Retrieval-Augmented Generation (RAG) Engine.
Coordinates:
1. PDF Text Extraction (with page boundary tracking via pypdf)
2. Recursive Semantic Chunking (with overlap to preserve context)
3. Vector Indexing & Cosine Similarity Retrieval (using Gemini Embeddings + TF-IDF fallback)
4. Context Augmentation & Citation Synthesis
"""
import os
import re
import json
import math
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from pydantic import BaseModel
import numpy as np
import pypdf

from app.config import (
    VECTORS_DIR,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    SIMILARITY_THRESHOLD
)
from app.gemini_service import gemini_service

logger = logging.getLogger(__name__)

# ==========================================
# Data Models
# ==========================================

class TextChunk(BaseModel):
    """Represents a discrete segment of a document with source metadata."""
    chunk_id: str
    filename: str
    page: int
    chunk_index: int
    text: str
    token_count_approx: int = 0
    embedding: Optional[List[float]] = None

class RetrievalResult(BaseModel):
    """Encapsulates a retrieved chunk alongside its relevance score."""
    chunk: TextChunk
    score: float
    retrieval_method: str = "dense_embedding"

# ==========================================
# 1. PDF Text Extraction
# ==========================================

class PDFExtractor:
    """Extracts raw text from PDF files while preserving page numbers."""
    
    @staticmethod
    def extract_pages(pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Extracts text page-by-page from a given PDF.
        Returns a list of dicts: [{'page': 1, 'text': '...'}]
        """
        pages_data = []
        try:
            reader = pypdf.PdfReader(str(pdf_path))
            total_pages = len(reader.pages)
            
            for idx, page in enumerate(reader.pages):
                page_num = idx + 1
                raw_text = page.extract_text() or ""
                # Normalize whitespace while preserving line structure
                cleaned_text = PDFExtractor._clean_text(raw_text)
                
                if cleaned_text:
                    pages_data.append({
                        "page": page_num,
                        "text": cleaned_text
                    })
                    
            logger.info(f"Extracted {len(pages_data)} pages from {pdf_path.name}")
            return pages_data
        except Exception as e:
            logger.error(f"Error parsing PDF {pdf_path}: {e}")
            raise e

    @staticmethod
    def _clean_text(text: str) -> str:
        """Cleans extracted PDF text, removing duplicate whitespace and orphaned hyphens."""
        if not text:
            return ""
        # Remove hyphenated line breaks (e.g., "algo-\nrithm" -> "algorithm")
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        # Normalize multiple spaces and tabs
        text = re.sub(r'[ \t]+', ' ', text)
        # Normalize excessive newlines (keep at most 2 for paragraph separation)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

# ==========================================
# 2. Semantic Text Chunker
# ==========================================

class RecursiveTextChunker:
    """
    Splits text recursively using natural semantic boundaries:
    paragraphs -> sentences -> words -> characters.
    Ensures context continuity via configurable window overlap.
    """
    
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Hierarchy of separators for natural splitting
        self.separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]

    def chunk_document(self, pages: List[Dict[str, Any]], filename: str) -> List[TextChunk]:
        """
        Takes extracted pages and produces structured TextChunk objects.
        """
        all_chunks: List[TextChunk] = []
        chunk_counter = 0

        for page_data in pages:
            page_num = page_data["page"]
            page_text = page_data["text"]
            
            page_splits = self._split_text(page_text)
            
            for split_text in page_splits:
                if len(split_text.strip()) < 20:
                    continue  # Ignore trivial tiny snippets
                
                chunk_id = f"{filename}_p{page_num}_c{chunk_counter}"
                approx_tokens = max(1, len(split_text) // 4)
                
                all_chunks.append(TextChunk(
                    chunk_id=chunk_id,
                    filename=filename,
                    page=page_num,
                    chunk_index=chunk_counter,
                    text=split_text.strip(),
                    token_count_approx=approx_tokens
                ))
                chunk_counter += 1
                
        return all_chunks

    def _split_text(self, text: str) -> List[str]:
        """Recursive chunking core algorithm."""
        final_chunks: List[str] = []
        
        def _recursive_split(current_text: str, sep_index: int):
            if len(current_text) <= self.chunk_size or sep_index >= len(self.separators):
                if current_text.strip():
                    final_chunks.append(current_text.strip())
                return

            separator = self.separators[sep_index]
            if separator == "":
                # Hard slice character by character if no separator fits
                splits = [current_text[i:i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size - self.chunk_overlap)]
                final_chunks.extend(splits)
                return

            parts = current_text.split(separator)
            buffer = ""

            for part in parts:
                candidate = (buffer + separator + part) if buffer else part
                if len(candidate) <= self.chunk_size:
                    buffer = candidate
                else:
                    if buffer:
                        final_chunks.append(buffer.strip())
                        # Sliding window overlap: keep tail of buffer
                        overlap_start = max(0, len(buffer) - self.chunk_overlap)
                        buffer = buffer[overlap_start:] + separator + part
                    else:
                        # Single piece is already larger than chunk_size, recurse with next finer separator
                        _recursive_split(part, sep_index + 1)
                        buffer = ""

            if buffer.strip():
                final_chunks.append(buffer.strip())

        _recursive_split(text, 0)
        return final_chunks

# ==========================================
# 3. Vector Store & Retrieval Engine
# ==========================================

class VectorStore:
    """
    In-memory vector database with persistence.
    Computes cosine similarity over dense vector embeddings.
    Provides a built-in TF-IDF fallback when API key is unconfigured.
    """
    
    def __init__(self, filename: str):
        self.filename = filename
        self.chunks: List[TextChunk] = []
        self.embeddings: Optional[np.ndarray] = None  # Shape: (N, D)
        self.is_dense: bool = False
        self._tfidf_vocab: Dict[str, int] = {}
        self._tfidf_idf: Optional[np.ndarray] = None
        self.index_file = VECTORS_DIR / f"{filename}.json"
        
        # Load from disk if previously cached
        self._load_from_disk()

    def _load_from_disk(self):
        """Loads chunks, precomputed embeddings, and vocabulary from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.chunks = [TextChunk(**item) for item in data.get("chunks", [])]
                    raw_emb = [c.embedding for c in self.chunks if c.embedding is not None]
                    if len(raw_emb) == len(self.chunks) and len(raw_emb) > 0:
                        self.embeddings = np.array(raw_emb, dtype=np.float32)
                    
                    self.is_dense = data.get("is_dense", False)
                    self._tfidf_vocab = data.get("tfidf_vocab", {})
                    raw_idf = data.get("tfidf_idf", [])
                    if raw_idf:
                        self._tfidf_idf = np.array(raw_idf, dtype=np.float32)
                    elif not self.is_dense and self.chunks:
                        # Reconstruct TF-IDF vocab and IDF on the fly
                        self._reconstruct_tfidf([c.text for c in self.chunks])
                        
                logger.info(f"Loaded existing index for {self.filename} with {len(self.chunks)} chunks.")
            except Exception as e:
                logger.warning(f"Could not load index cache for {self.filename}: {e}")

    def save_to_disk(self):
        """Saves current chunks, embeddings, and feature metadata to JSON cache."""
        try:
            idf_list = self._tfidf_idf.tolist() if self._tfidf_idf is not None else []
            payload = {
                "filename": self.filename,
                "total_chunks": len(self.chunks),
                "is_dense": self.is_dense,
                "tfidf_vocab": self._tfidf_vocab,
                "tfidf_idf": idf_list,
                "chunks": [c.model_dump() for c in self.chunks]
            }
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            logger.info(f"Saved vector index to {self.index_file}")
        except Exception as e:
            logger.error(f"Failed to save vector index for {self.filename}: {e}")

    def build_index(self, chunks: List[TextChunk]):
        """
        Generates embeddings for chunks and prepares the vector matrix.
        Uses Gemini's embedding model if configured, or TF-IDF if offline.
        """
        self.chunks = chunks
        texts = [c.text for c in chunks]

        if gemini_service.is_configured():
            try:
                logger.info(f"Generating Gemini embeddings for {len(chunks)} chunks in {self.filename}...")
                embeddings_list = gemini_service.get_embeddings(texts)
                for chunk, emb in zip(self.chunks, embeddings_list):
                    chunk.embedding = emb
                self.embeddings = np.array(embeddings_list, dtype=np.float32)
                self.is_dense = True
                self.save_to_disk()
                return
            except Exception as e:
                logger.warning(f"Gemini embedding call failed ({e}); falling back to term frequency vectorization.")

        # Fallback: TF-IDF vectorization (100% offline, zero external API required)
        self.is_dense = False
        self._build_tfidf_index(texts)
        self.save_to_disk()

    def _build_tfidf_index(self, texts: List[str]):
        """Builds an internal Term Frequency-Inverse Document Frequency (TF-IDF) representation."""
        self._reconstruct_tfidf(texts)

    def _reconstruct_tfidf(self, texts: List[str]):
        """Builds vocabulary and IDF vectors from texts."""
        vocabulary: Dict[str, int] = {}
        doc_word_counts = []
        
        for text in texts:
            words = re.findall(r'\b[a-zA-Z0-9_]{2,}\b', text.lower())
            counts: Dict[str, int] = {}
            for w in words:
                counts[w] = counts.get(w, 0) + 1
                if w not in vocabulary:
                    vocabulary[w] = len(vocabulary)
            doc_word_counts.append(counts)

        num_docs = len(texts)
        vocab_size = len(vocabulary)
        
        # Calculate IDF
        idf = np.zeros(vocab_size, dtype=np.float32)
        for w, idx in vocabulary.items():
            doc_freq = sum(1 for counts in doc_word_counts if w in counts)
            idf[idx] = math.log((1 + num_docs) / (1 + doc_freq)) + 1.0

        # Construct TF-IDF matrix
        matrix = np.zeros((num_docs, vocab_size), dtype=np.float32)
        for i, counts in enumerate(doc_word_counts):
            total_words = sum(counts.values()) or 1
            for w, count in counts.items():
                w_idx = vocabulary[w]
                tf = count / total_words
                matrix[i, w_idx] = tf * idf[w_idx]

        # Normalize rows to unit length for cosine similarity
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_matrix = matrix / norms

        self.embeddings = normalized_matrix
        for i, chunk in enumerate(self.chunks):
            chunk.embedding = normalized_matrix[i].tolist()
        self._tfidf_vocab = vocabulary
        self._tfidf_idf = idf

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[RetrievalResult]:
        """
        Executes a vector similarity query against indexed chunks.
        Computes cosine similarity and returns top-K ranking chunks.
        """
        if not self.chunks or self.embeddings is None:
            return []

        query_vector: Optional[np.ndarray] = None
        method = "dense_embedding"

        # Try dense embedding first if document has dense embeddings and Gemini is configured
        if self.is_dense and gemini_service.is_configured():
            try:
                emb = gemini_service.get_embeddings([query])[0]
                dense_vec = np.array(emb, dtype=np.float32)
                if dense_vec.shape[0] == self.embeddings.shape[1]:
                    query_vector = dense_vec
                    method = "gemini_embedding"
            except Exception as e:
                logger.warning(f"Query embedding with Gemini failed: {e}")

        # Fallback to TF-IDF if query_vector is not set
        if query_vector is None:
            method = "tfidf_similarity"
            query_vector = self._vectorize_query_tfidf(query)

        # Normalize query vector
        norm_q = np.linalg.norm(query_vector)
        if norm_q > 0:
            query_vector = query_vector / norm_q
        else:
            # Substring / keyword fallback if query has no direct match in tfidf vocab
            return self._keyword_fallback_search(query, top_k)

        # Cosine Similarity: S = (X . Q)
        scores = np.dot(self.embeddings, query_vector)

        # Get top-K indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0.0 and len(results) > 0:
                continue
            # Cap score between 0.0 and 1.0
            score_normalized = max(0.0, min(1.0, (score + 1.0) / 2.0 if method == "gemini_embedding" else score))
            results.append(RetrievalResult(
                chunk=self.chunks[idx],
                score=round(score_normalized, 4),
                retrieval_method=method
            ))

        return results

    def _keyword_fallback_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Simple token overlap fallback when query words are rare."""
        query_terms = set(re.findall(r'\b[a-zA-Z0-9_]{2,}\b', query.lower()))
        scored_chunks = []
        for chunk in self.chunks:
            chunk_terms = set(re.findall(r'\b[a-zA-Z0-9_]{2,}\b', chunk.text.lower()))
            overlap = len(query_terms.intersection(chunk_terms))
            if overlap > 0:
                score = min(0.9, overlap / (len(query_terms) + 1.0))
                scored_chunks.append((chunk, score))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return [
            RetrievalResult(chunk=c, score=round(s, 4), retrieval_method="keyword_overlap")
            for c, s in scored_chunks[:top_k]
        ]

    def _vectorize_query_tfidf(self, query: str) -> np.ndarray:
        """Projects a search query into the document TF-IDF feature space."""
        vocab = getattr(self, "_tfidf_vocab", {})
        idf = getattr(self, "_tfidf_idf", None)
        
        vector = np.zeros(len(vocab), dtype=np.float32)
        words = re.findall(r'\b[a-zA-Z0-9_]{2,}\b', query.lower())
        if not words or not vocab or idf is None:
            return vector

        counts: Dict[str, int] = {}
        for w in words:
            if w in vocab:
                counts[w] = counts.get(w, 0) + 1

        total = sum(counts.values()) or 1
        for w, c in counts.items():
            idx = vocab[w]
            tf = c / total
            vector[idx] = tf * idf[idx]

        return vector

# ==========================================
# 4. RAG Pipeline Coordinator
# ==========================================

class RAGPipeline:
    """
    Main manager for indexing documents, retrieving relevant context,
    and synthesizing context-grounded responses.
    """
    
    def __init__(self):
        self.chunker = RecursiveTextChunker()
        self.stores: Dict[str, VectorStore] = {}
        # Pre-load existing vector indexes from VECTORS_DIR
        self._load_existing_indices()

    def _load_existing_indices(self):
        """Scans the vectors directory and attaches cached indices."""
        for file in VECTORS_DIR.glob("*.json"):
            filename = file.stem
            store = VectorStore(filename)
            if store.chunks:
                self.stores[filename] = store

    def index_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Complete ingestion flow:
        PDF -> Page Extraction -> Recursive Chunking -> Vector Indexing -> Persistence.
        """
        filename = pdf_path.name
        pages = PDFExtractor.extract_pages(pdf_path)
        chunks = self.chunker.chunk_document(pages, filename)
        
        store = VectorStore(filename)
        store.build_index(chunks)
        self.stores[filename] = store

        return {
            "filename": filename,
            "total_pages": len(pages),
            "total_chunks": len(chunks)
        }

    def retrieve(self, query: str, filename: Optional[str] = None, top_k: int = DEFAULT_TOP_K) -> List[RetrievalResult]:
        """
        Retrieves top relevant chunks from a specified document or across all indexed documents.
        """
        if filename and filename in self.stores:
            return self.stores[filename].search(query, top_k=top_k)
        
        # If no specific document, search across all available stores and pool results
        all_results = []
        for store in self.stores.values():
            all_results.extend(store.search(query, top_k=top_k))
        
        # Sort pooled results descending by similarity score
        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results[:top_k]

    def build_augmented_prompt(self, user_query: str, retrieved_chunks: List[RetrievalResult]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Constructs a context-augmented prompt instructing Gemini to ground its answer in the excerpts.
        Returns (augmented_prompt, sources_list).
        """
        sources = []
        context_blocks = []

        for idx, result in enumerate(retrieved_chunks):
            chunk = result.chunk
            source_info = {
                "citation_id": idx + 1,
                "filename": chunk.filename,
                "page": chunk.page,
                "score": result.score,
                "text_preview": chunk.text[:180] + "..." if len(chunk.text) > 180 else chunk.text
            }
            sources.append(source_info)
            
            context_blocks.append(
                f"[Source {idx + 1} | Document: {chunk.filename} | Page: {chunk.page}]\n{chunk.text}"
            )

        context_text = "\n\n".join(context_blocks)
        
        prompt = (
            f"You are an expert Data Science and AI Assistant. Answer the user's question based strictly "
            f"on the provided document excerpts below. "
            f"If the information is not directly contained in the context, clearly explain what is and isn't covered, "
            f"and avoid making up unsupported facts. "
            f"Cite the source page numbers (e.g., [Page X] or [Source Y]) where relevant.\n\n"
            f"--- RELEVANT DOCUMENT EXCERPTS ---\n"
            f"{context_text}\n"
            f"--- END OF EXCERPTS ---\n\n"
            f"User Question: {user_query}\n"
            f"Detailed Answer:"
        )

        return prompt, sources

    def delete_document_index(self, filename: str):
        """Removes a document's vector store from memory and disk."""
        if filename in self.stores:
            del self.stores[filename]
        cache_path = VECTORS_DIR / f"{filename}.json"
        if cache_path.exists():
            cache_path.unlink()

# Singleton RAG Pipeline instance
rag_pipeline = RAGPipeline()
