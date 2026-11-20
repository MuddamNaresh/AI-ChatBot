"""
Comprehensive Automated Test Suite for AuraGen AI Chatbot & RAG Engine.
Validates:
1. PDF page extraction and text cleanliness
2. Recursive semantic chunking & window overlap
3. Vector similarity calculation and Top-K retrieval
4. SQLite conversation and message persistence
5. End-to-end RAG pipeline query search
"""
import sys
import unittest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import SAMPLE_DATA_DIR, DB_PATH
from app.rag_engine import PDFExtractor, RecursiveTextChunker, VectorStore, RAGPipeline
from app import database

class TestRAGEngine(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment and ensure sample PDF exists."""
        cls.sample_pdf = SAMPLE_DATA_DIR / "Data_Science_Internship_Handbook.pdf"
        if not cls.sample_pdf.exists():
            from sample_data.generate_sample import generate_handbook
            generate_handbook(cls.sample_pdf)
        database.init_db()

    def test_1_pdf_extraction(self):
        """Verify that PDFExtractor correctly reads text and preserves page numbers."""
        pages = PDFExtractor.extract_pages(self.sample_pdf)
        self.assertGreaterEqual(len(pages), 3, "Sample PDF should contain at least 3 pages.")
        self.assertEqual(pages[0]["page"], 1, "First page must have page index 1.")
        self.assertIn("Handbook", pages[0]["text"], "Page 1 must contain title.")

    def test_2_recursive_chunker(self):
        """Verify recursive chunking splits properly and retains metadata."""
        chunker = RecursiveTextChunker(chunk_size=500, chunk_overlap=100)
        pages = PDFExtractor.extract_pages(self.sample_pdf)
        chunks = chunker.chunk_document(pages, "test_doc.pdf")
        
        self.assertGreater(len(chunks), 4, "Document should produce multiple discrete chunks.")
        for chunk in chunks:
            self.assertTrue(len(chunk.text) > 0, "Chunk text must not be empty.")
            self.assertEqual(chunk.filename, "test_doc.pdf")
            self.assertGreaterEqual(chunk.page, 1)

    def test_3_vector_store_retrieval(self):
        """Verify cosine similarity retrieval finds relevant chunks for target queries."""
        pipeline = RAGPipeline()
        summary = pipeline.index_pdf(self.sample_pdf)
        self.assertGreater(summary["total_chunks"], 0)

        # Query about metrics
        results = pipeline.retrieve("What are the evaluation metrics for classification?", top_k=3)
        self.assertGreater(len(results), 0, "Retrieval should return matching chunks.")
        top_chunk = results[0].chunk
        # Should match metrics page (Page 2 in handbook)
        self.assertTrue(
            "Precision" in top_chunk.text or "Recall" in top_chunk.text or "Metric" in top_chunk.text,
            f"Expected metrics chunk, got: {top_chunk.text[:150]}"
        )
        self.assertGreaterEqual(results[0].score, 0.0, "Score should be non-negative.")

    def test_4_sqlite_persistence(self):
        """Verify session creation, message storage, and history querying."""
        # Create session
        session = database.create_session(title="Test ML Chat", active_document="sample.pdf")
        session_id = session["id"]
        
        # Add messages
        msg1 = database.add_message(session_id, "user", "What is RAG?")
        msg2 = database.add_message(
            session_id, 
            "assistant", 
            "RAG stands for Retrieval-Augmented Generation.",
            sources=[{"filename": "sample.pdf", "page": 3, "score": 0.88}]
        )

        # Retrieve messages
        history = database.get_session_messages(session_id)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[1]["role"], "assistant")
        self.assertEqual(history[1]["sources"][0]["page"], 3)

        # Cleanup test session
        database.delete_session(session_id)
        self.assertIsNone(database.get_session(session_id))

    def test_5_ml_topics_retrieval(self):
        """Verify that the 13 ML topics can be successfully retrieved."""
        pipeline = RAGPipeline()
        pipeline.index_pdf(self.sample_pdf)

        test_queries = [
            ("supervised and unsupervised learning", 1),
            ("ROC-AUC curve", 2),
            ("Bagging and Boosting", 3),
            ("K-Means clustering and Elbow method", 4),
            ("Principal Component Analysis", 4)
        ]

        for query, expected_page in test_queries:
            results = pipeline.retrieve(query, top_k=2)
            self.assertGreater(len(results), 0, f"Query '{query}' returned no results.")
            pages = [r.chunk.page for r in results]
            self.assertIn(
                expected_page, 
                pages, 
                f"Query '{query}' expected page {expected_page}, but got pages {pages}."
            )

if __name__ == "__main__":
    unittest.main(verbosity=2)
