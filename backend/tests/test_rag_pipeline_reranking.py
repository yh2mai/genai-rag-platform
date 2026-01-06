"""
Tests for RAG Pipeline Re-ranking functionality
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
from typing import List

from app.services.rag_pipeline import (
    RAGEngine, 
    CrossEncoderReranker, 
    ScoredChunk, 
    RankedChunk,
    HybridRetriever
)
from app.services.vector_store import SearchResult


class TestCrossEncoderReranker:
    """Test cross-encoder re-ranking functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.reranker = CrossEncoderReranker()
        
        # Create sample scored chunks
        self.sample_chunks = [
            ScoredChunk(
                chunk_id="chunk_1",
                document_id="doc_1",
                content="This is about machine learning algorithms and neural networks.",
                page_number=1,
                vector_score=0.8,
                bm25_score=0.6,
                combined_score=0.7,
                metadata={"filename": "ml_guide.pdf"}
            ),
            ScoredChunk(
                chunk_id="chunk_2", 
                document_id="doc_1",
                content="The weather today is sunny with a chance of rain.",
                page_number=2,
                vector_score=0.3,
                bm25_score=0.4,
                combined_score=0.35,
                metadata={"filename": "ml_guide.pdf"}
            ),
            ScoredChunk(
                chunk_id="chunk_3",
                document_id="doc_2", 
                content="Deep learning models require large datasets for training.",
                page_number=1,
                vector_score=0.9,
                bm25_score=0.7,
                combined_score=0.8,
                metadata={"filename": "deep_learning.pdf"}
            )
        ]
    
    def test_reranker_initialization(self):
        """Test cross-encoder reranker initialization"""
        reranker = CrossEncoderReranker()
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        # Model should be loaded if sentence_transformers is available
        if hasattr(reranker, 'model') and reranker.model is not None:
            assert reranker.model is not None
    
    def test_rerank_chunks_empty_input(self):
        """Test re-ranking with empty chunks"""
        result = self.reranker.rerank_chunks("test query", [])
        assert result == []
    
    def test_rerank_chunks_empty_query(self):
        """Test re-ranking with empty query"""
        result = self.reranker.rerank_chunks("", self.sample_chunks)
        assert len(result) == len(self.sample_chunks)
        # Should use fallback ranking
        for chunk in result:
            assert isinstance(chunk, RankedChunk)
            assert chunk.relevance_score == chunk.combined_score
    
    def test_rerank_chunks_with_query(self):
        """Test re-ranking with valid query and chunks"""
        query = "machine learning algorithms"
        result = self.reranker.rerank_chunks(query, self.sample_chunks)
        
        assert len(result) == len(self.sample_chunks)
        assert all(isinstance(chunk, RankedChunk) for chunk in result)
        
        # Results should be sorted by final_score in descending order
        scores = [chunk.final_score for chunk in result]
        assert scores == sorted(scores, reverse=True)
        
        # Check that all required fields are present
        for chunk in result:
            assert hasattr(chunk, 'relevance_score')
            assert hasattr(chunk, 'final_score')
            assert chunk.chunk_id in ['chunk_1', 'chunk_2', 'chunk_3']
    
    def test_rerank_chunks_with_top_k(self):
        """Test re-ranking with top_k limit"""
        query = "machine learning"
        top_k = 2
        result = self.reranker.rerank_chunks(query, self.sample_chunks, top_k=top_k)
        
        assert len(result) == top_k
        assert all(isinstance(chunk, RankedChunk) for chunk in result)
    
    def test_fallback_ranking(self):
        """Test fallback ranking when cross-encoder is not available"""
        result = self.reranker._fallback_ranking(self.sample_chunks, top_k=2)
        
        assert len(result) == 2
        assert all(isinstance(chunk, RankedChunk) for chunk in result)
        
        # Should be sorted by combined_score
        assert result[0].final_score >= result[1].final_score
        
        # Relevance score should equal combined score in fallback
        for chunk in result:
            assert chunk.relevance_score == chunk.combined_score
    
    def test_get_relevance_threshold(self):
        """Test relevance threshold calculation"""
        # Create ranked chunks with known scores
        ranked_chunks = [
            RankedChunk(
                chunk_id="chunk_1", document_id="doc_1", content="test",
                page_number=1, vector_score=0.8, bm25_score=0.6,
                combined_score=0.7, relevance_score=0.9, final_score=0.85,
                metadata={}
            ),
            RankedChunk(
                chunk_id="chunk_2", document_id="doc_1", content="test",
                page_number=2, vector_score=0.6, bm25_score=0.4,
                combined_score=0.5, relevance_score=0.7, final_score=0.65,
                metadata={}
            ),
            RankedChunk(
                chunk_id="chunk_3", document_id="doc_1", content="test",
                page_number=3, vector_score=0.4, bm25_score=0.2,
                combined_score=0.3, relevance_score=0.5, final_score=0.45,
                metadata={}
            )
        ]
        
        # Test median threshold
        threshold = self.reranker.get_relevance_threshold(ranked_chunks, percentile=0.5)
        assert threshold == 0.7  # Middle value
        
        # Test empty list
        threshold = self.reranker.get_relevance_threshold([], percentile=0.5)
        assert threshold == 0.0


class TestRAGEngineReranking:
    """Test RAG Engine with re-ranking functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Mock the dependencies to avoid loading actual models
        with patch('app.services.rag_pipeline.HybridRetriever'), \
             patch('app.services.rag_pipeline.CrossEncoderReranker'):
            self.rag_engine = RAGEngine()
    
    def test_rag_engine_initialization_with_reranking(self):
        """Test RAG engine initialization with re-ranking components"""
        with patch('app.services.rag_pipeline.HybridRetriever'), \
             patch('app.services.rag_pipeline.CrossEncoderReranker'):
            engine = RAGEngine()
            assert hasattr(engine, 'reranker')
            assert hasattr(engine, 'hybrid_retriever')
    
    def test_rerank_results(self):
        """Test re-ranking results method"""
        # Mock the reranker
        mock_reranker = Mock()
        mock_ranked_chunks = [Mock(spec=RankedChunk)]
        mock_reranker.rerank_chunks.return_value = mock_ranked_chunks
        
        self.rag_engine.reranker = mock_reranker
        
        query = "test query"
        chunks = [Mock(spec=ScoredChunk)]
        
        result = self.rag_engine.rerank_results(query, chunks, top_k=10)
        
        mock_reranker.rerank_chunks.assert_called_once_with(query, chunks, 10)
        assert result == mock_ranked_chunks
    
    def test_retrieve_and_rerank_pipeline(self):
        """Test complete retrieve and rerank pipeline"""
        # Mock hybrid retriever
        mock_scored_chunks = [Mock(spec=ScoredChunk)]
        mock_hybrid_retriever = Mock()
        mock_hybrid_retriever.hybrid_retrieve.return_value = mock_scored_chunks
        
        # Mock reranker
        mock_ranked_chunks = [Mock(spec=RankedChunk)]
        mock_reranker = Mock()
        mock_reranker.rerank_chunks.return_value = mock_ranked_chunks
        
        self.rag_engine.hybrid_retriever = mock_hybrid_retriever
        self.rag_engine.reranker = mock_reranker
        
        query = "test query"
        result = self.rag_engine.retrieve_and_rerank(
            query=query,
            retrieval_k=50,
            final_k=20
        )
        
        # Verify hybrid retrieval was called
        mock_hybrid_retriever.hybrid_retrieve.assert_called_once_with(
            query=query,
            top_k=50
        )
        
        # Verify re-ranking was called
        mock_reranker.rerank_chunks.assert_called_once_with(
            query,
            mock_scored_chunks,
            20
        )
        
        assert result == mock_ranked_chunks
    
    def test_retrieve_and_rerank_empty_query(self):
        """Test pipeline with empty query"""
        result = self.rag_engine.retrieve_and_rerank("")
        assert result == []
    
    def test_retrieve_and_rerank_no_chunks(self):
        """Test pipeline when no chunks are retrieved"""
        mock_hybrid_retriever = Mock()
        mock_hybrid_retriever.hybrid_retrieve.return_value = []
        self.rag_engine.hybrid_retriever = mock_hybrid_retriever
        
        result = self.rag_engine.retrieve_and_rerank("test query")
        assert result == []
    
    def test_optimize_context(self):
        """Test context optimization with token limits"""
        # Create sample ranked chunks
        ranked_chunks = [
            RankedChunk(
                chunk_id="chunk_1", document_id="doc_1", 
                content="A" * 1000,  # 1000 characters
                page_number=1, vector_score=0.8, bm25_score=0.6,
                combined_score=0.7, relevance_score=0.9, final_score=0.85,
                metadata={}
            ),
            RankedChunk(
                chunk_id="chunk_2", document_id="doc_1",
                content="B" * 2000,  # 2000 characters
                page_number=2, vector_score=0.6, bm25_score=0.4,
                combined_score=0.5, relevance_score=0.7, final_score=0.65,
                metadata={}
            ),
            RankedChunk(
                chunk_id="chunk_3", document_id="doc_1",
                content="C" * 3000,  # 3000 characters
                page_number=3, vector_score=0.4, bm25_score=0.2,
                combined_score=0.3, relevance_score=0.5, final_score=0.45,
                metadata={}
            )
        ]
        
        # Test with token limit that should include first two chunks
        context_window = self.rag_engine.optimize_context(
            ranked_chunks, 
            max_tokens=1000  # ~4000 characters
        )
        
        # Should select chunks that fit within limit
        assert len(context_window.selected_chunks) >= 1
        assert len(context_window.combined_context) > 0
        assert context_window.total_tokens > 0
        assert 0 <= context_window.utilization_ratio <= 1.0
        
        # Context should contain document and page information
        assert "Source" in context_window.combined_context
        assert "Page" in context_window.combined_context
    
    def test_optimize_context_empty_chunks(self):
        """Test context optimization with empty chunks"""
        context_window = self.rag_engine.optimize_context([])
        assert context_window.selected_chunks == []
        assert context_window.combined_context == ""
        assert context_window.total_tokens == 0
        assert context_window.utilization_ratio == 0.0
    
    def test_generate_citations(self):
        """Test citation generation"""
        ranked_chunks = [
            RankedChunk(
                chunk_id="chunk_1", document_id="doc_1",
                content="This is a test content for citation.",
                page_number=1, vector_score=0.8, bm25_score=0.6,
                combined_score=0.7, relevance_score=0.9, final_score=0.85,
                metadata={"filename": "test.pdf"}
            )
        ]
        
        citations = self.rag_engine.generate_citations(ranked_chunks)
        
        assert len(citations) == 1
        citation = citations[0]
        
        assert citation.document_id == "doc_1"
        assert citation.page_number == 1
        assert citation.relevance_score == 0.9
        assert citation.filename == "test.pdf"
        assert len(citation.chunk_content) > 0
    
    def test_retrieve_with_reranking_enabled(self):
        """Test main retrieve method with re-ranking enabled"""
        mock_ranked_chunks = [Mock(spec=RankedChunk)]
        
        # Mock the retrieve_and_rerank method
        self.rag_engine.retrieve_and_rerank = Mock(return_value=mock_ranked_chunks)
        
        result = self.rag_engine.retrieve("test query", top_k=10, use_reranking=True)
        
        self.rag_engine.retrieve_and_rerank.assert_called_once()
        assert result == mock_ranked_chunks
    
    def test_retrieve_with_reranking_disabled(self):
        """Test main retrieve method with re-ranking disabled"""
        # Mock hybrid retrieval
        mock_scored_chunks = [
            ScoredChunk(
                chunk_id="chunk_1", document_id="doc_1", content="test",
                page_number=1, vector_score=0.8, bm25_score=0.6,
                combined_score=0.7, metadata={}
            )
        ]
        
        mock_hybrid_retriever = Mock()
        mock_hybrid_retriever.hybrid_retrieve.return_value = mock_scored_chunks
        self.rag_engine.hybrid_retriever = mock_hybrid_retriever
        
        result = self.rag_engine.retrieve("test query", top_k=10, use_reranking=False)
        
        assert len(result) == 1
        assert isinstance(result[0], RankedChunk)
        # Should convert ScoredChunk to RankedChunk
        assert result[0].relevance_score == result[0].combined_score
        assert result[0].final_score == result[0].combined_score


if __name__ == "__main__":
    pytest.main([__file__])