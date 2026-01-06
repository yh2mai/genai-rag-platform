"""
Tests for Uncertainty Handling functionality in RAG Pipeline
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
from typing import List

from app.services.rag_pipeline import (
    UncertaintyHandler,
    ConfidenceMetrics,
    UncertaintyResponse,
    RankedChunk,
    ContextWindow,
    RAGEngine
)


class TestUncertaintyHandler:
    """Test uncertainty handling functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.uncertainty_handler = UncertaintyHandler(
            min_confidence_threshold=0.6,
            min_context_sufficiency=0.5,
            max_hallucination_risk=0.3
        )
        
        # Create sample ranked chunks with varying quality
        self.high_quality_chunks = [
            RankedChunk(
                chunk_id="chunk_1",
                document_id="doc_1",
                content="Machine learning algorithms are computational methods that enable systems to learn patterns from data without explicit programming.",
                page_number=1,
                vector_score=0.9,
                bm25_score=0.8,
                combined_score=0.85,
                relevance_score=0.9,
                final_score=0.88,
                metadata={"filename": "ml_guide.pdf"}
            ),
            RankedChunk(
                chunk_id="chunk_2",
                document_id="doc_2",
                content="Deep learning is a subset of machine learning that uses neural networks with multiple layers to model complex patterns.",
                page_number=1,
                vector_score=0.85,
                bm25_score=0.75,
                combined_score=0.8,
                relevance_score=0.85,
                final_score=0.83,
                metadata={"filename": "deep_learning.pdf"}
            )
        ]
        
        self.low_quality_chunks = [
            RankedChunk(
                chunk_id="chunk_3",
                document_id="doc_3",
                content="The weather today is sunny.",
                page_number=1,
                vector_score=0.2,
                bm25_score=0.1,
                combined_score=0.15,
                relevance_score=0.2,
                final_score=0.18,
                metadata={"filename": "weather.pdf"}
            )
        ]
    
    def test_uncertainty_handler_initialization(self):
        """Test uncertainty handler initialization with custom thresholds"""
        handler = UncertaintyHandler(
            min_confidence_threshold=0.7,
            min_context_sufficiency=0.6,
            max_hallucination_risk=0.2
        )
        
        assert handler.min_confidence_threshold == 0.7
        assert handler.min_context_sufficiency == 0.6
        assert handler.max_hallucination_risk == 0.2
    
    def test_calculate_confidence_metrics_empty_chunks(self):
        """Test confidence calculation with no chunks"""
        metrics = self.uncertainty_handler.calculate_confidence_metrics(
            [], "test query"
        )
        
        assert metrics.retrieval_confidence == 0.0
        assert metrics.context_sufficiency == 0.0
        assert metrics.hallucination_risk == 1.0
        assert metrics.overall_confidence == 0.0
        assert "No relevant documents found" in metrics.uncertainty_indicators
    
    def test_calculate_confidence_metrics_high_quality(self):
        """Test confidence calculation with high quality chunks"""
        query = "machine learning algorithms"
        metrics = self.uncertainty_handler.calculate_confidence_metrics(
            self.high_quality_chunks, query
        )
        
        # High quality chunks should have good confidence scores
        assert metrics.retrieval_confidence > 0.7
        assert metrics.context_sufficiency > 0.5
        assert metrics.hallucination_risk < 0.5
        assert metrics.overall_confidence > 0.6
        assert len(metrics.uncertainty_indicators) <= 2  # Should have few indicators
    
    def test_calculate_confidence_metrics_low_quality(self):
        """Test confidence calculation with low quality chunks"""
        query = "machine learning algorithms"
        metrics = self.uncertainty_handler.calculate_confidence_metrics(
            self.low_quality_chunks, query
        )
        
        # Low quality chunks should have poor confidence scores
        assert metrics.retrieval_confidence < 0.5
        assert metrics.context_sufficiency < 0.5
        assert metrics.hallucination_risk > 0.5
        assert metrics.overall_confidence < 0.4
        assert len(metrics.uncertainty_indicators) > 0
    
    def test_calculate_retrieval_confidence(self):
        """Test retrieval confidence calculation"""
        # Test with high scores
        confidence = self.uncertainty_handler._calculate_retrieval_confidence(
            self.high_quality_chunks
        )
        assert 0.7 <= confidence <= 1.0
        
        # Test with low scores
        confidence = self.uncertainty_handler._calculate_retrieval_confidence(
            self.low_quality_chunks
        )
        assert 0.0 <= confidence <= 0.5
        
        # Test with empty chunks
        confidence = self.uncertainty_handler._calculate_retrieval_confidence([])
        assert confidence == 0.0
    
    def test_calculate_context_sufficiency(self):
        """Test context sufficiency calculation"""
        query = "machine learning algorithms"
        
        # Test with relevant content
        sufficiency = self.uncertainty_handler._calculate_context_sufficiency(
            self.high_quality_chunks, query
        )
        assert sufficiency > 0.5
        
        # Test with irrelevant content
        sufficiency = self.uncertainty_handler._calculate_context_sufficiency(
            self.low_quality_chunks, query
        )
        assert sufficiency < 0.5
    
    def test_calculate_hallucination_risk(self):
        """Test hallucination risk calculation"""
        query = "machine learning algorithms"
        
        # Test with relevant chunks (low risk)
        risk = self.uncertainty_handler._calculate_hallucination_risk(
            self.high_quality_chunks, query
        )
        assert risk < 0.5
        
        # Test with irrelevant chunks (high risk)
        risk = self.uncertainty_handler._calculate_hallucination_risk(
            self.low_quality_chunks, query
        )
        assert risk > 0.5
        
        # Test with no chunks (maximum risk)
        risk = self.uncertainty_handler._calculate_hallucination_risk([], query)
        assert risk == 1.0
    
    def test_identify_uncertainty_indicators(self):
        """Test uncertainty indicator identification"""
        # Test with low confidence scores
        indicators = self.uncertainty_handler._identify_uncertainty_indicators(
            retrieval_confidence=0.3,
            context_sufficiency=0.2,
            hallucination_risk=0.8,
            ranked_chunks=self.low_quality_chunks
        )
        
        assert len(indicators) > 0
        assert any("Low retrieval confidence" in indicator for indicator in indicators)
        assert any("Insufficient context" in indicator for indicator in indicators)
        assert any("High hallucination risk" in indicator for indicator in indicators)
    
    def test_generate_uncertainty_response_confident(self):
        """Test uncertainty response generation for confident results"""
        confident_metrics = ConfidenceMetrics(
            retrieval_confidence=0.8,
            context_sufficiency=0.7,
            hallucination_risk=0.2,
            overall_confidence=0.75,
            uncertainty_indicators=[]
        )
        
        response = self.uncertainty_handler.generate_uncertainty_response(
            confident_metrics, "test query", self.high_quality_chunks
        )
        
        assert response.response_type == "confident"
        assert response.confidence_score == 0.75
        assert len(response.fallback_suggestions) == 0
        assert "relevant information" in response.message.lower()
    
    def test_generate_uncertainty_response_insufficient_context(self):
        """Test uncertainty response generation for insufficient context"""
        insufficient_metrics = ConfidenceMetrics(
            retrieval_confidence=0.5,
            context_sufficiency=0.3,  # Below threshold
            hallucination_risk=0.4,
            overall_confidence=0.4,
            uncertainty_indicators=["Insufficient context"]
        )
        
        response = self.uncertainty_handler.generate_uncertainty_response(
            insufficient_metrics, "test query", self.low_quality_chunks
        )
        
        assert response.response_type == "insufficient_context"
        assert response.confidence_score == 0.4
        assert len(response.fallback_suggestions) > 0
        assert "not be sufficient" in response.message
    
    def test_generate_uncertainty_response_uncertain(self):
        """Test uncertainty response generation for uncertain results"""
        uncertain_metrics = ConfidenceMetrics(
            retrieval_confidence=0.5,
            context_sufficiency=0.6,  # Above threshold
            hallucination_risk=0.5,
            overall_confidence=0.5,  # Below confidence threshold
            uncertainty_indicators=["Moderate uncertainty"]
        )
        
        response = self.uncertainty_handler.generate_uncertainty_response(
            uncertain_metrics, "test query", self.high_quality_chunks
        )
        
        assert response.response_type == "uncertain"
        assert response.confidence_score == 0.5
        assert len(response.fallback_suggestions) > 0
        assert "not fully confident" in response.message
    
    def test_generate_fallback_suggestions(self):
        """Test fallback suggestion generation"""
        suggestions = self.uncertainty_handler._generate_fallback_suggestions(
            "test query", self.high_quality_chunks
        )
        
        assert len(suggestions) <= 3  # Should limit to top 3
        assert len(suggestions) > 0
        assert any("rephrasing" in suggestion.lower() for suggestion in suggestions)
    
    def test_should_provide_response(self):
        """Test response provision decision"""
        # High confidence - should provide response
        high_confidence = ConfidenceMetrics(0.8, 0.7, 0.2, 0.75, [])
        assert self.uncertainty_handler.should_provide_response(high_confidence) == True
        
        # Low confidence but acceptable risk - should provide response
        medium_confidence = ConfidenceMetrics(0.4, 0.4, 0.5, 0.4, [])
        assert self.uncertainty_handler.should_provide_response(medium_confidence) == True
        
        # Very low confidence - should not provide response
        low_confidence = ConfidenceMetrics(0.1, 0.1, 0.9, 0.1, [])
        assert self.uncertainty_handler.should_provide_response(low_confidence) == False
        
        # High hallucination risk - should not provide response
        high_risk = ConfidenceMetrics(0.5, 0.5, 0.9, 0.4, [])
        assert self.uncertainty_handler.should_provide_response(high_risk) == False
    
    def test_get_confidence_explanation(self):
        """Test confidence explanation generation"""
        metrics = ConfidenceMetrics(0.8, 0.8, 0.3, 0.7, [])  # Changed context_sufficiency to 0.8
        explanation = self.uncertainty_handler.get_confidence_explanation(metrics)
        
        assert "Strong retrieval match" in explanation
        assert "sufficient context" in explanation
        assert "low hallucination risk" in explanation


class TestRAGEngineUncertaintyIntegration:
    """Test RAG Engine integration with uncertainty handling"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Mock the dependencies to avoid loading actual models
        with patch('app.services.rag_pipeline.HybridRetriever'), \
             patch('app.services.rag_pipeline.CrossEncoderReranker'), \
             patch('app.services.rag_pipeline.ContextOptimizer'):
            self.rag_engine = RAGEngine()
    
    def test_rag_engine_initialization_with_uncertainty_handler(self):
        """Test RAG engine initialization includes uncertainty handler"""
        with patch('app.services.rag_pipeline.HybridRetriever'), \
             patch('app.services.rag_pipeline.CrossEncoderReranker'), \
             patch('app.services.rag_pipeline.ContextOptimizer'), \
             patch('app.services.rag_pipeline.UncertaintyHandler'):
            engine = RAGEngine()
            assert hasattr(engine, 'uncertainty_handler')
    
    def test_calculate_confidence(self):
        """Test confidence calculation method"""
        mock_uncertainty_handler = Mock()
        mock_metrics = Mock(spec=ConfidenceMetrics)
        mock_uncertainty_handler.calculate_confidence_metrics.return_value = mock_metrics
        
        self.rag_engine.uncertainty_handler = mock_uncertainty_handler
        
        chunks = [Mock()]
        query = "test query"
        context_window = Mock()
        
        result = self.rag_engine.calculate_confidence(chunks, query, context_window)
        
        mock_uncertainty_handler.calculate_confidence_metrics.assert_called_once_with(
            chunks, query, context_window
        )
        assert result == mock_metrics
    
    def test_handle_uncertainty(self):
        """Test uncertainty handling method"""
        mock_uncertainty_handler = Mock()
        mock_response = Mock(spec=UncertaintyResponse)
        mock_uncertainty_handler.generate_uncertainty_response.return_value = mock_response
        
        self.rag_engine.uncertainty_handler = mock_uncertainty_handler
        
        metrics = Mock()
        query = "test query"
        chunks = [Mock()]
        
        result = self.rag_engine.handle_uncertainty(metrics, query, chunks)
        
        mock_uncertainty_handler.generate_uncertainty_response.assert_called_once_with(
            metrics, query, chunks
        )
        assert result == mock_response
    
    def test_should_provide_response(self):
        """Test response provision decision method"""
        mock_uncertainty_handler = Mock()
        mock_uncertainty_handler.should_provide_response.return_value = True
        
        self.rag_engine.uncertainty_handler = mock_uncertainty_handler
        
        metrics = Mock()
        result = self.rag_engine.should_provide_response(metrics)
        
        mock_uncertainty_handler.should_provide_response.assert_called_once_with(metrics)
        assert result == True
    
    def test_process_query_with_uncertainty_empty_query(self):
        """Test uncertainty processing with empty query"""
        result = self.rag_engine.process_query_with_uncertainty("")
        
        assert result["query"] == ""
        assert result["should_respond"] == False
        assert "Empty query" in result["error"]
        assert result["confidence_metrics"].overall_confidence == 0.0
        assert result["uncertainty_response"].response_type == "insufficient_context"
    
    def test_process_query_with_uncertainty_success(self):
        """Test successful uncertainty processing"""
        # Mock the retrieval pipeline
        mock_chunks = [Mock(spec=RankedChunk)]
        self.rag_engine.retrieve_and_rerank = Mock(return_value=mock_chunks)
        
        # Mock context optimization
        mock_context_window = Mock(spec=ContextWindow)
        mock_context_window.selected_chunks = mock_chunks
        self.rag_engine.optimize_context = Mock(return_value=mock_context_window)
        
        # Mock uncertainty handling
        mock_confidence = Mock(spec=ConfidenceMetrics)
        mock_confidence.overall_confidence = 0.8
        mock_uncertainty_response = Mock(spec=UncertaintyResponse)
        mock_uncertainty_response.response_type = "confident"
        
        self.rag_engine.calculate_confidence = Mock(return_value=mock_confidence)
        self.rag_engine.handle_uncertainty = Mock(return_value=mock_uncertainty_response)
        self.rag_engine.should_provide_response = Mock(return_value=True)
        
        # Mock context quality analysis
        self.rag_engine.context_optimizer.analyze_context_quality = Mock(return_value={})
        
        result = self.rag_engine.process_query_with_uncertainty("test query")
        
        assert result["query"] == "test query"
        assert result["should_respond"] == True
        assert result["confidence_metrics"] == mock_confidence
        assert result["uncertainty_response"] == mock_uncertainty_response
        assert "error" not in result
    
    def test_process_query_with_uncertainty_no_chunks(self):
        """Test uncertainty processing when no chunks are retrieved"""
        # Mock empty retrieval
        self.rag_engine.retrieve_and_rerank = Mock(return_value=[])
        
        # Mock empty context window
        empty_context = ContextWindow([], "", 0, 0.0, [])
        self.rag_engine.optimize_context = Mock(return_value=empty_context)
        
        # Mock low confidence
        mock_confidence = Mock(spec=ConfidenceMetrics)
        mock_confidence.overall_confidence = 0.0
        mock_uncertainty_response = Mock(spec=UncertaintyResponse)
        mock_uncertainty_response.response_type = "insufficient_context"
        
        self.rag_engine.calculate_confidence = Mock(return_value=mock_confidence)
        self.rag_engine.handle_uncertainty = Mock(return_value=mock_uncertainty_response)
        self.rag_engine.should_provide_response = Mock(return_value=False)
        
        result = self.rag_engine.process_query_with_uncertainty("test query")
        
        assert result["should_respond"] == False
        assert result["total_retrieved"] == 0
        assert result["selected_chunks"] == 0


if __name__ == "__main__":
    pytest.main([__file__])