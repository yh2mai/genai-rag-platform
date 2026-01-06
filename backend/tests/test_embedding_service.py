"""
Tests for embedding service functionality
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch

from app.services.embedding_service import EmbeddingService
from app.services.semantic_chunker import Chunk


class TestEmbeddingService:
    """Test cases for EmbeddingService class"""
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing"""
        chunks = []
        for i in range(3):
            chunk = Chunk(
                chunk_id=f"doc1_chunk_{i:04d}",
                document_id="doc1",
                content=f"This is sample content for chunk {i}. It contains meaningful text.",
                page_number=1,
                start_char=i * 100,
                end_char=(i + 1) * 100,
                chunk_index=i
            )
            chunks.append(chunk)
        return chunks
    
    @patch('app.services.embedding_service.SentenceTransformer')
    def test_embedding_service_initialization(self, mock_sentence_transformer):
        """Test embedding service initialization"""
        service = EmbeddingService(model_name="test-model")
        
        assert service.model_name == "test-model"
        assert service.model is None  # Model not loaded yet
        assert service.dimension == 384  # From settings
    
    @patch('app.services.embedding_service.SentenceTransformer')
    def test_model_loading(self, mock_sentence_transformer):
        """Test model loading functionality"""
        # Mock the sentence transformer
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3] * 128])  # 384 dimensions
        mock_sentence_transformer.return_value = mock_model
        
        service = EmbeddingService(model_name="test-model")
        service._load_model()
        
        # Verify model was loaded
        assert service.model is not None
        mock_sentence_transformer.assert_called_once_with("test-model")
    
    @patch('app.services.embedding_service.SentenceTransformer')
    def test_generate_embeddings(self, mock_sentence_transformer, sample_chunks):
        """Test embedding generation for chunks"""
        # Mock the sentence transformer
        mock_model = Mock()
        # Create mock embeddings for each chunk
        mock_embeddings = np.random.rand(len(sample_chunks), 384).astype(np.float32)
        mock_model.encode.return_value = mock_embeddings
        mock_sentence_transformer.return_value = mock_model
        
        service = EmbeddingService(model_name="test-model")
        embedded_chunks = service.generate_embeddings(sample_chunks)
        
        # Verify results
        assert len(embedded_chunks) == len(sample_chunks)
        
        for i, (original_chunk, embedded_chunk) in enumerate(zip(sample_chunks, embedded_chunks)):
            assert embedded_chunk.chunk_id == original_chunk.chunk_id
            assert embedded_chunk.document_id == original_chunk.document_id
            assert embedded_chunk.content == original_chunk.content
            assert embedded_chunk.page_number == original_chunk.page_number
            assert embedded_chunk.embedding_model == "test-model"
            assert embedded_chunk.embedding.shape == (384,)
            np.testing.assert_array_equal(embedded_chunk.embedding, mock_embeddings[i])
    
    @patch('app.services.embedding_service.SentenceTransformer')
    def test_generate_query_embedding(self, mock_sentence_transformer):
        """Test query embedding generation"""
        # Mock the sentence transformer
        mock_model = Mock()
        # First call is for dimension check, second is for actual query
        mock_model.encode.side_effect = [
            np.random.rand(1, 384).astype(np.float32),  # Dimension check
            np.random.rand(1, 384).astype(np.float32)   # Actual query
        ]
        mock_sentence_transformer.return_value = mock_model
        
        service = EmbeddingService(model_name="test-model")
        query = "What is the meaning of life?"
        
        result = service.generate_query_embedding(query)
        
        # Verify result
        assert result.shape == (384,)
        
        # Verify model was called twice (dimension check + actual query)
        assert mock_model.encode.call_count == 2
    
    def test_generate_query_embedding_empty_query(self):
        """Test query embedding with empty query"""
        service = EmbeddingService(model_name="test-model")
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            service.generate_query_embedding("")
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            service.generate_query_embedding("   ")
    
    def test_generate_embeddings_empty_chunks(self):
        """Test embedding generation with empty chunk list"""
        service = EmbeddingService(model_name="test-model")
        
        result = service.generate_embeddings([])
        assert result == []
    
    @patch('app.services.embedding_service.SentenceTransformer')
    def test_dimension_mismatch_handling(self, mock_sentence_transformer):
        """Test handling of dimension mismatch between config and model"""
        # Mock model that returns different dimension
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2]])  # 2 dimensions instead of 384
        mock_sentence_transformer.return_value = mock_model
        
        service = EmbeddingService(model_name="test-model")
        service._load_model()
        
        # Service should update dimension to match actual model
        assert service.dimension == 2
    
    def test_get_dimension(self):
        """Test getting embedding dimension"""
        service = EmbeddingService(model_name="test-model")
        assert service.get_dimension() == 384
    
    def test_get_model_name(self):
        """Test getting model name"""
        service = EmbeddingService(model_name="test-model")
        assert service.get_model_name() == "test-model"