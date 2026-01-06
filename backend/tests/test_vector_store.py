"""
Tests for FAISS vector store functionality
"""
import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from app.services.vector_store import VectorStore, EmbeddedChunk, SearchResult
from app.services.semantic_chunker import Chunk


class TestVectorStore:
    """Test cases for VectorStore class"""
    
    @pytest.fixture
    def temp_storage_path(self):
        """Create temporary storage directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def vector_store(self, temp_storage_path):
        """Create vector store instance for testing"""
        return VectorStore(
            dimension=384,
            index_type="flat",
            storage_path=temp_storage_path
        )
    
    @pytest.fixture
    def sample_embedded_chunks(self):
        """Create sample embedded chunks for testing"""
        chunks = []
        for i in range(5):
            embedding = np.random.rand(384).astype(np.float32)
            chunk = EmbeddedChunk(
                chunk_id=f"doc1_chunk_{i:04d}",
                document_id="doc1",
                content=f"This is sample content for chunk {i}",
                page_number=1,
                start_char=i * 100,
                end_char=(i + 1) * 100,
                chunk_index=i,
                embedding=embedding,
                embedding_model="test-model"
            )
            chunks.append(chunk)
        return chunks
    
    def test_vector_store_initialization(self, temp_storage_path):
        """Test vector store initialization"""
        store = VectorStore(
            dimension=384,
            index_type="flat",
            storage_path=temp_storage_path
        )
        
        assert store.dimension == 384
        assert store.index_type == "flat"
        assert store.storage_path == temp_storage_path
        assert store.index.ntotal == 0
        assert len(store.chunk_metadata) == 0
    
    def test_add_embeddings(self, vector_store, sample_embedded_chunks):
        """Test adding embeddings to vector store"""
        # Initially empty
        assert vector_store.index.ntotal == 0
        
        # Add embeddings
        vector_store.add_embeddings(sample_embedded_chunks)
        
        # Verify embeddings were added
        assert vector_store.index.ntotal == len(sample_embedded_chunks)
        assert len(vector_store.chunk_metadata) == len(sample_embedded_chunks)
        
        # Verify metadata is stored correctly
        for i, chunk in enumerate(sample_embedded_chunks):
            assert chunk.chunk_id in vector_store.chunk_id_to_index
            idx = vector_store.chunk_id_to_index[chunk.chunk_id]
            metadata = vector_store.chunk_metadata[idx]
            assert metadata['chunk_id'] == chunk.chunk_id
            assert metadata['document_id'] == chunk.document_id
            assert metadata['content'] == chunk.content
    
    def test_similarity_search(self, vector_store, sample_embedded_chunks):
        """Test similarity search functionality"""
        # Add embeddings
        vector_store.add_embeddings(sample_embedded_chunks)
        
        # Create query embedding (similar to first chunk)
        query_embedding = sample_embedded_chunks[0].embedding + np.random.rand(384) * 0.1
        
        # Perform search
        results = vector_store.similarity_search(query_embedding, k=3)
        
        # Verify results
        assert len(results) <= 3
        assert all(isinstance(result, SearchResult) for result in results)
        
        # Results should be sorted by similarity
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].similarity_score >= results[i + 1].similarity_score
    
    def test_get_document_chunks(self, vector_store, sample_embedded_chunks):
        """Test retrieving chunks by document ID"""
        # Add embeddings
        vector_store.add_embeddings(sample_embedded_chunks)
        
        # Get chunks for document
        results = vector_store.get_document_chunks("doc1")
        
        # Verify results
        assert len(results) == len(sample_embedded_chunks)
        assert all(result.document_id == "doc1" for result in results)
        
        # Should be sorted by chunk index
        for i in range(len(results) - 1):
            assert results[i].metadata['chunk_index'] <= results[i + 1].metadata['chunk_index']
    
    def test_get_chunk_by_id(self, vector_store, sample_embedded_chunks):
        """Test retrieving specific chunk by ID"""
        # Add embeddings
        vector_store.add_embeddings(sample_embedded_chunks)
        
        # Get specific chunk
        chunk_id = sample_embedded_chunks[0].chunk_id
        result = vector_store.get_chunk_by_id(chunk_id)
        
        # Verify result
        assert result is not None
        assert result.chunk_id == chunk_id
        assert result.document_id == sample_embedded_chunks[0].document_id
        assert result.content == sample_embedded_chunks[0].content
        
        # Test non-existent chunk
        result = vector_store.get_chunk_by_id("non_existent_chunk")
        assert result is None
    
    def test_remove_document(self, vector_store, sample_embedded_chunks):
        """Test removing document chunks"""
        # Add embeddings
        vector_store.add_embeddings(sample_embedded_chunks)
        initial_count = vector_store.index.ntotal
        
        # Remove document
        removed_count = vector_store.remove_document("doc1")
        
        # Verify removal
        assert removed_count == len(sample_embedded_chunks)
        
        # Chunks should not be found in metadata
        results = vector_store.get_document_chunks("doc1")
        assert len(results) == 0
        
        # Test removing non-existent document
        removed_count = vector_store.remove_document("non_existent_doc")
        assert removed_count == 0
    
    def test_get_stats(self, vector_store, sample_embedded_chunks):
        """Test getting vector store statistics"""
        # Initially empty
        stats = vector_store.get_stats()
        assert stats['total_vectors'] == 0
        assert stats['dimension'] == 384
        assert stats['index_type'] == "flat"
        assert stats['metadata_count'] == 0
        assert stats['unique_documents'] == 0
        
        # After adding embeddings
        vector_store.add_embeddings(sample_embedded_chunks)
        stats = vector_store.get_stats()
        assert stats['total_vectors'] == len(sample_embedded_chunks)
        assert stats['metadata_count'] == len(sample_embedded_chunks)
        assert stats['unique_documents'] == 1  # All chunks from same document
    
    def test_persistence(self, temp_storage_path, sample_embedded_chunks):
        """Test saving and loading vector store"""
        # Create and populate vector store
        store1 = VectorStore(
            dimension=384,
            index_type="flat",
            storage_path=temp_storage_path
        )
        store1.add_embeddings(sample_embedded_chunks)
        
        # Create new instance (should load existing data)
        store2 = VectorStore(
            dimension=384,
            index_type="flat",
            storage_path=temp_storage_path
        )
        
        # Verify data was loaded
        assert store2.index.ntotal == len(sample_embedded_chunks)
        assert len(store2.chunk_metadata) == len(sample_embedded_chunks)
        
        # Verify search works
        query_embedding = sample_embedded_chunks[0].embedding
        results = store2.similarity_search(query_embedding, k=1)
        assert len(results) > 0
    
    def test_clear(self, vector_store, sample_embedded_chunks):
        """Test clearing vector store"""
        # Add embeddings
        vector_store.add_embeddings(sample_embedded_chunks)
        assert vector_store.index.ntotal > 0
        
        # Clear store
        vector_store.clear()
        
        # Verify store is empty
        assert vector_store.index.ntotal == 0
        assert len(vector_store.chunk_metadata) == 0
        assert len(vector_store.chunk_id_to_index) == 0
        assert len(vector_store.index_to_chunk_id) == 0
    
    def test_empty_search(self, vector_store):
        """Test search on empty vector store"""
        query_embedding = np.random.rand(384).astype(np.float32)
        results = vector_store.similarity_search(query_embedding, k=5)
        assert len(results) == 0
    
    def test_score_threshold(self, vector_store, sample_embedded_chunks):
        """Test similarity search with score threshold"""
        # Add embeddings
        vector_store.add_embeddings(sample_embedded_chunks)
        
        # Search with high threshold (should return fewer results)
        query_embedding = np.random.rand(384).astype(np.float32)
        results_no_threshold = vector_store.similarity_search(query_embedding, k=5)
        results_with_threshold = vector_store.similarity_search(
            query_embedding, k=5, score_threshold=0.9
        )
        
        # With threshold should return fewer or equal results
        assert len(results_with_threshold) <= len(results_no_threshold)