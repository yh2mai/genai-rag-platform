"""
FAISS Vector Database Service for semantic search and embedding storage
"""
import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result with similarity score and metadata"""
    chunk_id: str
    document_id: str
    content: str
    page_number: int
    similarity_score: float
    metadata: Dict[str, Any]


@dataclass
class EmbeddedChunk:
    """Chunk with embedding vector"""
    chunk_id: str
    document_id: str
    content: str
    page_number: int
    start_char: int
    end_char: int
    chunk_index: int
    embedding: np.ndarray
    embedding_model: str


class VectorStore:
    """
    FAISS-based vector database for semantic search and embedding storage
    """
    
    def __init__(self, 
                 dimension: int = 384,  # Default for sentence-transformers/all-MiniLM-L6-v2
                 index_type: str = "flat",
                 storage_path: Optional[str] = None):
        """
        Initialize FAISS vector store
        
        Args:
            dimension: Embedding vector dimension
            index_type: FAISS index type ('flat', 'ivf', 'hnsw')
            storage_path: Path to store index and metadata files
        """
        self.dimension = dimension
        self.index_type = index_type
        self.storage_path = storage_path or settings.VECTOR_DB_PATH
        
        # Ensure storage directory exists
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize FAISS index
        self.index = self._create_index()
        
        # Metadata storage for chunk information
        self.chunk_metadata: Dict[int, Dict[str, Any]] = {}
        self.chunk_id_to_index: Dict[str, int] = {}
        self.index_to_chunk_id: Dict[int, str] = {}
        
        # Load existing index if available
        self._load_index()
        
        logger.info(f"Initialized FAISS vector store with {self.index.ntotal} vectors")
    
    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on configuration"""
        if self.index_type == "flat":
            # Exact search using L2 distance
            index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "ivf":
            # Inverted file index for faster approximate search
            quantizer = faiss.IndexFlatL2(self.dimension)
            nlist = 100  # Number of clusters
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        elif self.index_type == "hnsw":
            # Hierarchical Navigable Small World for fast approximate search
            index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        return index
    
    def add_embeddings(self, 
                      embedded_chunks: List[EmbeddedChunk]) -> None:
        """
        Add embeddings to the vector store
        
        Args:
            embedded_chunks: List of chunks with embeddings
        """
        if not embedded_chunks:
            logger.warning("No embedded chunks provided")
            return
        
        # Prepare embeddings matrix
        embeddings = np.array([chunk.embedding for chunk in embedded_chunks], dtype=np.float32)
        
        # Normalize embeddings for cosine similarity (if using L2 index)
        if self.index_type == "flat":
            faiss.normalize_L2(embeddings)
        
        # Get starting index for new embeddings
        start_idx = self.index.ntotal
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Store metadata
        for i, chunk in enumerate(embedded_chunks):
            idx = start_idx + i
            self.chunk_metadata[idx] = {
                'chunk_id': chunk.chunk_id,
                'document_id': chunk.document_id,
                'content': chunk.content,
                'page_number': chunk.page_number,
                'start_char': chunk.start_char,
                'end_char': chunk.end_char,
                'chunk_index': chunk.chunk_index,
                'embedding_model': chunk.embedding_model
            }
            self.chunk_id_to_index[chunk.chunk_id] = idx
            self.index_to_chunk_id[idx] = chunk.chunk_id
        
        logger.info(f"Added {len(embedded_chunks)} embeddings to vector store. Total: {self.index.ntotal}")
        
        # Save updated index
        self._save_index()
    
    def similarity_search(self, 
                         query_embedding: np.ndarray, 
                         k: int = 10,
                         score_threshold: Optional[float] = None) -> List[SearchResult]:
        """
        Perform similarity search using query embedding
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of search results with similarity scores
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []
        
        # Normalize query embedding for cosine similarity
        query_embedding = query_embedding.astype(np.float32)
        if self.index_type == "flat":
            query_embedding = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_embedding)
        else:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Perform search
        scores, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        # Convert to search results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
                
            # Convert L2 distance to similarity score (for flat index)
            if self.index_type == "flat":
                similarity_score = 1.0 / (1.0 + score)  # Convert distance to similarity
            else:
                similarity_score = score
            
            # Apply score threshold if specified
            if score_threshold is not None and similarity_score < score_threshold:
                continue
            
            # Get metadata
            metadata = self.chunk_metadata.get(idx, {})
            if not metadata:
                logger.warning(f"No metadata found for index {idx}")
                continue
            
            result = SearchResult(
                chunk_id=metadata['chunk_id'],
                document_id=metadata['document_id'],
                content=metadata['content'],
                page_number=metadata['page_number'],
                similarity_score=similarity_score,
                metadata=metadata
            )
            results.append(result)
        
        logger.info(f"Found {len(results)} similar chunks for query")
        return results
    
    def get_document_chunks(self, document_id: str) -> List[SearchResult]:
        """
        Retrieve all chunks for a specific document
        
        Args:
            document_id: Document identifier
            
        Returns:
            List of chunks for the document
        """
        results = []
        
        for idx, metadata in self.chunk_metadata.items():
            if metadata.get('document_id') == document_id:
                result = SearchResult(
                    chunk_id=metadata['chunk_id'],
                    document_id=metadata['document_id'],
                    content=metadata['content'],
                    page_number=metadata['page_number'],
                    similarity_score=1.0,  # No similarity calculation
                    metadata=metadata
                )
                results.append(result)
        
        # Sort by chunk index
        results.sort(key=lambda x: x.metadata.get('chunk_index', 0))
        
        logger.info(f"Retrieved {len(results)} chunks for document {document_id}")
        return results
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[SearchResult]:
        """
        Retrieve a specific chunk by ID
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Search result for the chunk or None if not found
        """
        idx = self.chunk_id_to_index.get(chunk_id)
        if idx is None:
            return None
        
        metadata = self.chunk_metadata.get(idx, {})
        if not metadata:
            return None
        
        return SearchResult(
            chunk_id=metadata['chunk_id'],
            document_id=metadata['document_id'],
            content=metadata['content'],
            page_number=metadata['page_number'],
            similarity_score=1.0,
            metadata=metadata
        )
    
    def remove_document(self, document_id: str) -> int:
        """
        Remove all chunks for a document from the vector store
        
        Args:
            document_id: Document identifier
            
        Returns:
            Number of chunks removed
        """
        # Find indices to remove
        indices_to_remove = []
        for idx, metadata in self.chunk_metadata.items():
            if metadata.get('document_id') == document_id:
                indices_to_remove.append(idx)
        
        if not indices_to_remove:
            logger.info(f"No chunks found for document {document_id}")
            return 0
        
        # FAISS doesn't support direct removal, so we need to rebuild the index
        # This is a limitation we'll note for future optimization
        logger.warning(f"Removing {len(indices_to_remove)} chunks requires index rebuild")
        
        # For now, we'll mark them as removed in metadata
        # In a production system, we'd implement periodic index rebuilding
        removed_count = 0
        for idx in indices_to_remove:
            if idx in self.chunk_metadata:
                chunk_id = self.chunk_metadata[idx]['chunk_id']
                del self.chunk_metadata[idx]
                del self.chunk_id_to_index[chunk_id]
                del self.index_to_chunk_id[idx]
                removed_count += 1
        
        # Save updated metadata
        self._save_metadata()
        
        logger.info(f"Marked {removed_count} chunks as removed for document {document_id}")
        return removed_count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics
        
        Returns:
            Dictionary with store statistics
        """
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'storage_path': self.storage_path,
            'metadata_count': len(self.chunk_metadata),
            'unique_documents': len(set(
                metadata.get('document_id', '') 
                for metadata in self.chunk_metadata.values()
            ))
        }
    
    def _save_index(self) -> None:
        """Save FAISS index and metadata to disk"""
        try:
            # Save FAISS index
            index_path = os.path.join(self.storage_path, "faiss_index.bin")
            faiss.write_index(self.index, index_path)
            
            # Save metadata
            self._save_metadata()
            
            logger.info(f"Saved vector store to {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            raise
    
    def _save_metadata(self) -> None:
        """Save metadata to disk"""
        metadata_path = os.path.join(self.storage_path, "metadata.pkl")
        metadata = {
            'chunk_metadata': self.chunk_metadata,
            'chunk_id_to_index': self.chunk_id_to_index,
            'index_to_chunk_id': self.index_to_chunk_id,
            'dimension': self.dimension,
            'index_type': self.index_type
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
    
    def _load_index(self) -> None:
        """Load FAISS index and metadata from disk"""
        index_path = os.path.join(self.storage_path, "faiss_index.bin")
        metadata_path = os.path.join(self.storage_path, "metadata.pkl")
        
        try:
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                # Load FAISS index
                self.index = faiss.read_index(index_path)
                
                # Load metadata
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                self.chunk_metadata = metadata.get('chunk_metadata', {})
                self.chunk_id_to_index = metadata.get('chunk_id_to_index', {})
                self.index_to_chunk_id = metadata.get('index_to_chunk_id', {})
                
                # Verify dimension consistency
                stored_dimension = metadata.get('dimension', self.dimension)
                if stored_dimension != self.dimension:
                    logger.warning(f"Dimension mismatch: expected {self.dimension}, got {stored_dimension}")
                
                logger.info(f"Loaded existing vector store with {self.index.ntotal} vectors")
            else:
                logger.info("No existing vector store found, starting fresh")
                
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            logger.info("Starting with fresh vector store")
            self.index = self._create_index()
            self.chunk_metadata = {}
            self.chunk_id_to_index = {}
            self.index_to_chunk_id = {}
    
    def clear(self) -> None:
        """Clear all data from the vector store"""
        self.index = self._create_index()
        self.chunk_metadata = {}
        self.chunk_id_to_index = {}
        self.index_to_chunk_id = {}
        
        # Remove stored files
        try:
            index_path = os.path.join(self.storage_path, "faiss_index.bin")
            metadata_path = os.path.join(self.storage_path, "metadata.pkl")
            
            if os.path.exists(index_path):
                os.remove(index_path)
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
                
            logger.info("Cleared vector store")
            
        except Exception as e:
            logger.error(f"Failed to clear vector store files: {e}")


# Global vector store instance
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get global vector store instance"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def init_vector_store(dimension: int = 384, 
                     index_type: str = "flat",
                     storage_path: Optional[str] = None) -> VectorStore:
    """
    Initialize global vector store with custom configuration
    
    Args:
        dimension: Embedding vector dimension
        index_type: FAISS index type
        storage_path: Custom storage path
        
    Returns:
        Initialized vector store instance
    """
    global _vector_store
    _vector_store = VectorStore(
        dimension=dimension,
        index_type=index_type,
        storage_path=storage_path
    )
    return _vector_store