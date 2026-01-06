"""
Embedding service using HuggingFace Sentence Transformers
"""
import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import logging

from app.core.config import settings
from app.services.semantic_chunker import Chunk
from app.services.vector_store import EmbeddedChunk

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating embeddings using HuggingFace Sentence Transformers
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize embedding service
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.model: Optional[SentenceTransformer] = None
        self.dimension = settings.EMBEDDING_DIMENSION
        
        logger.info(f"Initializing embedding service with model: {self.model_name}")
    
    def _load_model(self) -> None:
        """Load the sentence transformer model"""
        if self.model is None:
            try:
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded embedding model: {self.model_name}")
                
                # Verify dimension matches configuration
                test_embedding = self.model.encode(["test"], convert_to_numpy=True)
                actual_dimension = test_embedding.shape[1]
                
                if actual_dimension != self.dimension:
                    logger.warning(
                        f"Model dimension ({actual_dimension}) doesn't match "
                        f"configured dimension ({self.dimension}). "
                        f"Using actual dimension: {actual_dimension}"
                    )
                    self.dimension = actual_dimension
                    
            except Exception as e:
                logger.error(f"Failed to load embedding model {self.model_name}: {e}")
                raise
    
    def generate_embeddings(self, chunks: List[Chunk]) -> List[EmbeddedChunk]:
        """
        Generate embeddings for a list of chunks
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of chunks with embeddings
        """
        if not chunks:
            logger.warning("No chunks provided for embedding generation")
            return []
        
        # Load model if not already loaded
        self._load_model()
        
        # Extract text content from chunks
        texts = [chunk.content for chunk in chunks]
        
        try:
            # Generate embeddings in batch
            logger.info(f"Generating embeddings for {len(texts)} chunks")
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 10,
                batch_size=32
            )
            
            # Create embedded chunks
            embedded_chunks = []
            for chunk, embedding in zip(chunks, embeddings):
                embedded_chunk = EmbeddedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    page_number=chunk.page_number,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    chunk_index=chunk.chunk_index,
                    embedding=embedding,
                    embedding_model=self.model_name
                )
                embedded_chunks.append(embedded_chunk)
            
            logger.info(f"Successfully generated {len(embedded_chunks)} embeddings")
            return embedded_chunks
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query string
        
        Args:
            query: Query text
            
        Returns:
            Query embedding vector
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")
        
        # Load model if not already loaded
        self._load_model()
        
        try:
            embedding = self.model.encode([query], convert_to_numpy=True)
            return embedding[0]  # Return single embedding vector
            
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise
    
    def get_dimension(self) -> int:
        """Get the embedding dimension"""
        return self.dimension
    
    def get_model_name(self) -> str:
        """Get the model name"""
        return self.model_name


# Global embedding service instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get global embedding service instance"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def init_embedding_service(model_name: Optional[str] = None) -> EmbeddingService:
    """
    Initialize global embedding service with custom model
    
    Args:
        model_name: Custom model name
        
    Returns:
        Initialized embedding service instance
    """
    global _embedding_service
    _embedding_service = EmbeddingService(model_name=model_name)
    return _embedding_service