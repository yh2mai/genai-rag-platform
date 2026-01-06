"""
Test RAG Pipeline
"""
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import re
from collections import defaultdict
import tiktoken

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("Warning: rank_bm25 not available")
    BM25Okapi = None

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    print("Warning: sentence_transformers not available")
    CrossEncoder = None

from app.services.vector_store import VectorStore, SearchResult, get_vector_store
from app.services.embedding_service import EmbeddingService, get_embedding_service

logger = logging.getLogger(__name__)

@dataclass
class ConfidenceMetrics:
    """Confidence metrics for retrieval and response quality"""
    retrieval_confidence: float  # 0.0 to 1.0
    context_sufficiency: float   # 0.0 to 1.0
    hallucination_risk: float    # 0.0 to 1.0 (higher = more risk)
    overall_confidence: float    # 0.0 to 1.0
    uncertainty_indicators: List[str]  # List of uncertainty reasons
    
@dataclass
class UncertaintyResponse:
    """Response structure for uncertain queries"""
    response_type: str  # "confident", "uncertain", "insufficient_context"
    message: str
    confidence_score: float
    fallback_suggestions: List[str]
    context_issues: List[str]

@dataclass
class RankedChunk:
    """Re-ranked search result with cross-encoder relevance score"""
    chunk_id: str
    document_id: str
    content: str
    page_number: int
    vector_score: float
    bm25_score: float
    combined_score: float
    relevance_score: float  # Cross-encoder relevance score
    final_score: float      # Final combined score after re-ranking
    metadata: Dict[str, Any]

@dataclass
class Citation:
    """Citation information for source documents"""
    document_id: str
    filename: str
    page_number: int
    chunk_content: str
    relevance_score: float
    chunk_id: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None

@dataclass
class ContextWindow:
    """Optimized context window with selected chunks and metadata"""
    selected_chunks: List[RankedChunk]
    combined_context: str
    total_tokens: int
    utilization_ratio: float
    citations: List[Citation]

class TokenCounter:
    """Accurate token counting using tiktoken"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """Initialize token counter for specific model
        
        Args:
            model_name: Model name for tokenization (e.g., 'gpt-3.5-turbo', 'gpt-4')
        """
        self.model_name = model_name
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
            logger.info(f"Initialized token counter for model: {model_name}")
        except KeyError:
            # Fallback to cl100k_base encoding (used by GPT-3.5 and GPT-4)
            self.encoding = tiktoken.get_encoding("cl100k_base")
            logger.warning(f"Model {model_name} not found, using cl100k_base encoding")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken
        
        Args:
            text: Input text to count tokens for
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        
        try:
            tokens = self.encoding.encode(text)
            return len(tokens)
        except Exception as e:
            logger.error(f"Token counting failed: {e}")
            # Fallback to rough estimation
            return len(text) // 4
    
    def estimate_tokens(self, text: str) -> int:
        """Fast token estimation using character count
        
        Args:
            text: Input text
            
        Returns:
            Estimated number of tokens
        """
        # Rough estimation: ~4 characters per token for English text
        return max(1, len(text) // 4)
    
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit
        
        Args:
            text: Input text to truncate
            max_tokens: Maximum number of tokens allowed
            
        Returns:
            Truncated text that fits within token limit
        """
        if not text or max_tokens <= 0:
            return ""
        
        try:
            tokens = self.encoding.encode(text)
            if len(tokens) <= max_tokens:
                return text
            
            # Truncate tokens and decode back to text
            truncated_tokens = tokens[:max_tokens]
            truncated_text = self.encoding.decode(truncated_tokens)
            
            # Try to end at a sentence boundary
            last_period = truncated_text.rfind('.')
            last_newline = truncated_text.rfind('\n')
            
            # Use the latest sentence/paragraph boundary if it's in the last 20%
            boundary_pos = max(last_period, last_newline)
            if boundary_pos > len(truncated_text) * 0.8:
                truncated_text = truncated_text[:boundary_pos + 1]
            
            return truncated_text
            
        except Exception as e:
            logger.error(f"Token truncation failed: {e}")
            # Fallback to character-based truncation
            estimated_chars = max_tokens * 4
            return text[:estimated_chars]

class ContextOptimizer:
    """Advanced context window optimization with multiple selection strategies"""
    
    def __init__(self, token_counter: Optional[TokenCounter] = None):
        """Initialize context optimizer
        
        Args:
            token_counter: Token counter instance for accurate token counting
        """
        self.token_counter = token_counter or TokenCounter()
        logger.info("Initialized context optimizer")
    
    def optimize_context(self, 
                        chunks: List[RankedChunk], 
                        max_tokens: int = 4000,
                        strategy: str = "greedy_relevance",
                        reserve_tokens: int = 500) -> ContextWindow:
        """Select optimal chunks within context window limits using specified strategy
        
        Args:
            chunks: List of ranked chunks
            max_tokens: Maximum token limit for context
            strategy: Selection strategy ('greedy_relevance', 'diversity', 'coverage')
            reserve_tokens: Tokens to reserve for prompt and response
            
        Returns:
            ContextWindow with selected chunks and metadata
        """
        if not chunks:
            return ContextWindow(
                selected_chunks=[],
                combined_context="",
                total_tokens=0,
                utilization_ratio=0.0,
                citations=[]
            )
        
        available_tokens = max_tokens - reserve_tokens
        if available_tokens <= 0:
            logger.warning(f"No tokens available for context (max: {max_tokens}, reserve: {reserve_tokens})")
            return ContextWindow(
                selected_chunks=[],
                combined_context="",
                total_tokens=0,
                utilization_ratio=0.0,
                citations=[]
            )
        
        logger.info(f"Optimizing context with {len(chunks)} chunks, "
                   f"{available_tokens} available tokens, strategy: {strategy}")
        
        if strategy == "greedy_relevance":
            selected_chunks = self._greedy_relevance_selection(chunks, available_tokens)
        elif strategy == "diversity":
            selected_chunks = self._diversity_selection(chunks, available_tokens)
        elif strategy == "coverage":
            selected_chunks = self._coverage_selection(chunks, available_tokens)
        else:
            logger.warning(f"Unknown strategy {strategy}, using greedy_relevance")
            selected_chunks = self._greedy_relevance_selection(chunks, available_tokens)
        
        # Build combined context and calculate metrics
        combined_context = self._build_context(selected_chunks)
        total_tokens = self.token_counter.count_tokens(combined_context)
        utilization_ratio = total_tokens / max_tokens if max_tokens > 0 else 0.0
        
        # Generate citations
        citations = self._generate_citations(selected_chunks)
        
        context_window = ContextWindow(
            selected_chunks=selected_chunks,
            combined_context=combined_context,
            total_tokens=total_tokens,
            utilization_ratio=utilization_ratio,
            citations=citations
        )
        
        logger.info(f"Context optimization completed: {len(selected_chunks)} chunks, "
                   f"{total_tokens} tokens ({utilization_ratio:.2%} utilization)")
        
        return context_window
    
    def _greedy_relevance_selection(self, 
                                   chunks: List[RankedChunk], 
                                   max_tokens: int) -> List[RankedChunk]:
        """Greedy selection based on relevance scores
        
        Selects chunks in order of relevance score until token limit is reached
        """
        selected_chunks = []
        total_tokens = 0
        
        for chunk in chunks:
            chunk_tokens = self.token_counter.count_tokens(chunk.content)
            
            if total_tokens + chunk_tokens <= max_tokens:
                selected_chunks.append(chunk)
                total_tokens += chunk_tokens
            else:
                # Try to fit truncated version if it's the first chunk
                if not selected_chunks:
                    remaining_tokens = max_tokens - total_tokens
                    if remaining_tokens > 100:  # Minimum meaningful chunk size
                        truncated_content = self.token_counter.truncate_to_tokens(
                            chunk.content, remaining_tokens
                        )
                        if truncated_content:
                            truncated_chunk = self._create_truncated_chunk(chunk, truncated_content)
                            selected_chunks.append(truncated_chunk)
                break
        
        return selected_chunks
    
    def _diversity_selection(self, 
                           chunks: List[RankedChunk], 
                           max_tokens: int) -> List[RankedChunk]:
        """Selection strategy that maximizes document diversity
        
        Prioritizes chunks from different documents to provide broader coverage
        """
        selected_chunks = []
        total_tokens = 0
        used_documents = set()
        
        # First pass: select best chunk from each document
        for chunk in chunks:
            if chunk.document_id not in used_documents:
                chunk_tokens = self.token_counter.count_tokens(chunk.content)
                
                if total_tokens + chunk_tokens <= max_tokens:
                    selected_chunks.append(chunk)
                    total_tokens += chunk_tokens
                    used_documents.add(chunk.document_id)
                else:
                    break
        
        # Second pass: fill remaining space with highest relevance chunks
        for chunk in chunks:
            if chunk not in selected_chunks:
                chunk_tokens = self.token_counter.count_tokens(chunk.content)
                
                if total_tokens + chunk_tokens <= max_tokens:
                    selected_chunks.append(chunk)
                    total_tokens += chunk_tokens
                else:
                    break
        
        # Sort by relevance score to maintain quality
        selected_chunks.sort(key=lambda x: x.final_score, reverse=True)
        return selected_chunks
    
    def _coverage_selection(self, 
                          chunks: List[RankedChunk], 
                          max_tokens: int) -> List[RankedChunk]:
        """Selection strategy that maximizes topic coverage
        
        Uses a simple heuristic based on content similarity to avoid redundant chunks
        """
        if not chunks:
            return []
        
        selected_chunks = [chunks[0]]  # Always include the best chunk
        total_tokens = self.token_counter.count_tokens(chunks[0].content)
        
        for chunk in chunks[1:]:
            chunk_tokens = self.token_counter.count_tokens(chunk.content)
            
            if total_tokens + chunk_tokens > max_tokens:
                break
            
            # Check if chunk is sufficiently different from selected chunks
            if self._is_sufficiently_different(chunk, selected_chunks):
                selected_chunks.append(chunk)
                total_tokens += chunk_tokens
        
        return selected_chunks
    
    def _is_sufficiently_different(self, 
                                  candidate: RankedChunk, 
                                  selected: List[RankedChunk],
                                  similarity_threshold: float = 0.8) -> bool:
        """Check if candidate chunk is sufficiently different from selected chunks
        
        Uses simple word overlap as a proxy for content similarity
        """
        candidate_words = set(candidate.content.lower().split())
        
        for selected_chunk in selected:
            selected_words = set(selected_chunk.content.lower().split())
            
            if not candidate_words or not selected_words:
                continue
            
            # Calculate Jaccard similarity
            intersection = len(candidate_words & selected_words)
            union = len(candidate_words | selected_words)
            similarity = intersection / union if union > 0 else 0
            
            if similarity > similarity_threshold:
                return False
        
        return True
    
    def _build_context(self, chunks: List[RankedChunk]) -> str:
        """Build combined context string from selected chunks"""
        if not chunks:
            return ""
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            # Format each chunk with metadata
            chunk_header = f"[Source {i}: Document {chunk.document_id}, Page {chunk.page_number}]"
            context_parts.append(f"{chunk_header}\n{chunk.content}")
        
        return "\n\n".join(context_parts)
    
    def _generate_citations(self, chunks: List[RankedChunk]) -> List[Citation]:
        """Generate detailed citations from selected chunks"""
        citations = []
        
        for chunk in chunks:
            # Extract filename from metadata if available
            filename = chunk.metadata.get('filename', chunk.document_id)
            
            citation = Citation(
                document_id=chunk.document_id,
                filename=filename,
                page_number=chunk.page_number,
                chunk_content=chunk.content[:300] + "..." if len(chunk.content) > 300 else chunk.content,
                relevance_score=chunk.relevance_score,  # Use relevance_score instead of final_score
                chunk_id=chunk.chunk_id,
                start_char=chunk.metadata.get('start_char'),
                end_char=chunk.metadata.get('end_char')
            )
            citations.append(citation)
        
        return citations
    
    def _create_truncated_chunk(self, 
                               original_chunk: RankedChunk, 
                               truncated_content: str) -> RankedChunk:
        """Create a new chunk with truncated content"""
        return RankedChunk(
            chunk_id=original_chunk.chunk_id,
            document_id=original_chunk.document_id,
            content=truncated_content,
            page_number=original_chunk.page_number,
            vector_score=original_chunk.vector_score,
            bm25_score=original_chunk.bm25_score,
            combined_score=original_chunk.combined_score,
            relevance_score=original_chunk.relevance_score,
            final_score=original_chunk.final_score,
            metadata=original_chunk.metadata
        )
    
    def analyze_context_quality(self, context_window: ContextWindow) -> Dict[str, Any]:
        """Analyze the quality of the optimized context window
        
        Args:
            context_window: Optimized context window to analyze
            
        Returns:
            Dictionary with quality metrics
        """
        if not context_window.selected_chunks:
            return {
                "chunk_count": 0,
                "document_diversity": 0,
                "avg_relevance_score": 0.0,
                "token_utilization": 0.0,
                "coverage_score": 0.0
            }
        
        chunks = context_window.selected_chunks
        
        # Document diversity
        unique_documents = len(set(chunk.document_id for chunk in chunks))
        document_diversity = unique_documents / len(chunks) if chunks else 0
        
        # Average relevance score
        avg_relevance = sum(chunk.final_score for chunk in chunks) / len(chunks)
        
        # Coverage score (based on score distribution)
        scores = [chunk.final_score for chunk in chunks]
        score_range = max(scores) - min(scores) if len(scores) > 1 else 0
        coverage_score = 1.0 - (score_range / max(scores)) if max(scores) > 0 else 0
        
        return {
            "chunk_count": len(chunks),
            "document_diversity": document_diversity,
            "avg_relevance_score": avg_relevance,
            "token_utilization": context_window.utilization_ratio,
            "coverage_score": coverage_score,
            "total_tokens": context_window.total_tokens
        }
    """Re-ranked search result with cross-encoder relevance score"""
    chunk_id: str
    document_id: str
    content: str
    page_number: int
    vector_score: float
    bm25_score: float
    combined_score: float
    relevance_score: float  # Cross-encoder relevance score
    final_score: float      # Final combined score after re-ranking
    metadata: Dict[str, Any]

@dataclass
class ScoredChunk:
    """Search result with combined scoring from multiple retrieval methods"""
    chunk_id: str
    document_id: str
    content: str
    page_number: int
    vector_score: float
    bm25_score: float
    combined_score: float
    metadata: Dict[str, Any]

class HybridRetriever:
    """Hybrid retrieval system combining vector similarity search and BM25 keyword search"""
    
    def __init__(self, 
                 vector_store: Optional[VectorStore] = None,
                 embedding_service: Optional[EmbeddingService] = None):
        """Initialize hybrid retriever"""
        self.vector_store = vector_store or get_vector_store()
        self.embedding_service = embedding_service or get_embedding_service()
        
        # BM25 index for keyword search
        self.bm25_index: Optional[BM25Okapi] = None
        self.bm25_corpus: List[str] = []
        self.bm25_chunk_ids: List[str] = []
        
        # Build BM25 index from existing chunks
        if BM25Okapi is not None:
            self._build_bm25_index()
        
        logger.info("Initialized hybrid retriever")
    
    def _build_bm25_index(self) -> None:
        """Build BM25 index from all chunks in vector store"""
        if BM25Okapi is None:
            logger.warning("BM25 not available, skipping index build")
            return
            
        try:
            # Get all chunks from vector store
            stats = self.vector_store.get_stats()
            if stats['total_vectors'] == 0:
                logger.info("No chunks available for BM25 indexing")
                return
            
            # Extract text content and chunk IDs
            corpus = []
            chunk_ids = []
            
            # Iterate through all chunks in metadata
            for metadata in self.vector_store.chunk_metadata.values():
                content = metadata.get('content', '')
                chunk_id = metadata.get('chunk_id', '')
                
                if content and chunk_id:
                    # Tokenize content for BM25
                    tokens = self._tokenize(content)
                    corpus.append(tokens)
                    chunk_ids.append(chunk_id)
            
            if corpus:
                self.bm25_index = BM25Okapi(corpus)
                self.bm25_corpus = corpus
                self.bm25_chunk_ids = chunk_ids
                logger.info(f"Built BM25 index with {len(corpus)} documents")
            else:
                logger.warning("No valid content found for BM25 indexing")
                
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            self.bm25_index = None
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25 indexing"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return [token for token in tokens if len(token) > 1]
    
    def vector_search(self, 
                     query: str, 
                     k: int = 20,
                     score_threshold: Optional[float] = None) -> List[SearchResult]:
        """Perform vector similarity search"""
        try:
            query_embedding = self.embedding_service.generate_query_embedding(query)
            results = self.vector_store.similarity_search(
                query_embedding=query_embedding,
                k=k,
                score_threshold=score_threshold
            )
            logger.info(f"Vector search returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def bm25_search(self, 
                   query: str, 
                   k: int = 20) -> List[Tuple[str, float]]:
        """Perform BM25 keyword search"""
        if self.bm25_index is None or BM25Okapi is None:
            logger.warning("BM25 index not available")
            return []
        
        try:
            query_tokens = self._tokenize(query)
            if not query_tokens:
                logger.warning("No valid tokens in query")
                return []
            
            scores = self.bm25_index.get_scores(query_tokens)
            scored_chunks = [
                (chunk_id, score) 
                for chunk_id, score in zip(self.bm25_chunk_ids, scores)
            ]
            
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            top_results = scored_chunks[:k]
            
            logger.info(f"BM25 search returned {len(top_results)} results")
            return top_results
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
    
    def reciprocal_rank_fusion(self, 
                              vector_results: List[SearchResult],
                              bm25_results: List[Tuple[str, float]],
                              k: int = 60) -> List[ScoredChunk]:
        """Combine vector and BM25 results using Reciprocal Rank Fusion (RRF)"""
        vector_scores = {result.chunk_id: result.similarity_score for result in vector_results}
        bm25_scores = {chunk_id: score for chunk_id, score in bm25_results}
        
        vector_ranks = {result.chunk_id: rank + 1 for rank, result in enumerate(vector_results)}
        bm25_ranks = {chunk_id: rank + 1 for rank, (chunk_id, _) in enumerate(bm25_results)}
        
        all_chunk_ids = set(vector_scores.keys()) | set(bm25_scores.keys())
        
        rrf_scores = {}
        for chunk_id in all_chunk_ids:
            rrf_score = 0.0
            
            if chunk_id in vector_ranks:
                rrf_score += 1.0 / (k + vector_ranks[chunk_id])
            
            if chunk_id in bm25_ranks:
                rrf_score += 1.0 / (k + bm25_ranks[chunk_id])
            
            rrf_scores[chunk_id] = rrf_score
        
        scored_chunks = []
        for chunk_id, combined_score in rrf_scores.items():
            chunk_result = self.vector_store.get_chunk_by_id(chunk_id)
            if chunk_result is None:
                logger.warning(f"Chunk {chunk_id} not found in vector store")
                continue
            
            scored_chunk = ScoredChunk(
                chunk_id=chunk_id,
                document_id=chunk_result.document_id,
                content=chunk_result.content,
                page_number=chunk_result.page_number,
                vector_score=vector_scores.get(chunk_id, 0.0),
                bm25_score=bm25_scores.get(chunk_id, 0.0),
                combined_score=combined_score,
                metadata=chunk_result.metadata
            )
            scored_chunks.append(scored_chunk)
        
        scored_chunks.sort(key=lambda x: x.combined_score, reverse=True)
        logger.info(f"RRF fusion produced {len(scored_chunks)} scored chunks")
        return scored_chunks
    
    def hybrid_retrieve(self, 
                       query: str, 
                       top_k: int = 20,
                       vector_k: int = 50,
                       bm25_k: int = 50,
                       rrf_k: int = 60) -> List[ScoredChunk]:
        """Perform hybrid retrieval combining vector and BM25 search"""
        if not query.strip():
            logger.warning("Empty query provided")
            return []
        
        logger.info(f"Performing hybrid retrieval for query: '{query[:100]}...'")
        
        vector_results = self.vector_search(query, k=vector_k)
        bm25_results = self.bm25_search(query, k=bm25_k)
        
        scored_chunks = self.reciprocal_rank_fusion(
            vector_results=vector_results,
            bm25_results=bm25_results,
            k=rrf_k
        )
        
        final_results = scored_chunks[:top_k]
        logger.info(f"Hybrid retrieval returned {len(final_results)} final results")
        return final_results

class CrossEncoderReranker:
    """Cross-encoder model for re-ranking search results based on query-document relevance"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize cross-encoder re-ranker
        
        Args:
            model_name: HuggingFace model name for cross-encoder
        """
        self.model_name = model_name
        self.model: Optional[CrossEncoder] = None
        
        if CrossEncoder is not None:
            try:
                self.model = CrossEncoder(model_name)
                logger.info(f"Initialized cross-encoder model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load cross-encoder model {model_name}: {e}")
                self.model = None
        else:
            logger.warning("CrossEncoder not available, re-ranking will use fallback scoring")
    
    def rerank_chunks(self, 
                     query: str, 
                     chunks: List[ScoredChunk], 
                     top_k: Optional[int] = None) -> List[RankedChunk]:
        """Re-rank chunks using cross-encoder model
        
        Args:
            query: User query
            chunks: List of scored chunks from hybrid retrieval
            top_k: Number of top results to return (None for all)
            
        Returns:
            List of re-ranked chunks with relevance scores
        """
        if not chunks:
            logger.warning("No chunks provided for re-ranking")
            return []
        
        if not query.strip():
            logger.warning("Empty query provided for re-ranking")
            return self._fallback_ranking(chunks, top_k)
        
        logger.info(f"Re-ranking {len(chunks)} chunks for query: '{query[:100]}...'")
        
        if self.model is None:
            logger.warning("Cross-encoder model not available, using fallback ranking")
            return self._fallback_ranking(chunks, top_k)
        
        try:
            # Prepare query-document pairs for cross-encoder
            query_doc_pairs = []
            for chunk in chunks:
                # Truncate content to avoid token limits (typically 512 tokens for cross-encoders)
                content = chunk.content[:2000]  # Rough approximation of token limit
                query_doc_pairs.append([query, content])
            
            # Get relevance scores from cross-encoder
            relevance_scores = self.model.predict(query_doc_pairs)
            
            # Convert to list if numpy array
            if hasattr(relevance_scores, 'tolist'):
                relevance_scores = relevance_scores.tolist()
            
            # Create ranked chunks with combined scoring
            ranked_chunks = []
            for chunk, relevance_score in zip(chunks, relevance_scores):
                # Combine cross-encoder score with original hybrid score
                # Weight: 70% cross-encoder, 30% hybrid score
                final_score = 0.7 * float(relevance_score) + 0.3 * chunk.combined_score
                
                ranked_chunk = RankedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    page_number=chunk.page_number,
                    vector_score=chunk.vector_score,
                    bm25_score=chunk.bm25_score,
                    combined_score=chunk.combined_score,
                    relevance_score=float(relevance_score),
                    final_score=final_score,
                    metadata=chunk.metadata
                )
                ranked_chunks.append(ranked_chunk)
            
            # Sort by final score (descending)
            ranked_chunks.sort(key=lambda x: x.final_score, reverse=True)
            
            # Apply top-k filtering if specified
            if top_k is not None:
                ranked_chunks = ranked_chunks[:top_k]
            
            logger.info(f"Re-ranking completed, returning {len(ranked_chunks)} results")
            return ranked_chunks
            
        except Exception as e:
            logger.error(f"Cross-encoder re-ranking failed: {e}")
            return self._fallback_ranking(chunks, top_k)
    
    def _fallback_ranking(self, 
                         chunks: List[ScoredChunk], 
                         top_k: Optional[int] = None) -> List[RankedChunk]:
        """Fallback ranking when cross-encoder is not available
        
        Uses the original combined score as the relevance score
        """
        logger.info("Using fallback ranking based on hybrid scores")
        
        ranked_chunks = []
        for chunk in chunks:
            ranked_chunk = RankedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                content=chunk.content,
                page_number=chunk.page_number,
                vector_score=chunk.vector_score,
                bm25_score=chunk.bm25_score,
                combined_score=chunk.combined_score,
                relevance_score=chunk.combined_score,  # Use hybrid score as relevance
                final_score=chunk.combined_score,
                metadata=chunk.metadata
            )
            ranked_chunks.append(ranked_chunk)
        
        # Sort by combined score (already sorted, but ensure consistency)
        ranked_chunks.sort(key=lambda x: x.final_score, reverse=True)
        
        if top_k is not None:
            ranked_chunks = ranked_chunks[:top_k]
        
        return ranked_chunks
    
    def get_relevance_threshold(self, 
                               ranked_chunks: List[RankedChunk], 
                               percentile: float = 0.5) -> float:
        """Calculate relevance threshold based on score distribution
        
        Args:
            ranked_chunks: List of ranked chunks
            percentile: Percentile for threshold (0.5 = median)
            
        Returns:
            Relevance score threshold
        """
        if not ranked_chunks:
            return 0.0
        
        scores = [chunk.relevance_score for chunk in ranked_chunks]
        scores.sort(reverse=True)
        
        index = int(len(scores) * percentile)
        threshold = scores[min(index, len(scores) - 1)]
        
        logger.info(f"Calculated relevance threshold: {threshold:.4f} at {percentile*100}th percentile")
        return threshold

class UncertaintyHandler:
    """Handles uncertainty detection, confidence scoring, and fallback responses"""
    
    def __init__(self, 
                 min_confidence_threshold: float = 0.6,
                 min_context_sufficiency: float = 0.5,
                 max_hallucination_risk: float = 0.3):
        """Initialize uncertainty handler
        
        Args:
            min_confidence_threshold: Minimum confidence for confident responses
            min_context_sufficiency: Minimum context sufficiency score
            max_hallucination_risk: Maximum acceptable hallucination risk
        """
        self.min_confidence_threshold = min_confidence_threshold
        self.min_context_sufficiency = min_context_sufficiency
        self.max_hallucination_risk = max_hallucination_risk
        logger.info("Initialized uncertainty handler")
    
    def calculate_confidence_metrics(self, 
                                   ranked_chunks: List[RankedChunk],
                                   query: str,
                                   context_window: Optional[ContextWindow] = None) -> ConfidenceMetrics:
        """Calculate comprehensive confidence metrics for retrieval results
        
        Args:
            ranked_chunks: List of ranked chunks from retrieval
            query: Original user query
            context_window: Optional context window for additional analysis
            
        Returns:
            ConfidenceMetrics with detailed confidence scores
        """
        if not ranked_chunks:
            return ConfidenceMetrics(
                retrieval_confidence=0.0,
                context_sufficiency=0.0,
                hallucination_risk=1.0,
                overall_confidence=0.0,
                uncertainty_indicators=["No relevant documents found"]
            )
        
        # Calculate retrieval confidence based on scores and distribution
        retrieval_confidence = self._calculate_retrieval_confidence(ranked_chunks)
        
        # Calculate context sufficiency
        context_sufficiency = self._calculate_context_sufficiency(
            ranked_chunks, query, context_window
        )
        
        # Calculate hallucination risk
        hallucination_risk = self._calculate_hallucination_risk(
            ranked_chunks, query
        )
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            retrieval_confidence, context_sufficiency, hallucination_risk
        )
        
        # Identify uncertainty indicators
        uncertainty_indicators = self._identify_uncertainty_indicators(
            retrieval_confidence, context_sufficiency, hallucination_risk, ranked_chunks
        )
        
        return ConfidenceMetrics(
            retrieval_confidence=retrieval_confidence,
            context_sufficiency=context_sufficiency,
            hallucination_risk=hallucination_risk,
            overall_confidence=overall_confidence,
            uncertainty_indicators=uncertainty_indicators
        )
    
    def _calculate_retrieval_confidence(self, ranked_chunks: List[RankedChunk]) -> float:
        """Calculate confidence based on retrieval scores and distribution"""
        if not ranked_chunks:
            return 0.0
        
        scores = [chunk.final_score for chunk in ranked_chunks]
        
        # Top score confidence
        top_score = max(scores)
        top_score_confidence = min(top_score, 1.0)
        
        # Score distribution confidence (higher variance = lower confidence)
        if len(scores) > 1:
            mean_score = np.mean(scores)
            score_variance = np.var(scores)
            # Normalize variance to 0-1 range (lower variance = higher confidence)
            distribution_confidence = max(0.0, 1.0 - (score_variance / (mean_score + 1e-6)))
        else:
            distribution_confidence = 0.5  # Neutral when only one result
        
        # Number of results confidence (more results can indicate better coverage)
        num_results = len(ranked_chunks)
        results_confidence = min(1.0, num_results / 10.0)  # Normalize to 10 results
        
        # Weighted combination
        retrieval_confidence = (
            0.5 * top_score_confidence +
            0.3 * distribution_confidence +
            0.2 * results_confidence
        )
        
        return max(0.0, min(1.0, retrieval_confidence))
    
    def _calculate_context_sufficiency(self, 
                                     ranked_chunks: List[RankedChunk],
                                     query: str,
                                     context_window: Optional[ContextWindow] = None) -> float:
        """Calculate how sufficient the context is for answering the query"""
        if not ranked_chunks:
            return 0.0
        
        # Query complexity analysis
        query_words = set(query.lower().split())
        query_complexity = min(1.0, len(query_words) / 20.0)  # Normalize to 20 words
        
        # Content coverage analysis
        all_content = " ".join([chunk.content for chunk in ranked_chunks])
        content_words = set(all_content.lower().split())
        
        # Calculate query term coverage
        if query_words:
            coverage = len(query_words & content_words) / len(query_words)
        else:
            coverage = 0.0
        
        # Document diversity (more diverse sources = higher sufficiency)
        unique_docs = len(set(chunk.document_id for chunk in ranked_chunks))
        diversity_score = min(1.0, unique_docs / 5.0)  # Normalize to 5 documents
        
        # Content length sufficiency
        total_content_length = len(all_content)
        length_sufficiency = min(1.0, total_content_length / 2000.0)  # Normalize to 2000 chars
        
        # Context window utilization (if available)
        utilization_score = 1.0
        if context_window:
            utilization_score = context_window.utilization_ratio
        
        # Weighted combination
        context_sufficiency = (
            0.3 * coverage +
            0.2 * diversity_score +
            0.2 * length_sufficiency +
            0.2 * utilization_score +
            0.1 * (1.0 - query_complexity)  # Simpler queries need less context
        )
        
        return max(0.0, min(1.0, context_sufficiency))
    
    def _calculate_hallucination_risk(self, 
                                    ranked_chunks: List[RankedChunk],
                                    query: str) -> float:
        """Calculate risk of hallucination based on context quality"""
        if not ranked_chunks:
            return 1.0  # High risk when no context
        
        # Low relevance scores indicate higher hallucination risk
        relevance_scores = [chunk.relevance_score for chunk in ranked_chunks]
        avg_relevance = np.mean(relevance_scores)
        relevance_risk = 1.0 - min(1.0, avg_relevance)
        
        # Large gaps between top scores indicate uncertainty
        if len(relevance_scores) > 1:
            score_gap = relevance_scores[0] - relevance_scores[1]
            gap_risk = max(0.0, 0.5 - score_gap)  # Risk increases when gap is small
        else:
            gap_risk = 0.3  # Moderate risk with only one result
        
        # Query-context mismatch detection (simple keyword overlap)
        query_words = set(query.lower().split())
        all_content = " ".join([chunk.content for chunk in ranked_chunks])
        content_words = set(all_content.lower().split())
        
        if query_words:
            overlap_ratio = len(query_words & content_words) / len(query_words)
            mismatch_risk = 1.0 - overlap_ratio
        else:
            mismatch_risk = 0.5
        
        # Weighted combination
        hallucination_risk = (
            0.4 * relevance_risk +
            0.3 * gap_risk +
            0.3 * mismatch_risk
        )
        
        return max(0.0, min(1.0, hallucination_risk))
    
    def _calculate_overall_confidence(self, 
                                    retrieval_confidence: float,
                                    context_sufficiency: float,
                                    hallucination_risk: float) -> float:
        """Calculate overall confidence score"""
        # Weighted combination with hallucination risk as negative factor
        overall_confidence = (
            0.4 * retrieval_confidence +
            0.4 * context_sufficiency +
            0.2 * (1.0 - hallucination_risk)
        )
        
        return max(0.0, min(1.0, overall_confidence))
    
    def _identify_uncertainty_indicators(self, 
                                       retrieval_confidence: float,
                                       context_sufficiency: float,
                                       hallucination_risk: float,
                                       ranked_chunks: List[RankedChunk]) -> List[str]:
        """Identify specific reasons for uncertainty"""
        indicators = []
        
        if retrieval_confidence < 0.5:
            indicators.append("Low retrieval confidence - search results may not be relevant")
        
        if context_sufficiency < 0.5:
            indicators.append("Insufficient context - available information may not fully address the query")
        
        if hallucination_risk > 0.5:
            indicators.append("High hallucination risk - response may contain unsupported information")
        
        if not ranked_chunks:
            indicators.append("No relevant documents found")
        elif len(ranked_chunks) < 3:
            indicators.append("Limited number of relevant sources")
        
        # Check for low relevance scores
        if ranked_chunks:
            avg_relevance = np.mean([chunk.relevance_score for chunk in ranked_chunks])
            if avg_relevance < 0.3:
                indicators.append("Low relevance scores across all retrieved documents")
        
        # Check for document diversity
        if ranked_chunks:
            unique_docs = len(set(chunk.document_id for chunk in ranked_chunks))
            if unique_docs == 1:
                indicators.append("All results from single document - limited perspective")
        
        return indicators
    
    def generate_uncertainty_response(self, 
                                    confidence_metrics: ConfidenceMetrics,
                                    query: str,
                                    ranked_chunks: List[RankedChunk]) -> UncertaintyResponse:
        """Generate appropriate response based on confidence metrics"""
        
        # Determine response type based on confidence thresholds
        if confidence_metrics.overall_confidence >= self.min_confidence_threshold:
            response_type = "confident"
            message = "I found relevant information to answer your query."
            fallback_suggestions = []
            
        elif confidence_metrics.context_sufficiency < self.min_context_sufficiency:
            response_type = "insufficient_context"
            message = ("I found some relevant information, but it may not be sufficient "
                      "to fully answer your query. Please consider refining your question "
                      "or providing more specific details.")
            fallback_suggestions = self._generate_fallback_suggestions(query, ranked_chunks)
            
        else:
            response_type = "uncertain"
            message = ("I found some potentially relevant information, but I'm not fully "
                      "confident in the completeness or accuracy of the answer. "
                      "Please verify the information from the original sources.")
            fallback_suggestions = self._generate_fallback_suggestions(query, ranked_chunks)
        
        return UncertaintyResponse(
            response_type=response_type,
            message=message,
            confidence_score=confidence_metrics.overall_confidence,
            fallback_suggestions=fallback_suggestions,
            context_issues=confidence_metrics.uncertainty_indicators
        )
    
    def _generate_fallback_suggestions(self, 
                                     query: str,
                                     ranked_chunks: List[RankedChunk]) -> List[str]:
        """Generate fallback suggestions for uncertain queries"""
        suggestions = []
        
        # Suggest query refinement
        suggestions.append("Try rephrasing your question with more specific terms")
        suggestions.append("Break down complex questions into simpler parts")
        
        # Suggest based on available documents
        if ranked_chunks:
            available_docs = set(chunk.metadata.get('filename', chunk.document_id) 
                               for chunk in ranked_chunks)
            if len(available_docs) > 1:
                suggestions.append(f"Consider focusing on specific documents: {', '.join(list(available_docs)[:3])}")
            
            # Suggest based on page numbers
            pages = set(chunk.page_number for chunk in ranked_chunks)
            if len(pages) > 1:
                suggestions.append(f"Information spans multiple pages ({min(pages)}-{max(pages)}), consider narrowing the scope")
        
        # General suggestions
        suggestions.append("Upload additional relevant documents if available")
        suggestions.append("Consult the original documents directly for complete context")
        
        return suggestions[:3]  # Limit to top 3 suggestions
    
    def should_provide_response(self, confidence_metrics: ConfidenceMetrics) -> bool:
        """Determine if a response should be provided based on confidence metrics"""
        return (
            confidence_metrics.overall_confidence >= 0.3 and  # Minimum threshold
            confidence_metrics.hallucination_risk <= 0.8      # Maximum risk threshold
        )
    
    def get_confidence_explanation(self, confidence_metrics: ConfidenceMetrics) -> str:
        """Generate human-readable explanation of confidence scores"""
        explanations = []
        
        if confidence_metrics.retrieval_confidence >= 0.7:
            explanations.append("Strong retrieval match")
        elif confidence_metrics.retrieval_confidence >= 0.4:
            explanations.append("Moderate retrieval match")
        else:
            explanations.append("Weak retrieval match")
        
        if confidence_metrics.context_sufficiency >= 0.7:
            explanations.append("sufficient context")
        elif confidence_metrics.context_sufficiency >= 0.4:
            explanations.append("limited context")
        else:
            explanations.append("insufficient context")
        
        if confidence_metrics.hallucination_risk <= 0.3:
            explanations.append("low hallucination risk")
        elif confidence_metrics.hallucination_risk <= 0.6:
            explanations.append("moderate hallucination risk")
        else:
            explanations.append("high hallucination risk")
        
        return f"Confidence based on: {', '.join(explanations)}"

class RAGEngine:
    """Complete RAG pipeline with hybrid retrieval and re-ranking"""
    
    def __init__(self, 
                 hybrid_retriever: Optional[HybridRetriever] = None,
                 reranker: Optional[CrossEncoderReranker] = None,
                 context_optimizer: Optional[ContextOptimizer] = None,
                 uncertainty_handler: Optional[UncertaintyHandler] = None):
        """Initialize RAG engine"""
        self.hybrid_retriever = hybrid_retriever or HybridRetriever()
        self.reranker = reranker or CrossEncoderReranker()
        self.context_optimizer = context_optimizer or ContextOptimizer()
        self.uncertainty_handler = uncertainty_handler or UncertaintyHandler()
        logger.info("Initialized RAG engine with re-ranking, context optimization, and uncertainty handling")
    
    def hybrid_retrieve(self, 
                       query: str, 
                       top_k: int = 20,
                       **kwargs) -> List[ScoredChunk]:
        """Retrieve relevant chunks using hybrid search"""
        return self.hybrid_retriever.hybrid_retrieve(
            query=query,
            top_k=top_k,
            **kwargs
        )
    
    def rerank_results(self, 
                      query: str, 
                      chunks: List[ScoredChunk],
                      top_k: Optional[int] = None) -> List[RankedChunk]:
        """Re-rank search results using cross-encoder model
        
        Args:
            query: User query
            chunks: List of scored chunks from hybrid retrieval
            top_k: Number of top results to return
            
        Returns:
            List of re-ranked chunks with relevance scores
        """
        return self.reranker.rerank_chunks(query, chunks, top_k)
    
    def retrieve_and_rerank(self, 
                           query: str, 
                           retrieval_k: int = 50,
                           final_k: int = 20,
                           **kwargs) -> List[RankedChunk]:
        """Complete retrieval and re-ranking pipeline
        
        Args:
            query: User query
            retrieval_k: Number of chunks to retrieve initially
            final_k: Number of final re-ranked results
            **kwargs: Additional arguments for hybrid retrieval
            
        Returns:
            List of re-ranked chunks
        """
        if not query.strip():
            logger.warning("Empty query provided")
            return []
        
        logger.info(f"Starting retrieve and rerank pipeline for query: '{query[:100]}...'")
        
        # Step 1: Hybrid retrieval
        scored_chunks = self.hybrid_retrieve(
            query=query,
            top_k=retrieval_k,
            **kwargs
        )
        
        if not scored_chunks:
            logger.warning("No chunks retrieved from hybrid search")
            return []
        
        # Step 2: Re-ranking
        ranked_chunks = self.rerank_results(
            query=query,
            chunks=scored_chunks,
            top_k=final_k
        )
        
        logger.info(f"Pipeline completed, returning {len(ranked_chunks)} re-ranked results")
        return ranked_chunks
    
    def optimize_context(self, 
                        chunks: List[RankedChunk], 
                        max_tokens: int = 4000,
                        strategy: str = "greedy_relevance",
                        reserve_tokens: int = 500) -> ContextWindow:
        """Select optimal chunks within context window limits using advanced optimization
        
        Args:
            chunks: List of ranked chunks
            max_tokens: Maximum token limit for context
            strategy: Selection strategy ('greedy_relevance', 'diversity', 'coverage')
            reserve_tokens: Tokens to reserve for prompt and response
            
        Returns:
            ContextWindow with selected chunks and metadata
        """
        return self.context_optimizer.optimize_context(
            chunks=chunks,
            max_tokens=max_tokens,
            strategy=strategy,
            reserve_tokens=reserve_tokens
        )
    
    def generate_citations(self, used_chunks: List[RankedChunk]) -> List[Citation]:
        """Generate enhanced citations for used chunks
        
        Args:
            used_chunks: List of chunks used in response generation
            
        Returns:
            List of Citation objects with detailed metadata
        """
        return self.context_optimizer._generate_citations(used_chunks)
    
    def process_query(self, 
                     query: str,
                     max_context_tokens: int = 4000,
                     retrieval_k: int = 50,
                     final_k: int = 20,
                     context_strategy: str = "greedy_relevance",
                     use_reranking: bool = True) -> Dict[str, Any]:
        """Complete query processing pipeline with context optimization
        
        Args:
            query: User query
            max_context_tokens: Maximum tokens for context window
            retrieval_k: Number of chunks to retrieve initially
            final_k: Number of chunks for re-ranking
            context_strategy: Context selection strategy
            use_reranking: Whether to apply re-ranking
            
        Returns:
            Dictionary with processed results including context window and citations
        """
        if not query.strip():
            logger.warning("Empty query provided")
            return {
                "query": query,
                "context_window": ContextWindow([], "", 0, 0.0, []),
                "quality_metrics": {},
                "error": "Empty query provided"
            }
        
        logger.info(f"Processing query with context optimization: '{query[:100]}...'")
        
        try:
            # Step 1: Retrieve and optionally re-rank
            if use_reranking:
                ranked_chunks = self.retrieve_and_rerank(
                    query=query,
                    retrieval_k=retrieval_k,
                    final_k=final_k
                )
            else:
                scored_chunks = self.hybrid_retrieve(query=query, top_k=final_k)
                # Convert to RankedChunk for consistency
                ranked_chunks = []
                for chunk in scored_chunks:
                    ranked_chunk = RankedChunk(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        content=chunk.content,
                        page_number=chunk.page_number,
                        vector_score=chunk.vector_score,
                        bm25_score=chunk.bm25_score,
                        combined_score=chunk.combined_score,
                        relevance_score=chunk.combined_score,
                        final_score=chunk.combined_score,
                        metadata=chunk.metadata
                    )
                    ranked_chunks.append(ranked_chunk)
            
            if not ranked_chunks:
                logger.warning("No chunks retrieved for query")
                return {
                    "query": query,
                    "context_window": ContextWindow([], "", 0, 0.0, []),
                    "quality_metrics": {},
                    "error": "No relevant documents found"
                }
            
            # Step 2: Optimize context window
            context_window = self.optimize_context(
                chunks=ranked_chunks,
                max_tokens=max_context_tokens,
                strategy=context_strategy
            )
            
            # Step 3: Analyze context quality
            quality_metrics = self.context_optimizer.analyze_context_quality(context_window)
            
            result = {
                "query": query,
                "context_window": context_window,
                "quality_metrics": quality_metrics,
                "total_retrieved": len(ranked_chunks),
                "selected_chunks": len(context_window.selected_chunks)
            }
            
            logger.info(f"Query processing completed: {len(context_window.selected_chunks)} chunks selected, "
                       f"{context_window.total_tokens} tokens, "
                       f"{context_window.utilization_ratio:.2%} utilization")
            
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "query": query,
                "context_window": ContextWindow([], "", 0, 0.0, []),
                "quality_metrics": {},
                "error": str(e)
            }
    
    def retrieve(self, 
                query: str, 
                top_k: int = 20,
                use_reranking: bool = True,
                **kwargs) -> List[RankedChunk]:
        """Main retrieval method with optional re-ranking
        
        Args:
            query: User query
            top_k: Number of final results
            use_reranking: Whether to apply re-ranking
            **kwargs: Additional arguments
            
        Returns:
            List of ranked chunks (RankedChunk if reranking, converted ScoredChunk otherwise)
        """
        if use_reranking:
            return self.retrieve_and_rerank(
                query=query,
                retrieval_k=min(50, top_k * 3),  # Retrieve more for better re-ranking
                final_k=top_k,
                **kwargs
            )
        else:
            # Convert ScoredChunk to RankedChunk for consistency
            scored_chunks = self.hybrid_retrieve(query=query, top_k=top_k, **kwargs)
            ranked_chunks = []
            
            for chunk in scored_chunks:
                ranked_chunk = RankedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    page_number=chunk.page_number,
                    vector_score=chunk.vector_score,
                    bm25_score=chunk.bm25_score,
                    combined_score=chunk.combined_score,
                    relevance_score=chunk.combined_score,
                    final_score=chunk.combined_score,
                    metadata=chunk.metadata
                )
                ranked_chunks.append(ranked_chunk)
            
            return ranked_chunks

    def calculate_confidence(self, 
                           ranked_chunks: List[RankedChunk],
                           query: str,
                           context_window: Optional[ContextWindow] = None) -> ConfidenceMetrics:
        """Calculate confidence metrics for retrieval results
        
        Args:
            ranked_chunks: List of ranked chunks from retrieval
            query: Original user query
            context_window: Optional context window for additional analysis
            
        Returns:
            ConfidenceMetrics with detailed confidence scores
        """
        return self.uncertainty_handler.calculate_confidence_metrics(
            ranked_chunks, query, context_window
        )
    
    def handle_uncertainty(self, 
                         confidence_metrics: ConfidenceMetrics,
                         query: str,
                         ranked_chunks: List[RankedChunk]) -> UncertaintyResponse:
        """Handle uncertain queries and generate appropriate responses
        
        Args:
            confidence_metrics: Calculated confidence metrics
            query: Original user query
            ranked_chunks: Retrieved chunks
            
        Returns:
            UncertaintyResponse with appropriate handling
        """
        return self.uncertainty_handler.generate_uncertainty_response(
            confidence_metrics, query, ranked_chunks
        )
    
    def should_provide_response(self, confidence_metrics: ConfidenceMetrics) -> bool:
        """Determine if a response should be provided based on confidence
        
        Args:
            confidence_metrics: Calculated confidence metrics
            
        Returns:
            Boolean indicating whether to provide a response
        """
        return self.uncertainty_handler.should_provide_response(confidence_metrics)
    
    def process_query_with_uncertainty(self, 
                                     query: str,
                                     max_context_tokens: int = 4000,
                                     retrieval_k: int = 50,
                                     final_k: int = 20,
                                     context_strategy: str = "greedy_relevance",
                                     use_reranking: bool = True) -> Dict[str, Any]:
        """Complete query processing pipeline with uncertainty handling
        
        Args:
            query: User query
            max_context_tokens: Maximum tokens for context window
            retrieval_k: Number of chunks to retrieve initially
            final_k: Number of chunks for re-ranking
            context_strategy: Context selection strategy
            use_reranking: Whether to apply re-ranking
            
        Returns:
            Dictionary with processed results including uncertainty analysis
        """
        if not query.strip():
            logger.warning("Empty query provided")
            return {
                "query": query,
                "context_window": ContextWindow([], "", 0, 0.0, []),
                "confidence_metrics": ConfidenceMetrics(0.0, 0.0, 1.0, 0.0, ["Empty query"]),
                "uncertainty_response": UncertaintyResponse(
                    "insufficient_context", 
                    "Please provide a valid query.", 
                    0.0, 
                    ["Try asking a specific question"], 
                    ["Empty query provided"]
                ),
                "should_respond": False,
                "error": "Empty query provided"
            }
        
        logger.info(f"Processing query with uncertainty handling: '{query[:100]}...'")
        
        try:
            # Step 1: Retrieve and optionally re-rank
            if use_reranking:
                ranked_chunks = self.retrieve_and_rerank(
                    query=query,
                    retrieval_k=retrieval_k,
                    final_k=final_k
                )
            else:
                scored_chunks = self.hybrid_retrieve(query=query, top_k=final_k)
                # Convert to RankedChunk for consistency
                ranked_chunks = []
                for chunk in scored_chunks:
                    ranked_chunk = RankedChunk(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        content=chunk.content,
                        page_number=chunk.page_number,
                        vector_score=chunk.vector_score,
                        bm25_score=chunk.bm25_score,
                        combined_score=chunk.combined_score,
                        relevance_score=chunk.combined_score,
                        final_score=chunk.combined_score,
                        metadata=chunk.metadata
                    )
                    ranked_chunks.append(ranked_chunk)
            
            # Step 2: Optimize context window
            context_window = ContextWindow([], "", 0, 0.0, [])
            if ranked_chunks:
                context_window = self.optimize_context(
                    chunks=ranked_chunks,
                    max_tokens=max_context_tokens,
                    strategy=context_strategy
                )
            
            # Step 3: Calculate confidence metrics
            confidence_metrics = self.calculate_confidence(
                ranked_chunks, query, context_window
            )
            
            # Step 4: Handle uncertainty
            uncertainty_response = self.handle_uncertainty(
                confidence_metrics, query, ranked_chunks
            )
            
            # Step 5: Determine if response should be provided
            should_respond = self.should_provide_response(confidence_metrics)
            
            # Step 6: Analyze context quality
            quality_metrics = {}
            if context_window.selected_chunks:
                quality_metrics = self.context_optimizer.analyze_context_quality(context_window)
            
            result = {
                "query": query,
                "context_window": context_window,
                "confidence_metrics": confidence_metrics,
                "uncertainty_response": uncertainty_response,
                "should_respond": should_respond,
                "quality_metrics": quality_metrics,
                "total_retrieved": len(ranked_chunks),
                "selected_chunks": len(context_window.selected_chunks)
            }
            
            logger.info(f"Query processing with uncertainty completed: "
                       f"confidence={confidence_metrics.overall_confidence:.3f}, "
                       f"should_respond={should_respond}, "
                       f"response_type={uncertainty_response.response_type}")
            
            return result
            
        except Exception as e:
            logger.error(f"Query processing with uncertainty failed: {e}")
            error_confidence = ConfidenceMetrics(0.0, 0.0, 1.0, 0.0, [f"Processing error: {str(e)}"])
            error_response = UncertaintyResponse(
                "insufficient_context",
                "An error occurred while processing your query. Please try again.",
                0.0,
                ["Try rephrasing your question", "Check if documents are properly uploaded"],
                [f"System error: {str(e)}"]
            )
            
            return {
                "query": query,
                "context_window": ContextWindow([], "", 0, 0.0, []),
                "confidence_metrics": error_confidence,
                "uncertainty_response": error_response,
                "should_respond": False,
                "quality_metrics": {},
                "error": str(e)
            }

if __name__ == "__main__":
    print("Testing RAG pipeline with enhanced context optimization...")
    
    # Test token counter
    token_counter = TokenCounter()
    print("✓ TokenCounter created")
    
    # Test context optimizer
    context_optimizer = ContextOptimizer()
    print("✓ ContextOptimizer created")
    
    # Test hybrid retriever
    retriever = HybridRetriever()
    print("✓ HybridRetriever created")
    
    # Test cross-encoder re-ranker
    reranker = CrossEncoderReranker()
    print("✓ CrossEncoderReranker created")
    
    # Test complete RAG engine with context optimization
    engine = RAGEngine()
    print("✓ RAGEngine with context optimization created")
    
    # Test uncertainty handler
    uncertainty_handler = UncertaintyHandler()
    print("✓ UncertaintyHandler created")
    
    print("All classes created successfully")
    print("Context optimization system implemented with:")
    print("  - Accurate token counting using tiktoken")
    print("  - Multiple selection strategies (greedy_relevance, diversity, coverage)")
    print("  - Enhanced citation generation with detailed metadata")
    print("  - Context quality analysis and metrics")
    print("Uncertainty handling system implemented with:")
    print("  - Confidence scoring for retrieval results")
    print("  - Context sufficiency analysis")
    print("  - Hallucination risk assessment")
    print("  - Fallback responses for insufficient context")
    print("  - Uncertainty indicators and explanations")


# Factory function for creating RAG engine instances
def get_rag_engine() -> RAGEngine:
    """
    Factory function to create and configure a RAGEngine instance.
    
    Returns:
        Configured RAGEngine instance
    """
    return RAGEngine()