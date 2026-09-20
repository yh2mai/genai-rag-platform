"""
Agent Orchestration System using LangGraph

This module implements the multi-agent workflow orchestration system using LangGraph
to coordinate Planner, Retriever, Answer, and Evaluator agents for complex query processing.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import re
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Metadata for processed documents"""
    document_id: str
    filename: str
    file_size: int
    page_count: int
    creation_date: datetime
    processing_date: datetime
    checksum: str


@dataclass
class Chunk:
    """Basic document chunk structure"""
    chunk_id: str
    document_id: str
    content: str
    page_number: int
    start_char: int
    end_char: int
    metadata: DocumentMetadata


@dataclass
class EmbeddedChunk(Chunk):
    """Document chunk with embedding vector"""
    embedding: np.ndarray
    embedding_model: str


@dataclass
class ScoredChunk(Chunk):
    """Document chunk with retrieval score"""
    score: float
    retrieval_method: str  # 'vector', 'keyword', or 'hybrid'


@dataclass
class RankedChunk(ScoredChunk):
    """Document chunk with re-ranking score"""
    rank: int
    rerank_score: float
    vector_score: float = 0.0
    bm25_score: float = 0.0
    combined_score: float = 0.0
    relevance_score: float = 0.0
    final_score: float = 0.0


@dataclass
class SearchResult:
    """Vector database search result"""
    chunk: Chunk
    score: float
    distance: float


@dataclass
class Citation:
    """Citation information for generated responses"""
    document_id: str
    filename: str
    page_number: int
    chunk_content: str
    relevance_score: float


@dataclass
class AgentState:
    """
    Central state object that flows through the LangGraph workflow.
    Contains all information needed by agents during query processing.
    """
    # Input
    query: str
    
    # Planning phase
    plan: Optional[str] = None
    
    # Retrieval phase
    retrieved_chunks: List[RankedChunk] = field(default_factory=list)
    
    # Answer generation phase
    response: Optional[str] = None
    citations: List[Citation] = field(default_factory=list)
    
    # Evaluation phase
    evaluation_score: Optional[float] = None
    
    # Workflow control
    iteration_count: int = 0
    max_iterations: int = 3
    
    # Messages for LangGraph compatibility
    messages: List[BaseMessage] = field(default_factory=list)
    
    # Additional metadata
    processing_start_time: Optional[datetime] = None
    processing_end_time: Optional[datetime] = None
    error_message: Optional[str] = None


class AgentOrchestrator:
    """
    Main orchestrator class that manages the multi-agent workflow using LangGraph.
    
    Coordinates the following agents:
    - Planner Agent: Analyzes queries and creates execution plans
    - Retriever Agent: Fetches relevant documents based on the plan
    - Answer Agent: Synthesizes responses using retrieved context
    - Evaluator Agent: Validates response quality and triggers re-retrieval if needed
    """
    
    def __init__(self):
        """Initialize the agent orchestrator"""
        self.workflow = None
        self._setup_workflow()
    
    def _setup_workflow(self) -> None:
        """Set up the LangGraph workflow with agent nodes and edges"""
        # Create the state graph
        workflow = StateGraph(AgentState)
        
        # Add agent nodes
        workflow.add_node("planner", self.plan_query)
        workflow.add_node("retriever", self.retrieve_documents)
        workflow.add_node("answer", self.generate_answer)
        workflow.add_node("evaluator", self.evaluate_response)
        
        # Define the workflow edges
        workflow.set_entry_point("planner")
        
        # Planner -> Retriever
        workflow.add_edge("planner", "retriever")
        
        # Retriever -> Answer
        workflow.add_edge("retriever", "answer")
        
        # Answer -> Evaluator
        workflow.add_edge("answer", "evaluator")
        
        # Evaluator decision logic (will be implemented in individual agent tasks)
        workflow.add_conditional_edges(
            "evaluator",
            self._should_continue,
            {
                "continue": "retriever",  # Re-retrieve if quality is insufficient
                "end": END
            }
        )
        
        # Compile the workflow
        self.workflow = workflow.compile()
        logger.info("LangGraph workflow initialized successfully")
    
    def create_workflow(self) -> StateGraph:
        """
        Create and return the LangGraph workflow.
        
        Returns:
            StateGraph: Compiled LangGraph workflow
        """
        return self.workflow
    
    def plan_query(self, state: AgentState) -> AgentState:
        """
        Planner Agent: Analyzes user queries and decomposes complex questions.
        
        Implements query analysis, decomposition, and retrieval strategy planning
        according to requirement 3.1.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state with planning information
        """
        logger.info(f"Planning query: {state.query}")
        
        # Initialize processing timestamp
        if not state.processing_start_time:
            state.processing_start_time = datetime.now()
        
        try:
            # Analyze query complexity and intent
            query_analysis = self._analyze_query_complexity(state.query)
            
            # Decompose complex queries into sub-questions
            decomposition = self._decompose_query(state.query, query_analysis)
            
            # Plan retrieval strategy based on query type
            retrieval_strategy = self._plan_retrieval_strategy(state.query, query_analysis)
            
            # Generate execution plan
            execution_plan = self._generate_execution_plan(
                state.query, 
                query_analysis, 
                decomposition, 
                retrieval_strategy
            )
            
            # Update state with planning results
            state.plan = execution_plan
            
            logger.info(f"Query plan created successfully: {execution_plan}")
            return state
            
        except Exception as e:
            logger.error(f"Error in query planning: {str(e)}")
            state.error_message = f"Planning error: {str(e)}"
            return state
    
    def retrieve_documents(self, state: AgentState) -> AgentState:
        """
        Retriever Agent: Fetches relevant documents based on the query plan.
        
        Implements requirement 3.2: Connect to RAG pipeline for document retrieval,
        add query reformulation capabilities, and implement retrieval result processing.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state with retrieved documents
        """
        logger.info(f"Retrieving documents for query: {state.query}")
        
        try:
            # Import RAG pipeline components
            from app.services.rag_pipeline import RAGEngine, get_rag_engine
            
            # Get RAG engine instance
            try:
                rag_engine = get_rag_engine()
            except (ImportError, AttributeError):
                # Fallback to creating new instance if factory function not available
                rag_engine = RAGEngine()
            
            # Extract retrieval parameters from plan if available
            retrieval_params = self._extract_retrieval_params_from_plan(state.plan)
            
            # Perform query reformulation if needed
            reformulated_queries = self._reformulate_query(state.query, state.plan)
            
            # Retrieve documents for each query variant
            all_retrieved_chunks = []
            
            for query_variant in reformulated_queries:
                logger.info(f"Retrieving for query variant: '{query_variant[:100]}...'")
                
                # Use RAG engine to retrieve and rerank documents
                ranked_chunks = rag_engine.retrieve_and_rerank(
                    query=query_variant,
                    retrieval_k=retrieval_params.get('retrieval_k', 50),
                    final_k=retrieval_params.get('final_k', 20)
                )
                
                # Convert RankedChunk to our AgentState RankedChunk format
                for chunk in ranked_chunks:
                    agent_chunk = RankedChunk(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        content=chunk.content,
                        page_number=chunk.page_number,
                        start_char=chunk.metadata.get('start_char', 0),
                        end_char=chunk.metadata.get('end_char', len(chunk.content)),
                        metadata=chunk.metadata,
                        score=chunk.final_score,
                        retrieval_method='hybrid',
                        rank=0,  # Will be set during processing
                        rerank_score=chunk.relevance_score,
                        vector_score=chunk.vector_score,
                        bm25_score=chunk.bm25_score,
                        combined_score=chunk.combined_score,
                        relevance_score=chunk.relevance_score,
                        final_score=chunk.final_score
                    )
                    all_retrieved_chunks.append(agent_chunk)
            
            # Process and deduplicate results
            processed_chunks = self._process_retrieval_results(
                all_retrieved_chunks, 
                state.query,
                retrieval_params
            )
            
            # Update state with retrieved chunks
            state.retrieved_chunks = processed_chunks
            
            # Calculate and log retrieval quality metrics
            retrieval_metrics = self._calculate_retrieval_metrics(processed_chunks, state.query)
            logger.info(f"Retrieval metrics: {retrieval_metrics}")
            
            logger.info(f"Retrieved {len(state.retrieved_chunks)} document chunks after processing")
            return state
            
        except Exception as e:
            logger.error(f"Error in document retrieval: {str(e)}")
            state.error_message = f"Retrieval error: {str(e)}"
            state.retrieved_chunks = []
            return state
    
    def generate_answer(self, state: AgentState) -> AgentState:
        """
        Answer Agent: Synthesizes responses using retrieved context.
        
        Implements requirement 3.3: Generate grounded responses using retrieved context,
        add grounding verification to source documents, and implement citation formatting.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state with generated response and citations
        """
        logger.info(f"Generating answer for query: {state.query}")
        
        try:
            # Check if we have retrieved chunks to work with
            if not state.retrieved_chunks:
                logger.warning("No retrieved chunks available for answer generation")
                state.response = "I don't have enough information to answer your question. Please try rephrasing your query or ensure that relevant documents have been uploaded to the system."
                state.citations = []
                return state
            
            # Step 1: Prepare context from retrieved chunks
            context_info = self._prepare_context_for_synthesis(state.retrieved_chunks, state.query)
            
            # Step 2: Generate response using LLM with context
            response_text = self._synthesize_response_with_llm(
                query=state.query,
                context=context_info['combined_context'],
                plan=state.plan
            )
            
            # Step 3: Verify grounding and extract citations
            grounding_result = self._verify_response_grounding(
                response=response_text,
                retrieved_chunks=state.retrieved_chunks,
                query=state.query
            )
            
            # Step 4: Format final response with citations
            final_response = self._format_response_with_citations(
                response=grounding_result['verified_response'],
                citations=grounding_result['citations']
            )
            
            # Update state with results
            state.response = final_response
            state.citations = grounding_result['citations']
            
            logger.info(f"Answer generated successfully with {len(state.citations)} citations")
            return state
            
        except Exception as e:
            logger.error(f"Error in answer generation: {str(e)}")
            state.error_message = f"Answer generation error: {str(e)}"
            state.response = "I encountered an error while generating the answer. Please try again."
            state.citations = []
            return state
    
    def evaluate_response(self, state: AgentState) -> AgentState:
        """
        Evaluator Agent: Validates response quality and detects hallucinations.
        
        Implements requirement 3.4: Check responses for hallucinations and accuracy.
        Creates response quality assessment, adds hallucination detection, and 
        implements re-retrieval triggering logic.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state with evaluation score
        """
        logger.info("Evaluating response quality and detecting hallucinations")
        
        try:
            # Step 1: Initialize evaluation metrics
            evaluation_metrics = {
                'response_quality': 0.0,
                'hallucination_score': 0.0,
                'grounding_score': 0.0,
                'completeness_score': 0.0,
                'citation_accuracy': 0.0,
                'overall_score': 0.0
            }
            
            # Step 2: Assess response quality
            if state.response and state.retrieved_chunks:
                evaluation_metrics = self._assess_response_quality(
                    state.response, 
                    state.retrieved_chunks, 
                    state.query,
                    state.citations
                )
            elif not state.response:
                logger.warning("No response to evaluate")
                evaluation_metrics['overall_score'] = 0.0
            elif not state.retrieved_chunks:
                logger.warning("No retrieved chunks available for evaluation")
                evaluation_metrics['overall_score'] = 0.3  # Low score for no context
            
            # Step 3: Detect hallucinations using uncertainty handler
            hallucination_analysis = self._detect_hallucinations(
                state.response, 
                state.retrieved_chunks, 
                state.query
            )
            
            # Update evaluation metrics with hallucination analysis
            evaluation_metrics['hallucination_score'] = hallucination_analysis['hallucination_risk']
            evaluation_metrics['grounding_score'] = hallucination_analysis['grounding_confidence']
            
            # Step 4: Calculate final evaluation score
            final_score = self._calculate_final_evaluation_score(evaluation_metrics)
            
            # Step 5: Determine if re-retrieval is needed
            should_re_retrieve = self._should_trigger_re_retrieval(
                final_score, 
                evaluation_metrics, 
                state.iteration_count, 
                state.max_iterations
            )
            
            # Step 6: Update state with evaluation results
            state.evaluation_score = final_score
            state.iteration_count += 1
            state.processing_end_time = datetime.now()
            
            # Log evaluation results
            logger.info(f"Response evaluation completed:")
            logger.info(f"  - Overall Score: {final_score:.3f}")
            logger.info(f"  - Response Quality: {evaluation_metrics['response_quality']:.3f}")
            logger.info(f"  - Hallucination Risk: {evaluation_metrics['hallucination_score']:.3f}")
            logger.info(f"  - Grounding Score: {evaluation_metrics['grounding_score']:.3f}")
            logger.info(f"  - Citation Accuracy: {evaluation_metrics['citation_accuracy']:.3f}")
            logger.info(f"  - Should Re-retrieve: {should_re_retrieve}")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in response evaluation: {str(e)}")
            state.error_message = f"Evaluation error: {str(e)}"
            state.evaluation_score = 0.5  # Neutral score on error
            state.iteration_count += 1
            state.processing_end_time = datetime.now()
            return state
    
    def _should_continue(self, state: AgentState) -> str:
        """
        Decision function to determine if the workflow should continue or end.
        
        Uses the Evaluator Agent's assessment to determine if re-retrieval is needed.
        
        Args:
            state: Current agent state
            
        Returns:
            "continue" if re-retrieval is needed, "end" if workflow should complete
        """
        try:
            # Check if we have an evaluation score
            if state.evaluation_score is None:
                logger.warning("No evaluation score available, ending workflow")
                return "end"
            
            # Check if we've reached max iterations
            if state.iteration_count >= state.max_iterations:
                logger.info(f"Max iterations ({state.max_iterations}) reached, ending workflow")
                return "end"
            
            # Use evaluation score to determine continuation
            # Lower threshold for re-retrieval to ensure quality
            if state.evaluation_score < 0.7:
                logger.info(f"Evaluation score {state.evaluation_score:.3f} below threshold (0.7), continuing for re-retrieval")
                return "continue"
            
            logger.info(f"Evaluation score {state.evaluation_score:.3f} acceptable, ending workflow")
            return "end"
            
        except Exception as e:
            logger.error(f"Error in workflow continuation decision: {e}")
            return "end"  # Conservative approach: end on error
    
    def _extract_retrieval_params_from_plan(self, plan: Optional[str]) -> Dict[str, Any]:
        """
        Extract retrieval parameters from the execution plan.
        
        Args:
            plan: Execution plan string from planner agent
            
        Returns:
            Dictionary with retrieval parameters
        """
        default_params = {
            'retrieval_k': 50,
            'final_k': 20,
            'vector_weight': 0.7,
            'keyword_weight': 0.3,
            'use_reranking': True,
            'context_strategy': 'greedy_relevance'
        }
        
        if not plan:
            return default_params
        
        params = default_params.copy()
        
        try:
            # Extract top_k from plan
            if "Retrieve top" in plan:
                import re
                match = re.search(r"Retrieve top (\d+) chunks", plan)
                if match:
                    params['retrieval_k'] = int(match.group(1))
            
            # Extract re-rank parameter
            if "Re-rank to top" in plan:
                import re
                match = re.search(r"Re-rank to top (\d+) chunks", plan)
                if match:
                    params['final_k'] = int(match.group(1))
            
            # Extract vector and keyword weights
            if "Vector weight:" in plan:
                import re
                match = re.search(r"Vector weight: ([\d.]+)", plan)
                if match:
                    params['vector_weight'] = float(match.group(1))
            
            if "Keyword weight:" in plan:
                import re
                match = re.search(r"Keyword weight: ([\d.]+)", plan)
                if match:
                    params['keyword_weight'] = float(match.group(1))
            
            logger.info(f"Extracted retrieval parameters from plan: {params}")
            
        except Exception as e:
            logger.warning(f"Error extracting parameters from plan: {e}, using defaults")
        
        return params
    
    def _reformulate_query(self, original_query: str, plan: Optional[str]) -> List[str]:
        """
        Reformulate the query to improve retrieval effectiveness.
        
        Implements query reformulation capabilities as required by task 6.3.
        
        Args:
            original_query: Original user query
            plan: Execution plan from planner agent
            
        Returns:
            List of query variants for retrieval
        """
        query_variants = [original_query]  # Always include original query
        
        try:
            # Extract sub-questions from plan if available
            if plan and "Sub-question" in plan:
                import re
                sub_questions = re.findall(r"Sub-question \d+: (.+)", plan)
                for sub_q in sub_questions:
                    if sub_q.strip() and sub_q.strip() != original_query:
                        query_variants.append(sub_q.strip())
            
            # Generate keyword-focused variants
            keyword_variant = self._generate_keyword_variant(original_query)
            if keyword_variant and keyword_variant not in query_variants:
                query_variants.append(keyword_variant)
            
            # Generate expanded variants for complex queries
            if len(original_query.split()) > 5:  # Complex query
                expanded_variant = self._generate_expanded_variant(original_query)
                if expanded_variant and expanded_variant not in query_variants:
                    query_variants.append(expanded_variant)
            
            # Limit to maximum 4 variants to avoid over-retrieval
            query_variants = query_variants[:4]
            
            logger.info(f"Generated {len(query_variants)} query variants for retrieval")
            for i, variant in enumerate(query_variants):
                logger.debug(f"Query variant {i+1}: {variant}")
            
        except Exception as e:
            logger.warning(f"Error in query reformulation: {e}, using original query only")
            query_variants = [original_query]
        
        return query_variants
    
    def _generate_keyword_variant(self, query: str) -> Optional[str]:
        """
        Generate a keyword-focused variant of the query.
        
        Args:
            query: Original query
            
        Returns:
            Keyword-focused query variant or None
        """
        try:
            # Remove question words and focus on key terms
            question_words = {"what", "how", "why", "when", "where", "who", "which", "can", "does", "is", "are", "do"}
            words = query.lower().split()
            
            # Filter out question words and common words
            common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
            keywords = [word for word in words if word not in question_words and word not in common_words and len(word) > 2]
            
            if len(keywords) >= 2:
                # Create keyword-focused query
                keyword_query = " ".join(keywords[:5])  # Limit to top 5 keywords
                return keyword_query
            
        except Exception as e:
            logger.warning(f"Error generating keyword variant: {e}")
        
        return None
    
    def _generate_expanded_variant(self, query: str) -> Optional[str]:
        """
        Generate an expanded variant with related terms.
        
        Args:
            query: Original query
            
        Returns:
            Expanded query variant or None
        """
        try:
            # Simple expansion by adding related terms based on query type
            query_lower = query.lower()
            
            # Add domain-specific terms
            if any(term in query_lower for term in ["machine learning", "ml", "ai", "artificial intelligence"]):
                return f"{query} algorithms models training data"
            elif any(term in query_lower for term in ["python", "programming", "code"]):
                return f"{query} syntax functions libraries development"
            elif any(term in query_lower for term in ["database", "sql", "data"]):
                return f"{query} tables queries storage management"
            elif any(term in query_lower for term in ["network", "networking", "internet"]):
                return f"{query} protocols connectivity infrastructure"
            else:
                # Generic expansion
                return f"{query} overview concepts principles"
                
        except Exception as e:
            logger.warning(f"Error generating expanded variant: {e}")
        
        return None
    
    def _process_retrieval_results(self, 
                                 chunks: List[RankedChunk], 
                                 original_query: str,
                                 retrieval_params: Dict[str, Any]) -> List[RankedChunk]:
        """
        Process and optimize retrieval results.
        
        Implements retrieval result processing as required by task 6.3.
        
        Args:
            chunks: Raw retrieved chunks from all query variants
            original_query: Original user query
            retrieval_params: Retrieval parameters
            
        Returns:
            Processed and optimized list of chunks
        """
        if not chunks:
            return []
        
        try:
            # Step 1: Deduplicate chunks by chunk_id
            unique_chunks = {}
            for chunk in chunks:
                if chunk.chunk_id not in unique_chunks:
                    unique_chunks[chunk.chunk_id] = chunk
                else:
                    # Keep the chunk with higher relevance score
                    if chunk.relevance_score > unique_chunks[chunk.chunk_id].relevance_score:
                        unique_chunks[chunk.chunk_id] = chunk
            
            deduplicated_chunks = list(unique_chunks.values())
            logger.info(f"Deduplicated {len(chunks)} chunks to {len(deduplicated_chunks)} unique chunks")
            
            # Step 2: Re-score chunks based on original query relevance
            rescored_chunks = self._rescore_chunks_for_original_query(
                deduplicated_chunks, original_query
            )
            
            # Step 3: Apply diversity filtering if many results
            if len(rescored_chunks) > retrieval_params.get('final_k', 20):
                filtered_chunks = self._apply_diversity_filtering(
                    rescored_chunks, 
                    target_count=retrieval_params.get('final_k', 20)
                )
            else:
                filtered_chunks = rescored_chunks
            
            # Step 4: Sort by final relevance score and set ranks
            filtered_chunks.sort(key=lambda x: x.final_score, reverse=True)
            
            # Set rank field
            for i, chunk in enumerate(filtered_chunks):
                chunk.rank = i + 1
            
            logger.info(f"Processed retrieval results: {len(filtered_chunks)} final chunks")
            return filtered_chunks
            
        except Exception as e:
            logger.error(f"Error processing retrieval results: {e}")
            # Return original chunks sorted by relevance as fallback
            chunks.sort(key=lambda x: x.relevance_score, reverse=True)
            return chunks[:retrieval_params.get('final_k', 20)]
    
    def _rescore_chunks_for_original_query(self, 
                                         chunks: List[RankedChunk], 
                                         original_query: str) -> List[RankedChunk]:
        """
        Re-score chunks based on relevance to the original query.
        
        Args:
            chunks: List of chunks to re-score
            original_query: Original user query
            
        Returns:
            List of chunks with updated scores
        """
        try:
            # Simple keyword-based relevance scoring
            query_words = set(original_query.lower().split())
            
            for chunk in chunks:
                content_words = set(chunk.content.lower().split())
                
                # Calculate keyword overlap score
                if query_words:
                    overlap_score = len(query_words & content_words) / len(query_words)
                else:
                    overlap_score = 0.0
                
                # Combine with existing relevance score (weighted average)
                # 70% original relevance, 30% keyword overlap
                chunk.final_score = 0.7 * chunk.relevance_score + 0.3 * overlap_score
            
            logger.debug(f"Re-scored {len(chunks)} chunks for original query relevance")
            
        except Exception as e:
            logger.warning(f"Error re-scoring chunks: {e}")
            # Keep original scores as fallback
            for chunk in chunks:
                chunk.final_score = chunk.relevance_score
        
        return chunks
    
    def _apply_diversity_filtering(self, 
                                 chunks: List[RankedChunk], 
                                 target_count: int) -> List[RankedChunk]:
        """
        Apply diversity filtering to avoid redundant chunks.
        
        Args:
            chunks: List of chunks to filter
            target_count: Target number of chunks to return
            
        Returns:
            Filtered list of diverse chunks
        """
        if len(chunks) <= target_count:
            return chunks
        
        try:
            # Sort by relevance score first
            chunks.sort(key=lambda x: x.final_score, reverse=True)
            
            selected_chunks = [chunks[0]]  # Always include the best chunk
            
            for chunk in chunks[1:]:
                if len(selected_chunks) >= target_count:
                    break
                
                # Check if chunk is sufficiently different from selected chunks
                is_diverse = True
                for selected_chunk in selected_chunks:
                    similarity = self._calculate_chunk_similarity(chunk, selected_chunk)
                    if similarity > 0.8:  # Too similar
                        is_diverse = False
                        break
                
                if is_diverse:
                    selected_chunks.append(chunk)
            
            # If we don't have enough diverse chunks, fill with remaining high-scoring chunks
            while len(selected_chunks) < target_count and len(selected_chunks) < len(chunks):
                for chunk in chunks:
                    if chunk not in selected_chunks:
                        selected_chunks.append(chunk)
                        break
            
            logger.info(f"Applied diversity filtering: {len(chunks)} -> {len(selected_chunks)} chunks")
            return selected_chunks
            
        except Exception as e:
            logger.warning(f"Error applying diversity filtering: {e}")
            return chunks[:target_count]
    
    def _calculate_chunk_similarity(self, chunk1: RankedChunk, chunk2: RankedChunk) -> float:
        """
        Calculate similarity between two chunks.
        
        Args:
            chunk1: First chunk
            chunk2: Second chunk
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        try:
            # Simple word overlap similarity
            words1 = set(chunk1.content.lower().split())
            words2 = set(chunk2.content.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating chunk similarity: {e}")
            return 0.0
    
    def _calculate_retrieval_metrics(self, 
                                   chunks: List[RankedChunk], 
                                   query: str) -> Dict[str, Any]:
        """
        Calculate metrics for retrieval quality assessment.
        
        Args:
            chunks: Retrieved chunks
            query: Original query
            
        Returns:
            Dictionary with retrieval metrics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_relevance_score": 0.0,
                "document_diversity": 0,
                "query_coverage": 0.0
            }
        
        try:
            # Basic metrics
            total_chunks = len(chunks)
            avg_relevance = sum(chunk.final_score for chunk in chunks) / total_chunks
            
            # Document diversity
            unique_documents = len(set(chunk.document_id for chunk in chunks))
            document_diversity = unique_documents / total_chunks if total_chunks > 0 else 0
            
            # Query coverage (keyword overlap)
            query_words = set(query.lower().split())
            all_content = " ".join(chunk.content for chunk in chunks)
            content_words = set(all_content.lower().split())
            
            if query_words:
                query_coverage = len(query_words & content_words) / len(query_words)
            else:
                query_coverage = 0.0
            
            return {
                "total_chunks": total_chunks,
                "avg_relevance_score": avg_relevance,
                "document_diversity": document_diversity,
                "query_coverage": query_coverage,
                "unique_documents": unique_documents
            }
            
        except Exception as e:
            logger.warning(f"Error calculating retrieval metrics: {e}")
            return {
                "total_chunks": len(chunks),
                "avg_relevance_score": 0.0,
                "document_diversity": 0.0,
                "query_coverage": 0.0
            }
    
    def _prepare_context_for_synthesis(self, chunks: List[RankedChunk], query: str) -> Dict[str, Any]:
        """
        Prepare context from retrieved chunks for response synthesis.
        
        Args:
            chunks: List of retrieved and ranked chunks
            query: Original user query
            
        Returns:
            Dictionary with prepared context information
        """
        if not chunks:
            return {
                'combined_context': '',
                'chunk_summaries': [],
                'total_tokens': 0,
                'source_documents': set()
            }
        
        try:
            # Sort chunks by relevance score to prioritize best content
            sorted_chunks = sorted(chunks, key=lambda x: x.final_score, reverse=True)
            
            # Prepare context sections
            context_sections = []
            chunk_summaries = []
            source_documents = set()
            total_tokens = 0
            
            for i, chunk in enumerate(sorted_chunks[:10]):  # Limit to top 10 chunks
                # Create context section with source information
                section = f"[Source {i+1}: {chunk.metadata.filename}, Page {chunk.page_number}]\n{chunk.content.strip()}\n"
                context_sections.append(section)
                
                # Track chunk summary for citation purposes
                chunk_summaries.append({
                    'chunk_id': chunk.chunk_id,
                    'document_id': chunk.document_id,
                    'filename': chunk.metadata.filename,
                    'page_number': chunk.page_number,
                    'content_preview': chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                    'relevance_score': chunk.final_score,
                    'source_index': i + 1
                })
                
                # Track unique source documents
                source_documents.add(chunk.document_id)
                
                # Estimate token count (rough approximation: 1 token ≈ 4 characters)
                total_tokens += len(section) // 4
            
            # Combine all context sections
            combined_context = "\n".join(context_sections)
            
            logger.info(f"Prepared context from {len(sorted_chunks)} chunks, {len(source_documents)} unique documents")
            
            return {
                'combined_context': combined_context,
                'chunk_summaries': chunk_summaries,
                'total_tokens': total_tokens,
                'source_documents': source_documents
            }
            
        except Exception as e:
            logger.error(f"Error preparing context for synthesis: {e}")
            return {
                'combined_context': '',
                'chunk_summaries': [],
                'total_tokens': 0,
                'source_documents': set()
            }
    
    def _synthesize_response_with_llm(self, query: str, context: str, plan: Optional[str] = None) -> str:
        """
        Synthesize response using LLM with provided context.
        
        Args:
            query: User query
            context: Prepared context from retrieved documents
            plan: Optional execution plan from planner agent
            
        Returns:
            Generated response text
        """
        try:
            # Import LLM client
            from app.services.llm_client import DeepSeekClient
            
            # Create LLM client
            llm_client = DeepSeekClient()
            
            # Create system prompt for answer synthesis
            system_prompt = self._create_answer_synthesis_prompt(plan)
            
            # Create user message with context and query
            user_message = self._create_user_message_with_context(query, context)
            
            # Prepare messages for LLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # Generate response using LLM
            import asyncio
            try:
                # Try to get the current event loop
                loop = asyncio.get_running_loop()
                response = loop.run_until_complete(llm_client.generate_response(messages))
            except RuntimeError:
                # No running loop, create a new one
                response = asyncio.run(llm_client.generate_response(messages))
            
            logger.info("Response synthesized successfully using LLM")
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error synthesizing response with LLM: {e}")
            # Fallback to simple context-based response
            return self._create_fallback_response(query, context)
    
    def _create_answer_synthesis_prompt(self, plan: Optional[str] = None) -> str:
        """
        Create system prompt for answer synthesis.
        
        Args:
            plan: Optional execution plan from planner agent
            
        Returns:
            System prompt for LLM
        """
        base_prompt = """You are an intelligent assistant that provides accurate, well-grounded answers based on provided context from enterprise documents.

INSTRUCTIONS:
1. Answer the user's question using ONLY the information provided in the context
2. Be comprehensive but concise in your response
3. If the context doesn't contain sufficient information, clearly state what information is missing
4. Use specific details and examples from the context when relevant
5. Maintain a professional, informative tone
6. Structure your response logically with clear explanations
7. Reference specific sources when making claims (e.g., "According to Source 1...")

GROUNDING REQUIREMENTS:
- Every factual claim must be supported by the provided context
- Do not add information not present in the context
- If you're uncertain about something, express that uncertainty
- Clearly distinguish between what is explicitly stated vs. what can be reasonably inferred

CITATION GUIDELINES:
- Reference sources using the format provided in the context (e.g., "Source 1", "Source 2")
- When making specific claims, indicate which source supports that claim
- If multiple sources support the same point, mention all relevant sources"""
        
        # Add plan-specific guidance if available
        if plan and "comparison" in plan.lower():
            base_prompt += "\n\nSPECIAL INSTRUCTIONS: This appears to be a comparison query. Structure your response to clearly compare and contrast the different aspects mentioned in the query, using evidence from the provided sources."
        elif plan and "procedural" in plan.lower():
            base_prompt += "\n\nSPECIAL INSTRUCTIONS: This appears to be a procedural query. Structure your response as a clear, step-by-step explanation using the information from the provided sources."
        elif plan and "definition" in plan.lower():
            base_prompt += "\n\nSPECIAL INSTRUCTIONS: This appears to be a definition query. Provide a clear, comprehensive definition followed by relevant details and examples from the sources."
        
        return base_prompt
    
    def _create_user_message_with_context(self, query: str, context: str) -> str:
        """
        Create user message combining query and context.
        
        Args:
            query: User query
            context: Prepared context from documents
            
        Returns:
            Formatted user message
        """
        if not context.strip():
            return f"""I need to answer this question but no relevant context was found:

Question: {query}

Please provide a response indicating that insufficient information is available to answer the question."""
        
        return f"""Context from enterprise documents:

{context}

Question: {query}

Please provide a comprehensive answer based on the context above. Remember to reference the sources when making specific claims."""
    
    def _create_fallback_response(self, query: str, context: str) -> str:
        """
        Create a fallback response when LLM is unavailable.
        
        Args:
            query: User query
            context: Available context
            
        Returns:
            Fallback response text
        """
        if not context.strip():
            return "I don't have enough information in the available documents to answer your question. Please try rephrasing your query or ensure that relevant documents have been uploaded."
        
        # Simple fallback: return context with basic formatting
        return f"""Based on the available documents, here is the relevant information for your query "{query}":

{context}

Note: This is a simplified response due to system limitations. For a more comprehensive analysis, please try again later."""
    
    def _verify_response_grounding(self, response: str, retrieved_chunks: List[RankedChunk], query: str) -> Dict[str, Any]:
        """
        Verify that the response is properly grounded in the source documents.
        
        Args:
            response: Generated response text
            retrieved_chunks: Original retrieved chunks
            query: User query
            
        Returns:
            Dictionary with verified response and citations
        """
        try:
            # Extract source references from the response
            source_references = self._extract_source_references(response)
            
            # Create citations based on source references and chunks
            citations = self._create_citations_from_references(source_references, retrieved_chunks)
            
            # Verify grounding by checking if claims are supported
            grounding_score = self._calculate_grounding_score(response, retrieved_chunks)
            
            # Clean up response if needed (remove or fix unsupported claims)
            verified_response = self._clean_response_for_grounding(response, grounding_score)
            
            logger.info(f"Response grounding verified with score: {grounding_score:.2f}")
            
            return {
                'verified_response': verified_response,
                'citations': citations,
                'grounding_score': grounding_score,
                'source_references': source_references
            }
            
        except Exception as e:
            logger.error(f"Error verifying response grounding: {e}")
            # Fallback: create basic citations from all chunks
            fallback_citations = self._create_fallback_citations(retrieved_chunks[:5])  # Top 5 chunks
            return {
                'verified_response': response,
                'citations': fallback_citations,
                'grounding_score': 0.5,  # Neutral score
                'source_references': []
            }
    
    def _extract_source_references(self, response: str) -> List[Dict[str, Any]]:
        """
        Extract source references from the generated response.
        
        Args:
            response: Generated response text
            
        Returns:
            List of source reference information
        """
        import re
        
        source_references = []
        
        try:
            # Look for patterns like "Source 1", "Source 2", etc.
            source_pattern = r'Source (\d+)'
            matches = re.finditer(source_pattern, response, re.IGNORECASE)
            
            for match in matches:
                source_num = int(match.group(1))
                start_pos = match.start()
                end_pos = match.end()
                
                # Extract surrounding context (50 chars before and after)
                context_start = max(0, start_pos - 50)
                context_end = min(len(response), end_pos + 50)
                context = response[context_start:context_end]
                
                source_references.append({
                    'source_number': source_num,
                    'position': start_pos,
                    'context': context,
                    'match_text': match.group(0)
                })
            
            # Remove duplicates based on source number
            unique_refs = {}
            for ref in source_references:
                source_num = ref['source_number']
                if source_num not in unique_refs:
                    unique_refs[source_num] = ref
            
            logger.info(f"Extracted {len(unique_refs)} unique source references from response")
            return list(unique_refs.values())
            
        except Exception as e:
            logger.warning(f"Error extracting source references: {e}")
            return []
    
    def _create_citations_from_references(self, source_references: List[Dict[str, Any]], retrieved_chunks: List[RankedChunk]) -> List[Citation]:
        """
        Create citation objects from source references and retrieved chunks.
        
        Args:
            source_references: Extracted source references from response
            retrieved_chunks: Original retrieved chunks
            
        Returns:
            List of Citation objects
        """
        citations = []
        
        try:
            # Sort chunks by relevance for consistent source numbering
            sorted_chunks = sorted(retrieved_chunks, key=lambda x: x.final_score, reverse=True)
            
            # Create citations for referenced sources
            for ref in source_references:
                source_num = ref['source_number']
                
                # Get the corresponding chunk (source numbers are 1-indexed)
                if 1 <= source_num <= len(sorted_chunks):
                    chunk = sorted_chunks[source_num - 1]
                    
                    citation = Citation(
                        document_id=chunk.document_id,
                        filename=chunk.metadata.filename,
                        page_number=chunk.page_number,
                        chunk_content=chunk.content[:300] + "..." if len(chunk.content) > 300 else chunk.content,
                        relevance_score=chunk.final_score
                    )
                    citations.append(citation)
            
            # If no source references found, create citations from top chunks
            if not citations and retrieved_chunks:
                logger.info("No source references found, creating citations from top chunks")
                top_chunks = sorted(retrieved_chunks, key=lambda x: x.final_score, reverse=True)[:3]
                
                for chunk in top_chunks:
                    citation = Citation(
                        document_id=chunk.document_id,
                        filename=chunk.metadata.filename,
                        page_number=chunk.page_number,
                        chunk_content=chunk.content[:300] + "..." if len(chunk.content) > 300 else chunk.content,
                        relevance_score=chunk.final_score
                    )
                    citations.append(citation)
            
            logger.info(f"Created {len(citations)} citations from source references")
            return citations
            
        except Exception as e:
            logger.error(f"Error creating citations from references: {e}")
            return []
    
    def _calculate_grounding_score(self, response: str, retrieved_chunks: List[RankedChunk]) -> float:
        """
        Calculate how well the response is grounded in the source documents.
        
        Args:
            response: Generated response text
            retrieved_chunks: Retrieved chunks used for context
            
        Returns:
            Grounding score between 0.0 and 1.0
        """
        try:
            if not response.strip() or not retrieved_chunks:
                return 0.0
            
            # Simple grounding score based on keyword overlap
            response_words = set(response.lower().split())
            
            # Remove common words that don't indicate grounding
            common_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
            }
            response_content_words = response_words - common_words
            
            if not response_content_words:
                return 0.0
            
            # Calculate overlap with source content
            total_overlap = 0
            total_source_words = 0
            
            for chunk in retrieved_chunks:
                chunk_words = set(chunk.content.lower().split()) - common_words
                total_source_words += len(chunk_words)
                
                # Calculate overlap between response and this chunk
                overlap = len(response_content_words & chunk_words)
                total_overlap += overlap
            
            # Calculate grounding score
            if total_source_words > 0:
                grounding_score = min(total_overlap / len(response_content_words), 1.0)
            else:
                grounding_score = 0.0
            
            # Bonus for source references
            if "Source" in response:
                grounding_score = min(grounding_score + 0.2, 1.0)
            
            return grounding_score
            
        except Exception as e:
            logger.warning(f"Error calculating grounding score: {e}")
            return 0.5  # Neutral score as fallback
    
    def _clean_response_for_grounding(self, response: str, grounding_score: float) -> str:
        """
        Clean up response to improve grounding if score is low.
        
        Args:
            response: Original response text
            grounding_score: Calculated grounding score
            
        Returns:
            Cleaned response text
        """
        # If grounding score is acceptable, return as-is
        if grounding_score >= 0.6:
            return response
        
        # For low grounding scores, add a disclaimer
        disclaimer = "\n\nNote: This response is based on the available documents. Some information may require additional sources for complete accuracy."
        
        return response + disclaimer
    
    def _create_fallback_citations(self, chunks: List[RankedChunk]) -> List[Citation]:
        """
        Create fallback citations when source reference extraction fails.
        
        Args:
            chunks: Retrieved chunks to create citations from
            
        Returns:
            List of Citation objects
        """
        citations = []
        
        try:
            for chunk in chunks:
                citation = Citation(
                    document_id=chunk.document_id,
                    filename=chunk.metadata.filename,
                    page_number=chunk.page_number,
                    chunk_content=chunk.content[:300] + "..." if len(chunk.content) > 300 else chunk.content,
                    relevance_score=chunk.final_score
                )
                citations.append(citation)
            
            return citations
            
        except Exception as e:
            logger.error(f"Error creating fallback citations: {e}")
            return []
    
    def _format_response_with_citations(self, response: str, citations: List[Citation]) -> str:
        """
        Format the final response with proper citations.
        
        Args:
            response: Generated response text
            citations: List of citations to include
            
        Returns:
            Formatted response with citations
        """
        if not citations:
            return response
        
        try:
            # Add citations section to the response
            formatted_response = response
            
            if not formatted_response.endswith('\n'):
                formatted_response += '\n'
            
            formatted_response += '\n**Sources:**\n'
            
            # Format each citation
            for i, citation in enumerate(citations, 1):
                citation_text = f"{i}. {citation.filename}, Page {citation.page_number}"
                
                # Add relevance score if it's meaningful
                if citation.relevance_score > 0:
                    citation_text += f" (Relevance: {citation.relevance_score:.2f})"
                
                formatted_response += f"{citation_text}\n"
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error formatting response with citations: {e}")
            return response  # Return original response if formatting fails
    
    def _analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """
        Analyze query complexity and intent to inform planning decisions.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary containing query analysis results
        """
        analysis = {
            "query_type": "simple",
            "complexity_score": 1,
            "intent": "factual",
            "requires_decomposition": False,
            "keywords": [],
            "entities": [],
            "question_words": []
        }
        
        query_lower = query.lower().strip()
        
        # Detect question words and intent
        question_words = ["what", "how", "why", "when", "where", "who", "which", "can", "does", "is", "are"]
        found_question_words = [word for word in question_words if word in query_lower.split()]
        analysis["question_words"] = found_question_words
        
        # Determine query type based on structure and content
        if any(word in query_lower for word in ["compare", "difference", "versus", "vs", "contrast"]):
            analysis["query_type"] = "comparison"
            analysis["complexity_score"] = 3
            analysis["requires_decomposition"] = True
        elif any(word in query_lower for word in ["explain", "describe", "how does", "what is"]):
            analysis["query_type"] = "explanation"
            analysis["complexity_score"] = 2
        elif any(word in query_lower for word in ["list", "enumerate", "what are", "types of"]):
            analysis["query_type"] = "enumeration"
            analysis["complexity_score"] = 2
        elif any(word in query_lower for word in ["step", "process", "procedure", "how to"]):
            analysis["query_type"] = "procedural"
            analysis["complexity_score"] = 3
            analysis["requires_decomposition"] = True
        elif "?" in query and len(query.split()) > 10:
            analysis["query_type"] = "complex"
            analysis["complexity_score"] = 3
            analysis["requires_decomposition"] = True
        
        # Extract keywords (simple approach - remove common words)
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        words = query_lower.replace("?", "").replace(".", "").split()
        analysis["keywords"] = [word for word in words if word not in common_words and len(word) > 2]
        
        # Determine intent
        if any(word in query_lower for word in ["define", "what is", "meaning"]):
            analysis["intent"] = "definition"
        elif any(word in query_lower for word in ["how", "process", "step"]):
            analysis["intent"] = "procedural"
        elif any(word in query_lower for word in ["why", "reason", "cause"]):
            analysis["intent"] = "causal"
        elif any(word in query_lower for word in ["compare", "difference", "better"]):
            analysis["intent"] = "comparative"
        else:
            analysis["intent"] = "factual"
        
        logger.info(f"Query analysis completed: {analysis}")
        return analysis
    
    def _decompose_query(self, query: str, analysis: Dict[str, Any]) -> List[str]:
        """
        Decompose complex queries into simpler sub-questions.
        
        Args:
            query: Original user query
            analysis: Query analysis results
            
        Returns:
            List of sub-questions for complex queries, or original query for simple ones
        """
        if not analysis["requires_decomposition"]:
            return [query]
        
        sub_questions = []
        query_type = analysis["query_type"]
        
        if query_type == "comparison":
            # Extract entities being compared - look for patterns like "X and Y", "X vs Y", "X versus Y"
            keywords = analysis["keywords"]
            
            # Try to find comparison patterns
            query_lower = query.lower()
            
            # Look for "X and Y" pattern
            if " and " in query_lower:
                parts = query_lower.split(" and ")
                if len(parts) >= 2:
                    # Extract the last word before "and" and first few words after "and"
                    before_and = parts[0].split()[-1] if parts[0].split() else ""
                    after_and = parts[1].split()[:2]  # Take first 2 words after "and"
                    
                    if before_and and after_and:
                        entity1 = before_and
                        entity2 = " ".join(after_and).replace("programming", "").replace("languages", "").strip()
                        
                        sub_questions = [
                            f"What is {entity1}?",
                            f"What is {entity2}?",
                            f"What are the key differences between {entity1} and {entity2}?"
                        ]
                    else:
                        # Fallback: use first few keywords
                        if len(keywords) >= 3:
                            sub_questions = [
                                f"What is {keywords[1]}?",  # Skip "compare"
                                f"What is {keywords[2]}?",
                                f"What are the key differences between {keywords[1]} and {keywords[2]}?"
                            ]
                        else:
                            sub_questions = [query]
                else:
                    sub_questions = [query]
            else:
                # Fallback for other comparison patterns
                if len(keywords) >= 3:
                    sub_questions = [
                        f"What is {keywords[1]}?",  # Skip "compare"
                        f"What is {keywords[2]}?",
                        f"What are the key differences between {keywords[1]} and {keywords[2]}?"
                    ]
                else:
                    sub_questions = [query]
        
        elif query_type == "procedural":
            # Break down procedural questions
            main_topic = " ".join(analysis["keywords"][:3])  # Use first few keywords
            sub_questions = [
                f"What is {main_topic}?",
                f"What are the main steps involved in {main_topic}?",
                f"What are the requirements or prerequisites for {main_topic}?"
            ]
        
        elif query_type == "complex":
            # For complex queries, try to identify main concepts
            keywords = analysis["keywords"][:3]  # Limit to top 3 keywords
            sub_questions = [f"What is {keyword}?" for keyword in keywords]
            sub_questions.append(query)  # Include original query as final step
        
        else:
            sub_questions = [query]  # Fallback for other types
        
        logger.info(f"Query decomposed into {len(sub_questions)} sub-questions: {sub_questions}")
        return sub_questions
    
    def _plan_retrieval_strategy(self, query: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plan the retrieval strategy based on query analysis.
        
        Args:
            query: User query string
            analysis: Query analysis results
            
        Returns:
            Dictionary containing retrieval strategy configuration
        """
        strategy = {
            "primary_method": "hybrid",  # Default to hybrid retrieval
            "vector_weight": 0.7,
            "keyword_weight": 0.3,
            "top_k": 10,
            "rerank_top_k": 5,
            "use_reranking": True,
            "search_keywords": analysis["keywords"],
            "query_expansion": False,
            "filter_criteria": None
        }
        
        query_type = analysis["query_type"]
        complexity_score = analysis["complexity_score"]
        
        # Adjust strategy based on query type
        if query_type == "definition":
            # For definitions, prioritize exact keyword matches
            strategy["keyword_weight"] = 0.5
            strategy["vector_weight"] = 0.5
            strategy["top_k"] = 5
        
        elif query_type == "procedural":
            # For procedural queries, get more context
            strategy["top_k"] = 15
            strategy["rerank_top_k"] = 8
            strategy["query_expansion"] = True
        
        elif query_type == "comparison":
            # For comparisons, need comprehensive retrieval
            strategy["top_k"] = 20
            strategy["rerank_top_k"] = 10
            strategy["vector_weight"] = 0.8  # Favor semantic similarity
            strategy["keyword_weight"] = 0.2
        
        elif query_type == "enumeration":
            # For lists/enumerations, favor keyword matching
            strategy["keyword_weight"] = 0.6
            strategy["vector_weight"] = 0.4
            strategy["top_k"] = 12
        
        # Adjust for complexity
        if complexity_score >= 3:
            strategy["top_k"] = min(strategy["top_k"] + 5, 25)  # Cap at 25
            strategy["rerank_top_k"] = min(strategy["rerank_top_k"] + 3, 12)  # Cap at 12
        
        logger.info(f"Retrieval strategy planned: {strategy}")
        return strategy
    
    def _generate_execution_plan(
        self, 
        query: str, 
        analysis: Dict[str, Any], 
        decomposition: List[str], 
        retrieval_strategy: Dict[str, Any]
    ) -> str:
        """
        Generate a comprehensive execution plan for the query.
        
        Args:
            query: Original user query
            analysis: Query analysis results
            decomposition: List of sub-questions
            retrieval_strategy: Retrieval strategy configuration
            
        Returns:
            Formatted execution plan string
        """
        plan_parts = []
        
        # Add query analysis summary
        plan_parts.append(f"QUERY ANALYSIS:")
        plan_parts.append(f"- Type: {analysis['query_type']}")
        plan_parts.append(f"- Complexity: {analysis['complexity_score']}/3")
        plan_parts.append(f"- Intent: {analysis['intent']}")
        plan_parts.append(f"- Keywords: {', '.join(analysis['keywords'][:5])}")  # Limit to 5 keywords
        
        # Add decomposition if applicable
        if len(decomposition) > 1:
            plan_parts.append(f"\nQUERY DECOMPOSITION:")
            for i, sub_q in enumerate(decomposition, 1):
                plan_parts.append(f"- Sub-question {i}: {sub_q}")
        
        # Add retrieval strategy
        plan_parts.append(f"\nRETRIEVAL STRATEGY:")
        plan_parts.append(f"- Method: {retrieval_strategy['primary_method']}")
        plan_parts.append(f"- Vector weight: {retrieval_strategy['vector_weight']}")
        plan_parts.append(f"- Keyword weight: {retrieval_strategy['keyword_weight']}")
        plan_parts.append(f"- Retrieve top {retrieval_strategy['top_k']} chunks")
        plan_parts.append(f"- Re-rank to top {retrieval_strategy['rerank_top_k']} chunks")
        
        # Add execution steps
        plan_parts.append(f"\nEXECUTION STEPS:")
        plan_parts.append("1. Execute retrieval strategy to find relevant documents")
        
        if len(decomposition) > 1:
            plan_parts.append("2. Process each sub-question systematically")
            plan_parts.append("3. Synthesize comprehensive answer addressing all aspects")
        else:
            plan_parts.append("2. Synthesize direct answer from retrieved context")
        
        plan_parts.append("4. Generate proper citations with source references")
        plan_parts.append("5. Evaluate response quality and completeness")
        
        execution_plan = "\n".join(plan_parts)
        logger.info("Execution plan generated successfully")
        return execution_plan
    
    def _assess_response_quality(self, 
                               response: str, 
                               retrieved_chunks: List[RankedChunk], 
                               query: str,
                               citations: List[Citation]) -> Dict[str, float]:
        """
        Assess the quality of the generated response.
        
        Args:
            response: Generated response text
            retrieved_chunks: Retrieved chunks used for context
            query: Original user query
            citations: Generated citations
            
        Returns:
            Dictionary with quality assessment scores
        """
        try:
            metrics = {
                'response_quality': 0.0,
                'hallucination_score': 0.0,
                'grounding_score': 0.0,
                'completeness_score': 0.0,
                'citation_accuracy': 0.0,
                'overall_score': 0.0
            }
            
            if not response or not response.strip():
                return metrics
            
            # 1. Response Quality Assessment
            metrics['response_quality'] = self._assess_response_content_quality(response, query)
            
            # 2. Grounding Assessment
            metrics['grounding_score'] = self._assess_response_grounding_quality(
                response, retrieved_chunks
            )
            
            # 3. Completeness Assessment
            metrics['completeness_score'] = self._assess_response_completeness(
                response, query, retrieved_chunks
            )
            
            # 4. Citation Accuracy Assessment
            metrics['citation_accuracy'] = self._assess_citation_accuracy(
                response, citations, retrieved_chunks
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error assessing response quality: {e}")
            return {
                'response_quality': 0.0,
                'hallucination_score': 0.0,
                'grounding_score': 0.0,
                'completeness_score': 0.0,
                'citation_accuracy': 0.0,
                'overall_score': 0.0
            }
    
    def _assess_response_content_quality(self, response: str, query: str) -> float:
        """
        Assess the content quality of the response.
        
        Args:
            response: Generated response text
            query: Original user query
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        try:
            quality_score = 0.0
            
            # Length appropriateness (not too short, not too long)
            response_length = len(response.strip())
            if 50 <= response_length <= 2000:
                length_score = 1.0
            elif response_length < 50:
                length_score = response_length / 50.0
            else:
                length_score = max(0.5, 2000.0 / response_length)
            
            quality_score += 0.2 * length_score
            
            # Query relevance (keyword overlap)
            query_words = set(query.lower().split())
            response_words = set(response.lower().split())
            
            if query_words:
                relevance_score = len(query_words & response_words) / len(query_words)
            else:
                relevance_score = 0.0
            
            quality_score += 0.3 * relevance_score
            
            # Structure quality (presence of clear sentences)
            sentences = response.split('.')
            sentence_count = len([s for s in sentences if len(s.strip()) > 10])
            structure_score = min(1.0, sentence_count / 5.0)  # Normalize to 5 sentences
            
            quality_score += 0.2 * structure_score
            
            # Information density (avoid repetitive content)
            unique_words = len(set(response.lower().split()))
            total_words = len(response.split())
            
            if total_words > 0:
                density_score = unique_words / total_words
            else:
                density_score = 0.0
            
            quality_score += 0.3 * density_score
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.warning(f"Error assessing response content quality: {e}")
            return 0.5
    
    def _assess_response_grounding_quality(self, 
                                         response: str, 
                                         retrieved_chunks: List[RankedChunk]) -> float:
        """
        Assess how well the response is grounded in the source documents.
        
        Args:
            response: Generated response text
            retrieved_chunks: Retrieved chunks used for context
            
        Returns:
            Grounding score between 0.0 and 1.0
        """
        try:
            if not response.strip() or not retrieved_chunks:
                return 0.0
            
            # Extract content words from response (excluding common words)
            common_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
            }
            
            response_words = set(response.lower().split()) - common_words
            
            if not response_words:
                return 0.0
            
            # Calculate overlap with source content
            total_overlap = 0
            total_source_words = 0
            
            for chunk in retrieved_chunks:
                chunk_words = set(chunk.content.lower().split()) - common_words
                total_source_words += len(chunk_words)
                
                # Calculate overlap between response and this chunk
                overlap = len(response_words & chunk_words)
                total_overlap += overlap
            
            # Calculate grounding score
            if len(response_words) > 0:
                grounding_score = min(total_overlap / len(response_words), 1.0)
            else:
                grounding_score = 0.0
            
            # Bonus for explicit source references
            if "Source" in response or "source" in response.lower():
                grounding_score = min(grounding_score + 0.2, 1.0)
            
            return grounding_score
            
        except Exception as e:
            logger.warning(f"Error assessing response grounding: {e}")
            return 0.5
    
    def _assess_response_completeness(self, 
                                    response: str, 
                                    query: str, 
                                    retrieved_chunks: List[RankedChunk]) -> float:
        """
        Assess how completely the response addresses the query.
        
        Args:
            response: Generated response text
            query: Original user query
            retrieved_chunks: Retrieved chunks used for context
            
        Returns:
            Completeness score between 0.0 and 1.0
        """
        try:
            if not response.strip():
                return 0.0
            
            # Query type analysis for completeness expectations
            query_lower = query.lower()
            
            # Different query types have different completeness requirements
            if any(word in query_lower for word in ["what is", "define", "definition"]):
                # Definition queries should have clear explanations
                completeness_score = self._assess_definition_completeness(response, query)
            elif any(word in query_lower for word in ["how", "process", "step"]):
                # Procedural queries should have step-by-step information
                completeness_score = self._assess_procedural_completeness(response, query)
            elif any(word in query_lower for word in ["compare", "difference", "versus"]):
                # Comparison queries should address multiple aspects
                completeness_score = self._assess_comparison_completeness(response, query)
            elif any(word in query_lower for word in ["list", "enumerate", "types"]):
                # Enumeration queries should provide multiple items
                completeness_score = self._assess_enumeration_completeness(response, query)
            else:
                # General factual queries
                completeness_score = self._assess_general_completeness(response, query)
            
            # Adjust based on available context
            if retrieved_chunks:
                context_richness = min(1.0, len(retrieved_chunks) / 5.0)  # Normalize to 5 chunks
                completeness_score = completeness_score * (0.7 + 0.3 * context_richness)
            
            return max(0.0, min(1.0, completeness_score))
            
        except Exception as e:
            logger.warning(f"Error assessing response completeness: {e}")
            return 0.5
    
    def _assess_definition_completeness(self, response: str, query: str) -> float:
        """Assess completeness for definition queries"""
        # Definition should have clear explanation and examples
        has_explanation = len(response.split('.')) >= 2
        has_examples = any(word in response.lower() for word in ["example", "such as", "including", "like"])
        
        score = 0.0
        if has_explanation:
            score += 0.7
        if has_examples:
            score += 0.3
        
        return score
    
    def _assess_procedural_completeness(self, response: str, query: str) -> float:
        """Assess completeness for procedural queries"""
        # Procedural should have steps or process information
        has_steps = any(word in response.lower() for word in ["step", "first", "then", "next", "finally"])
        has_process = any(word in response.lower() for word in ["process", "procedure", "method"])
        
        score = 0.0
        if has_steps:
            score += 0.6
        if has_process:
            score += 0.4
        
        return score
    
    def _assess_comparison_completeness(self, response: str, query: str) -> float:
        """Assess completeness for comparison queries"""
        # Comparison should address multiple aspects
        has_differences = any(word in response.lower() for word in ["difference", "differ", "unlike", "whereas"])
        has_similarities = any(word in response.lower() for word in ["similar", "both", "same", "common"])
        
        score = 0.0
        if has_differences:
            score += 0.6
        if has_similarities:
            score += 0.4
        
        return score
    
    def _assess_enumeration_completeness(self, response: str, query: str) -> float:
        """Assess completeness for enumeration queries"""
        # Enumeration should provide multiple items
        has_list_markers = any(marker in response for marker in ["1.", "2.", "•", "-", "first", "second"])
        item_count = len([s for s in response.split('.') if len(s.strip()) > 10])
        
        score = 0.0
        if has_list_markers:
            score += 0.5
        if item_count >= 3:
            score += 0.5
        
        return score
    
    def _assess_general_completeness(self, response: str, query: str) -> float:
        """Assess completeness for general queries"""
        # General queries should be informative and address the question
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        if query_words:
            coverage = len(query_words & response_words) / len(query_words)
        else:
            coverage = 0.0
        
        # Length-based completeness (reasonable response length)
        length_score = min(1.0, len(response) / 200.0)  # Normalize to 200 characters
        
        return 0.6 * coverage + 0.4 * length_score
    
    def _assess_citation_accuracy(self, 
                                response: str, 
                                citations: List[Citation], 
                                retrieved_chunks: List[RankedChunk]) -> float:
        """
        Assess the accuracy of citations in the response.
        
        Args:
            response: Generated response text
            citations: Generated citations
            retrieved_chunks: Retrieved chunks used for context
            
        Returns:
            Citation accuracy score between 0.0 and 1.0
        """
        try:
            if not citations:
                # If no citations but response references sources, penalize
                if "Source" in response or "source" in response.lower():
                    return 0.3  # Low score for missing citations
                else:
                    return 0.8  # Neutral score if no source references
            
            accuracy_score = 0.0
            
            # Check if citations match retrieved chunks
            chunk_ids = {chunk.chunk_id for chunk in retrieved_chunks}
            valid_citations = 0
            
            for citation in citations:
                # Check if citation corresponds to actual retrieved content
                matching_chunks = [
                    chunk for chunk in retrieved_chunks 
                    if (chunk.document_id == citation.document_id and 
                        chunk.page_number == citation.page_number)
                ]
                
                if matching_chunks:
                    valid_citations += 1
                    
                    # Check if citation content matches chunk content
                    for chunk in matching_chunks:
                        if citation.chunk_content.strip() in chunk.content:
                            accuracy_score += 0.2  # Bonus for accurate content match
            
            # Base accuracy from valid citations
            if citations:
                base_accuracy = valid_citations / len(citations)
            else:
                base_accuracy = 0.0
            
            # Combine base accuracy with content match bonuses
            final_accuracy = min(1.0, base_accuracy + accuracy_score)
            
            return final_accuracy
            
        except Exception as e:
            logger.warning(f"Error assessing citation accuracy: {e}")
            return 0.5
    
    def _detect_hallucinations(self, 
                             response: str, 
                             retrieved_chunks: List[RankedChunk], 
                             query: str) -> Dict[str, float]:
        """
        Detect potential hallucinations in the response using uncertainty analysis.
        
        Args:
            response: Generated response text
            retrieved_chunks: Retrieved chunks used for context
            query: Original user query
            
        Returns:
            Dictionary with hallucination analysis results
        """
        try:
            # Use the existing UncertaintyHandler for hallucination detection
            from app.services.rag_pipeline import UncertaintyHandler
            
            uncertainty_handler = UncertaintyHandler(
                min_confidence_threshold=0.6,
                min_context_sufficiency=0.5,
                max_hallucination_risk=0.3
            )
            
            # Calculate confidence metrics for the retrieved chunks
            confidence_metrics = uncertainty_handler.calculate_confidence_metrics(
                retrieved_chunks, query
            )
            
            # Additional hallucination detection based on response content
            response_hallucination_risk = self._analyze_response_hallucination_risk(
                response, retrieved_chunks, query
            )
            
            # Combine uncertainty handler results with response-specific analysis
            combined_hallucination_risk = (
                0.6 * confidence_metrics.hallucination_risk +
                0.4 * response_hallucination_risk
            )
            
            # Grounding confidence is inverse of hallucination risk
            grounding_confidence = 1.0 - combined_hallucination_risk
            
            return {
                'hallucination_risk': combined_hallucination_risk,
                'grounding_confidence': grounding_confidence,
                'retrieval_confidence': confidence_metrics.retrieval_confidence,
                'context_sufficiency': confidence_metrics.context_sufficiency,
                'uncertainty_indicators': confidence_metrics.uncertainty_indicators
            }
            
        except Exception as e:
            logger.error(f"Error detecting hallucinations: {e}")
            return {
                'hallucination_risk': 0.5,
                'grounding_confidence': 0.5,
                'retrieval_confidence': 0.5,
                'context_sufficiency': 0.5,
                'uncertainty_indicators': ["Error in hallucination detection"]
            }
    
    def _analyze_response_hallucination_risk(self, 
                                           response: str, 
                                           retrieved_chunks: List[RankedChunk], 
                                           query: str) -> float:
        """
        Analyze response-specific hallucination risk indicators.
        
        Args:
            response: Generated response text
            retrieved_chunks: Retrieved chunks used for context
            query: Original user query
            
        Returns:
            Hallucination risk score between 0.0 and 1.0
        """
        try:
            if not response or not response.strip():
                return 1.0  # High risk for empty response
            
            risk_score = 0.0
            
            # 1. Check for specific claims not supported by context
            response_claims = self._extract_factual_claims(response)
            unsupported_claims = self._identify_unsupported_claims(response_claims, retrieved_chunks)
            
            if response_claims:
                unsupported_ratio = len(unsupported_claims) / len(response_claims)
                risk_score += 0.4 * unsupported_ratio
            
            # 2. Check for overly confident language without strong evidence
            confident_phrases = [
                "definitely", "certainly", "always", "never", "absolutely", 
                "without doubt", "guaranteed", "proven fact"
            ]
            
            confident_count = sum(1 for phrase in confident_phrases if phrase in response.lower())
            if confident_count > 0:
                # High confidence language increases risk if context is weak
                avg_relevance = np.mean([chunk.relevance_score for chunk in retrieved_chunks]) if retrieved_chunks else 0.0
                if avg_relevance < 0.7:  # Weak context with confident language
                    risk_score += 0.2 * min(confident_count / 3.0, 1.0)
            
            # 3. Check for numerical claims or specific dates without source support
            import re
            numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', response)
            dates = re.findall(r'\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b', response)
            
            specific_claims = len(numbers) + len(dates)
            if specific_claims > 0:
                # Check if these specific claims are supported by context
                context_text = " ".join([chunk.content for chunk in retrieved_chunks])
                supported_claims = 0
                
                for number in numbers:
                    if number in context_text:
                        supported_claims += 1
                
                for date in dates:
                    if date in context_text:
                        supported_claims += 1
                
                if specific_claims > 0:
                    unsupported_specific_ratio = 1.0 - (supported_claims / specific_claims)
                    risk_score += 0.3 * unsupported_specific_ratio
            
            # 4. Check for contradictions with source material
            contradiction_risk = self._detect_contradictions(response, retrieved_chunks)
            risk_score += 0.1 * contradiction_risk
            
            return max(0.0, min(1.0, risk_score))
            
        except Exception as e:
            logger.warning(f"Error analyzing response hallucination risk: {e}")
            return 0.5
    
    def _extract_factual_claims(self, response: str) -> List[str]:
        """Extract factual claims from the response"""
        # Simple approach: split by sentences and filter for factual statements
        sentences = [s.strip() for s in response.split('.') if len(s.strip()) > 10]
        
        # Filter for sentences that make factual claims (contain verbs like "is", "are", "has", etc.)
        factual_verbs = ["is", "are", "was", "were", "has", "have", "had", "can", "will", "does", "do"]
        factual_claims = []
        
        for sentence in sentences:
            if any(verb in sentence.lower().split() for verb in factual_verbs):
                factual_claims.append(sentence)
        
        return factual_claims
    
    def _identify_unsupported_claims(self, claims: List[str], retrieved_chunks: List[RankedChunk]) -> List[str]:
        """Identify claims that are not supported by the retrieved context"""
        if not retrieved_chunks:
            return claims  # All claims are unsupported if no context
        
        context_text = " ".join([chunk.content.lower() for chunk in retrieved_chunks])
        unsupported_claims = []
        
        for claim in claims:
            claim_words = set(claim.lower().split())
            context_words = set(context_text.split())
            
            # Simple support check: if claim has significant word overlap with context
            if claim_words:
                overlap_ratio = len(claim_words & context_words) / len(claim_words)
                if overlap_ratio < 0.3:  # Less than 30% overlap indicates potential lack of support
                    unsupported_claims.append(claim)
        
        return unsupported_claims
    
    def _detect_contradictions(self, response: str, retrieved_chunks: List[RankedChunk]) -> float:
        """Detect potential contradictions between response and source material"""
        # Simple contradiction detection based on negation patterns
        contradiction_indicators = ["not", "no", "never", "cannot", "isn't", "aren't", "doesn't", "don't"]
        
        response_lower = response.lower()
        context_text = " ".join([chunk.content.lower() for chunk in retrieved_chunks])
        
        # Count negation patterns in response vs context
        response_negations = sum(1 for indicator in contradiction_indicators if indicator in response_lower)
        context_negations = sum(1 for indicator in contradiction_indicators if indicator in context_text)
        
        # If response has significantly more negations than context, potential contradiction
        if context_negations > 0:
            negation_ratio = response_negations / (context_negations + 1)
            return min(1.0, negation_ratio / 3.0)  # Normalize
        
        return 0.0
    
    def _calculate_final_evaluation_score(self, evaluation_metrics: Dict[str, float]) -> float:
        """
        Calculate the final evaluation score from individual metrics.
        
        Args:
            evaluation_metrics: Dictionary with individual evaluation scores
            
        Returns:
            Final evaluation score between 0.0 and 1.0
        """
        try:
            # Weighted combination of evaluation metrics
            weights = {
                'response_quality': 0.25,
                'grounding_score': 0.30,
                'completeness_score': 0.20,
                'citation_accuracy': 0.15,
                'hallucination_penalty': 0.10  # Penalty for high hallucination risk
            }
            
            final_score = (
                weights['response_quality'] * evaluation_metrics.get('response_quality', 0.0) +
                weights['grounding_score'] * evaluation_metrics.get('grounding_score', 0.0) +
                weights['completeness_score'] * evaluation_metrics.get('completeness_score', 0.0) +
                weights['citation_accuracy'] * evaluation_metrics.get('citation_accuracy', 0.0) +
                weights['hallucination_penalty'] * (1.0 - evaluation_metrics.get('hallucination_score', 0.5))
            )
            
            # Update the overall score in metrics
            evaluation_metrics['overall_score'] = final_score
            
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            logger.error(f"Error calculating final evaluation score: {e}")
            return 0.5
    
    def _should_trigger_re_retrieval(self, 
                                   evaluation_score: float, 
                                   evaluation_metrics: Dict[str, float], 
                                   current_iteration: int, 
                                   max_iterations: int) -> bool:
        """
        Determine if re-retrieval should be triggered based on evaluation results.
        
        Args:
            evaluation_score: Overall evaluation score
            evaluation_metrics: Detailed evaluation metrics
            current_iteration: Current iteration count
            max_iterations: Maximum allowed iterations
            
        Returns:
            True if re-retrieval should be triggered, False otherwise
        """
        try:
            # Don't re-retrieve if we've reached max iterations
            if current_iteration >= max_iterations:
                logger.info("Max iterations reached, not triggering re-retrieval")
                return False
            
            # Trigger re-retrieval if overall score is low
            if evaluation_score < 0.6:
                logger.info(f"Low evaluation score ({evaluation_score:.3f}), triggering re-retrieval")
                return True
            
            # Trigger re-retrieval if hallucination risk is high
            hallucination_risk = evaluation_metrics.get('hallucination_score', 0.0)
            if hallucination_risk > 0.7:
                logger.info(f"High hallucination risk ({hallucination_risk:.3f}), triggering re-retrieval")
                return True
            
            # Trigger re-retrieval if grounding is very poor
            grounding_score = evaluation_metrics.get('grounding_score', 1.0)
            if grounding_score < 0.4:
                logger.info(f"Poor grounding ({grounding_score:.3f}), triggering re-retrieval")
                return True
            
            # Trigger re-retrieval if completeness is very poor
            completeness_score = evaluation_metrics.get('completeness_score', 1.0)
            if completeness_score < 0.3:
                logger.info(f"Poor completeness ({completeness_score:.3f}), triggering re-retrieval")
                return True
            
            logger.info("Evaluation scores acceptable, not triggering re-retrieval")
            return False
            
        except Exception as e:
            logger.error(f"Error determining re-retrieval trigger: {e}")
            return False  # Conservative approach: don't re-retrieve on error
    
    async def process_query(self, query: str) -> AgentState:
        """
        Process a user query through the complete agent workflow.
        
        Args:
            query: User query string
            
        Returns:
            Final agent state with response and metadata
        """
        logger.info(f"Processing query through agent workflow: {query}")
        
        # Initialize state
        initial_state = AgentState(
            query=query,
            processing_start_time=datetime.now()
        )
        
        try:
            # Run the workflow
            result = await self.workflow.ainvoke(initial_state)
            
            # LangGraph returns a dict, so we need to convert it back to AgentState
            if isinstance(result, dict):
                # Extract the state from the result
                final_state = AgentState(
                    query=result.get('query', query),
                    plan=result.get('plan'),
                    retrieved_chunks=result.get('retrieved_chunks', []),
                    response=result.get('response'),
                    citations=result.get('citations', []),
                    evaluation_score=result.get('evaluation_score'),
                    iteration_count=result.get('iteration_count', 0),
                    max_iterations=result.get('max_iterations', 3),
                    messages=result.get('messages', []),
                    processing_start_time=result.get('processing_start_time'),
                    processing_end_time=result.get('processing_end_time'),
                    error_message=result.get('error_message')
                )
            else:
                final_state = result
            
            logger.info("Query processing completed successfully")
            return final_state
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            initial_state.error_message = str(e)
            initial_state.processing_end_time = datetime.now()
            return initial_state


# Factory function for creating agent orchestrator instances
def create_agent_orchestrator() -> AgentOrchestrator:
    """
    Factory function to create and configure an AgentOrchestrator instance.
    
    Returns:
        Configured AgentOrchestrator instance
    """
    return AgentOrchestrator()