"""
Tests for the Agent Orchestration System

Tests the basic LangGraph workflow setup and agent state management.
"""

import pytest
from datetime import datetime
from app.services.agent_orchestrator import (
    AgentOrchestrator,
    AgentState,
    DocumentMetadata,
    Chunk,
    Citation,
    RankedChunk,
    create_agent_orchestrator
)


class TestAgentState:
    """Test the AgentState data structure"""
    
    def test_agent_state_initialization(self):
        """Test that AgentState initializes with correct defaults"""
        query = "What is the capital of France?"
        state = AgentState(query=query)
        
        assert state.query == query
        assert state.plan is None
        assert state.retrieved_chunks == []
        assert state.response is None
        assert state.citations == []
        assert state.evaluation_score is None
        assert state.iteration_count == 0
        assert state.max_iterations == 3
        assert state.messages == []
        assert state.processing_start_time is None
        assert state.processing_end_time is None
        assert state.error_message is None
    
    def test_agent_state_with_custom_values(self):
        """Test AgentState with custom initialization values"""
        query = "Test query"
        plan = "Test plan"
        max_iterations = 5
        
        state = AgentState(
            query=query,
            plan=plan,
            max_iterations=max_iterations
        )
        
        assert state.query == query
        assert state.plan == plan
        assert state.max_iterations == max_iterations


class TestDataStructures:
    """Test the data structures used in agent orchestration"""
    
    def test_document_metadata_creation(self):
        """Test DocumentMetadata structure"""
        metadata = DocumentMetadata(
            document_id="doc_123",
            filename="test.pdf",
            file_size=1024,
            page_count=5,
            creation_date=datetime.now(),
            processing_date=datetime.now(),
            checksum="abc123"
        )
        
        assert metadata.document_id == "doc_123"
        assert metadata.filename == "test.pdf"
        assert metadata.file_size == 1024
        assert metadata.page_count == 5
        assert metadata.checksum == "abc123"
    
    def test_chunk_creation(self):
        """Test Chunk structure"""
        metadata = DocumentMetadata(
            document_id="doc_123",
            filename="test.pdf",
            file_size=1024,
            page_count=5,
            creation_date=datetime.now(),
            processing_date=datetime.now(),
            checksum="abc123"
        )
        
        chunk = Chunk(
            chunk_id="chunk_1",
            document_id="doc_123",
            content="This is test content",
            page_number=1,
            start_char=0,
            end_char=20,
            metadata=metadata
        )
        
        assert chunk.chunk_id == "chunk_1"
        assert chunk.document_id == "doc_123"
        assert chunk.content == "This is test content"
        assert chunk.page_number == 1
        assert chunk.start_char == 0
        assert chunk.end_char == 20
        assert chunk.metadata == metadata
    
    def test_citation_creation(self):
        """Test Citation structure"""
        citation = Citation(
            document_id="doc_123",
            filename="test.pdf",
            page_number=1,
            chunk_content="Relevant content",
            relevance_score=0.95
        )
        
        assert citation.document_id == "doc_123"
        assert citation.filename == "test.pdf"
        assert citation.page_number == 1
        assert citation.chunk_content == "Relevant content"
        assert citation.relevance_score == 0.95


class TestAgentOrchestrator:
    """Test the AgentOrchestrator class"""
    
    def test_orchestrator_initialization(self):
        """Test that AgentOrchestrator initializes correctly"""
        orchestrator = AgentOrchestrator()
        
        assert orchestrator.workflow is not None
        # Verify the workflow has the expected structure
        workflow = orchestrator.create_workflow()
        assert workflow is not None
    
    def test_factory_function(self):
        """Test the factory function creates a valid orchestrator"""
        orchestrator = create_agent_orchestrator()
        
        assert isinstance(orchestrator, AgentOrchestrator)
        assert orchestrator.workflow is not None
    
    def test_plan_query_implementation(self):
        """Test the planner agent implementation"""
        orchestrator = AgentOrchestrator()
        query = "What is machine learning?"
        state = AgentState(query=query)
        
        updated_state = orchestrator.plan_query(state)
        
        assert updated_state.query == query
        assert updated_state.plan is not None
        assert "QUERY ANALYSIS:" in updated_state.plan
        assert "RETRIEVAL STRATEGY:" in updated_state.plan
        assert "EXECUTION STEPS:" in updated_state.plan
        assert updated_state.processing_start_time is not None
        assert updated_state.error_message is None
    
    def test_retrieve_documents_placeholder(self):
        """Test the retriever agent placeholder implementation"""
        orchestrator = AgentOrchestrator()
        query = "What is machine learning?"
        state = AgentState(query=query)
        
        updated_state = orchestrator.retrieve_documents(state)
        
        assert updated_state.query == query
        assert isinstance(updated_state.retrieved_chunks, list)
        assert len(updated_state.retrieved_chunks) == 0  # Placeholder returns empty list
    
    def test_generate_answer_placeholder(self):
        """Test the answer agent implementation with no retrieved chunks"""
        orchestrator = AgentOrchestrator()
        query = "What is machine learning?"
        state = AgentState(query=query)
        
        updated_state = orchestrator.generate_answer(state)
        
        assert updated_state.query == query
        assert updated_state.response is not None
        assert "don't have enough information" in updated_state.response
        assert isinstance(updated_state.citations, list)
        assert len(updated_state.citations) == 0
    
    def test_generate_answer_with_chunks(self):
        """Test the answer agent implementation with retrieved chunks"""
        orchestrator = AgentOrchestrator()
        query = "What is machine learning?"
        
        # Create mock chunks with metadata
        metadata = DocumentMetadata(
            document_id="doc_1",
            filename="ml_guide.pdf",
            file_size=1024,
            page_count=10,
            creation_date=datetime.now(),
            processing_date=datetime.now(),
            checksum="abc123"
        )
        
        chunks = [
            RankedChunk(
                chunk_id="chunk_1",
                document_id="doc_1",
                content="Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.",
                page_number=1,
                start_char=0,
                end_char=150,
                metadata=metadata,
                score=0.9,
                retrieval_method='hybrid',
                rank=1,
                rerank_score=0.9,
                vector_score=0.85,
                bm25_score=0.8,
                combined_score=0.87,
                relevance_score=0.9,
                final_score=0.9
            ),
            RankedChunk(
                chunk_id="chunk_2",
                document_id="doc_1",
                content="There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.",
                page_number=2,
                start_char=200,
                end_char=320,
                metadata=metadata,
                score=0.8,
                retrieval_method='hybrid',
                rank=2,
                rerank_score=0.8,
                vector_score=0.75,
                bm25_score=0.7,
                combined_score=0.77,
                relevance_score=0.8,
                final_score=0.8
            )
        ]
        
        state = AgentState(query=query, retrieved_chunks=chunks)
        updated_state = orchestrator.generate_answer(state)
        
        assert updated_state.query == query
        assert updated_state.response is not None
        assert len(updated_state.response) > 50  # Should be a substantial response
        assert isinstance(updated_state.citations, list)
        assert len(updated_state.citations) > 0  # Should have citations
        
        # Check that response contains source references or citations
        response_lower = updated_state.response.lower()
        assert "source" in response_lower or "sources:" in response_lower
        
        # Verify citations have correct structure
        for citation in updated_state.citations:
            assert hasattr(citation, 'document_id')
            assert hasattr(citation, 'filename')
            assert hasattr(citation, 'page_number')
            assert hasattr(citation, 'chunk_content')
            assert hasattr(citation, 'relevance_score')
    
    def test_evaluate_response_placeholder(self):
        """Test the evaluator agent implementation"""
        orchestrator = AgentOrchestrator()
        query = "What is machine learning?"
        state = AgentState(query=query, iteration_count=0)
        
        updated_state = orchestrator.evaluate_response(state)
        
        assert updated_state.query == query
        assert updated_state.evaluation_score is not None
        assert 0.0 <= updated_state.evaluation_score <= 1.0  # Score should be in valid range
        assert updated_state.iteration_count == 1
        assert updated_state.processing_end_time is not None
        
        # With no response and no chunks, score should be low
        assert updated_state.evaluation_score < 0.5
    
    def test_evaluate_response_with_content(self):
        """Test the evaluator agent with actual response and chunks"""
        orchestrator = AgentOrchestrator()
        query = "What is machine learning?"
        
        # Create mock chunks with metadata
        metadata = DocumentMetadata(
            document_id="doc_1",
            filename="ml_guide.pdf",
            file_size=1024,
            page_count=10,
            creation_date=datetime.now(),
            processing_date=datetime.now(),
            checksum="abc123"
        )
        
        chunks = [
            RankedChunk(
                chunk_id="chunk_1",
                document_id="doc_1",
                content="Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.",
                page_number=1,
                start_char=0,
                end_char=150,
                metadata=metadata,
                score=0.9,
                retrieval_method='hybrid',
                rank=1,
                rerank_score=0.9,
                vector_score=0.85,
                bm25_score=0.8,
                combined_score=0.87,
                relevance_score=0.9,
                final_score=0.9
            )
        ]
        
        citations = [
            Citation(
                document_id="doc_1",
                filename="ml_guide.pdf",
                page_number=1,
                chunk_content="Machine learning is a subset of artificial intelligence...",
                relevance_score=0.9
            )
        ]
        
        response = "Machine learning is a subset of artificial intelligence that enables computers to learn from data. According to Source 1, it allows systems to make decisions without explicit programming."
        
        state = AgentState(
            query=query, 
            response=response,
            retrieved_chunks=chunks,
            citations=citations,
            iteration_count=0
        )
        
        updated_state = orchestrator.evaluate_response(state)
        
        assert updated_state.query == query
        assert updated_state.evaluation_score is not None
        assert 0.0 <= updated_state.evaluation_score <= 1.0
        assert updated_state.iteration_count == 1
        assert updated_state.processing_end_time is not None
        
        # With good response and chunks, score should be higher
        assert updated_state.evaluation_score > 0.3
    
    def test_should_continue_logic(self):
        """Test the workflow continuation decision logic"""
        orchestrator = AgentOrchestrator()
        
        # Test case: Low score, should continue
        state_continue = AgentState(
            query="test",
            evaluation_score=0.6,
            iteration_count=1,
            max_iterations=3
        )
        assert orchestrator._should_continue(state_continue) == "continue"
        
        # Test case: High score, should end
        state_end = AgentState(
            query="test",
            evaluation_score=0.8,
            iteration_count=1,
            max_iterations=3
        )
        assert orchestrator._should_continue(state_end) == "end"
        
        # Test case: Max iterations reached, should end
        state_max_iter = AgentState(
            query="test",
            evaluation_score=0.6,
            iteration_count=3,
            max_iterations=3
        )
        assert orchestrator._should_continue(state_max_iter) == "end"
    
    def test_query_analysis_simple(self):
        """Test query analysis for simple queries"""
        orchestrator = AgentOrchestrator()
        
        # Test simple factual query
        analysis = orchestrator._analyze_query_complexity("What is Python?")
        assert analysis["query_type"] == "explanation"
        assert analysis["complexity_score"] == 2
        assert analysis["intent"] == "definition"
        assert "python" in analysis["keywords"]
        assert not analysis["requires_decomposition"]
    
    def test_query_analysis_complex(self):
        """Test query analysis for complex queries"""
        orchestrator = AgentOrchestrator()
        
        # Test comparison query
        analysis = orchestrator._analyze_query_complexity("Compare Python and Java programming languages")
        assert analysis["query_type"] == "comparison"
        assert analysis["complexity_score"] == 3
        assert analysis["intent"] == "comparative"
        assert analysis["requires_decomposition"]
        assert "python" in analysis["keywords"]
        assert "java" in analysis["keywords"]
    
    def test_query_decomposition(self):
        """Test query decomposition for complex queries"""
        orchestrator = AgentOrchestrator()
        
        # Test comparison query decomposition
        analysis = {"query_type": "comparison", "requires_decomposition": True, "keywords": ["python", "java"]}
        decomposition = orchestrator._decompose_query("Compare Python and Java", analysis)
        
        assert len(decomposition) == 3
        assert "What is python?" in decomposition
        assert "What is java?" in decomposition
        assert "differences between python and java" in decomposition[2].lower()
    
    def test_retrieval_strategy_planning(self):
        """Test retrieval strategy planning"""
        orchestrator = AgentOrchestrator()
        
        # Test strategy for definition query
        analysis = {"query_type": "definition", "complexity_score": 1, "keywords": ["python"]}
        strategy = orchestrator._plan_retrieval_strategy("What is Python?", analysis)
        
        assert strategy["primary_method"] == "hybrid"
        assert strategy["keyword_weight"] == 0.5
        assert strategy["vector_weight"] == 0.5
        assert strategy["top_k"] == 5
        assert strategy["use_reranking"]
    
    def test_execution_plan_generation(self):
        """Test execution plan generation"""
        orchestrator = AgentOrchestrator()
        
        query = "What is machine learning?"
        analysis = {"query_type": "explanation", "complexity_score": 2, "intent": "definition", "keywords": ["machine", "learning"]}
        decomposition = [query]
        strategy = {"primary_method": "hybrid", "vector_weight": 0.7, "keyword_weight": 0.3, "top_k": 10, "rerank_top_k": 5}
        
        plan = orchestrator._generate_execution_plan(query, analysis, decomposition, strategy)
        
        assert "QUERY ANALYSIS:" in plan
        assert "RETRIEVAL STRATEGY:" in plan
        assert "EXECUTION STEPS:" in plan
        assert "Type: explanation" in plan
        assert "Complexity: 2/3" in plan
    
    @pytest.mark.asyncio
    async def test_process_query_basic(self):
        """Test basic query processing through the workflow"""
        orchestrator = AgentOrchestrator()
        query = "What is artificial intelligence?"
        
        result_state = await orchestrator.process_query(query)
        
        assert result_state.query == query
        assert result_state.plan is not None
        assert result_state.response is not None
        assert result_state.evaluation_score is not None
        assert result_state.processing_start_time is not None
        assert result_state.error_message is None
    
    def test_retrieve_documents_with_rag_pipeline(self):
        """Test the retriever agent with RAG pipeline integration"""
        orchestrator = AgentOrchestrator()
        query = "What is machine learning?"
        
        # Create state with a plan
        state = AgentState(
            query=query,
            plan="QUERY ANALYSIS:\n- Type: explanation\n- Complexity: 2/3\n\nRETRIEVAL STRATEGY:\n- Method: hybrid\n- Vector weight: 0.7\n- Keyword weight: 0.3\n- Retrieve top 10 chunks\n- Re-rank to top 5 chunks"
        )
        
        # Test retrieval (may fail if no documents are loaded, but should not crash)
        updated_state = orchestrator.retrieve_documents(state)
        
        assert updated_state.query == query
        assert isinstance(updated_state.retrieved_chunks, list)
        # The result may be empty if no documents are loaded, but should not error
        assert updated_state.error_message is None or "Retrieval error:" in updated_state.error_message
    
    def test_query_reformulation(self):
        """Test query reformulation capabilities"""
        orchestrator = AgentOrchestrator()
        
        # Test simple query
        variants = orchestrator._reformulate_query("What is Python?", None)
        assert len(variants) >= 1
        assert "What is Python?" in variants
        
        # Test complex query with plan
        plan = "Sub-question 1: What is machine learning?\nSub-question 2: What are neural networks?"
        variants = orchestrator._reformulate_query("Compare machine learning and neural networks", plan)
        assert len(variants) >= 2
        assert "Compare machine learning and neural networks" in variants
        assert "What is machine learning?" in variants
    
    def test_retrieval_parameter_extraction(self):
        """Test extraction of retrieval parameters from plan"""
        orchestrator = AgentOrchestrator()
        
        plan = """RETRIEVAL STRATEGY:
        - Method: hybrid
        - Vector weight: 0.8
        - Keyword weight: 0.2
        - Retrieve top 20 chunks
        - Re-rank to top 10 chunks"""
        
        params = orchestrator._extract_retrieval_params_from_plan(plan)
        
        assert params['retrieval_k'] == 20
        assert params['final_k'] == 10
        assert params['vector_weight'] == 0.8
        assert params['keyword_weight'] == 0.2
    
    def test_retrieval_metrics_calculation(self):
        """Test calculation of retrieval quality metrics"""
        orchestrator = AgentOrchestrator()
        
        # Create mock chunks
        metadata = DocumentMetadata(
            document_id="doc_1",
            filename="test.pdf",
            file_size=1024,
            page_count=1,
            creation_date=datetime.now(),
            processing_date=datetime.now(),
            checksum="abc123"
        )
        
        chunks = [
            RankedChunk(
                chunk_id="chunk_1",
                document_id="doc_1",
                content="Machine learning is a subset of artificial intelligence",
                page_number=1,
                start_char=0,
                end_char=50,
                metadata=metadata,
                score=0.9,
                retrieval_method='hybrid',
                rank=1,
                rerank_score=0.9,
                vector_score=0.8,
                bm25_score=0.7,
                combined_score=0.85,
                relevance_score=0.9,
                final_score=0.9
            ),
            RankedChunk(
                chunk_id="chunk_2",
                document_id="doc_2",
                content="Artificial intelligence encompasses machine learning",
                page_number=1,
                start_char=0,
                end_char=50,
                metadata=metadata,
                score=0.8,
                retrieval_method='hybrid',
                rank=2,
                rerank_score=0.8,
                vector_score=0.7,
                bm25_score=0.6,
                combined_score=0.75,
                relevance_score=0.8,
                final_score=0.8
            )
        ]
        
        metrics = orchestrator._calculate_retrieval_metrics(chunks, "What is machine learning?")
        
        assert metrics['total_chunks'] == 2
        assert abs(metrics['avg_relevance_score'] - 0.85) < 0.001
        assert metrics['document_diversity'] == 1.0  # 2 unique docs / 2 total chunks
        assert metrics['query_coverage'] > 0.0  # Should have some keyword overlap