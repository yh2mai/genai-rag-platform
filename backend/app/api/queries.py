"""
Query processing API endpoints for RAG pipeline and agent orchestration
"""
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
import logging
import json
import asyncio

from app.models.database import get_db, QueryLog
from app.services.agent_orchestrator import AgentOrchestrator, AgentState
from app.services.rag_pipeline import RAGEngine
from app.services.llm_service import get_llm_service
from app.services.cost_performance_tracker import CostPerformanceTracker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/queries", tags=["queries"])

# Initialize services
agent_orchestrator = AgentOrchestrator()
rag_pipeline = RAGEngine()
cost_tracker = CostPerformanceTracker()
llm_service = get_llm_service()


class QueryRequest(BaseModel):
    """Query request model"""
    query: str = Field(..., min_length=1, max_length=2000, description="User query")
    max_chunks: Optional[int] = Field(default=5, ge=1, le=20, description="Maximum number of chunks to retrieve")
    use_agents: Optional[bool] = Field(default=True, description="Whether to use agent orchestration")
    include_citations: Optional[bool] = Field(default=True, description="Whether to include citations")


class CitationResponse(BaseModel):
    """Citation response model"""
    document_id: str
    filename: str
    page_number: int
    chunk_content: str
    relevance_score: float


class QueryResponse(BaseModel):
    """Query response model"""
    query_id: str
    query: str
    response: str
    citations: List[CitationResponse]
    processing_time: float
    token_usage: Optional[Dict[str, int]] = None
    evaluation_score: Optional[float] = None
    status: str


class QueryStatus(BaseModel):
    """Query processing status model"""
    query_id: str
    status: str  # "processing", "completed", "failed"
    progress: Optional[float] = None
    message: Optional[str] = None
    estimated_completion: Optional[str] = None


class QueryHistoryResponse(BaseModel):
    """Query history response model"""
    queries: List[QueryResponse]
    total: int
    skip: int
    limit: int


# Store for tracking ongoing queries
ongoing_queries: Dict[str, QueryStatus] = {}


async def process_query_background(
    query_id: str,
    query_request: QueryRequest,
    db_session: Session
):
    """Background task to process query through agent workflow"""
    try:
        logger.info(f"Starting background query processing for {query_id}")
        
        # Update status
        ongoing_queries[query_id] = QueryStatus(
            query_id=query_id,
            status="processing",
            progress=0.1,
            message="Initializing query processing"
        )
        
        start_time = datetime.now()
        print("process_query_background")
        if query_request.use_agents:
            # Use agent orchestration
            ongoing_queries[query_id].message = "Processing through agent workflow"
            ongoing_queries[query_id].progress = 0.3
            print("process_query_background -- using agent")
            result = await agent_orchestrator.process_query(query_request.query)

            # Extract response data
            response_text = result.response or "I couldn't generate a response for your query."
            citations = []
            
            if query_request.include_citations and result.citations:
                for citation in result.citations:
                    citations.append(CitationResponse(
                        document_id=citation.document_id,
                        filename=citation.filename,
                        page_number=citation.page_number,
                        chunk_content=citation.chunk_content[:200] + "..." if len(citation.chunk_content) > 200 else citation.chunk_content,
                        relevance_score=citation.relevance_score
                    ))
            
            evaluation_score = result.evaluation_score
            
        else:
            # Use direct RAG pipeline
            ongoing_queries[query_id].message = "Processing through RAG pipeline"
            ongoing_queries[query_id].progress = 0.3
            
            # Retrieve relevant chunks
            retrieved_chunks = rag_pipeline.hybrid_retrieve(
                query=query_request.query,
                top_k=query_request.max_chunks * 2  # Get more for re-ranking
            )
            
            ongoing_queries[query_id].progress = 0.5
            
            # Re-rank results
            if retrieved_chunks:
                ranked_chunks = rag_pipeline.rerank_results(
                    query=query_request.query,
                    chunks=retrieved_chunks
                )[:query_request.max_chunks]
            else:
                ranked_chunks = []
            
            ongoing_queries[query_id].progress = 0.7
            
            # Generate response using LLM
            if ranked_chunks:
                context_window = rag_pipeline.optimize_context(ranked_chunks, max_tokens=4000)
                context_text = context_window.combined_context if hasattr(context_window, 'combined_context') else str(context_window)
                
                # Use LLM to generate a natural response
                llm_response = llm_service.generate_response(
                    query=query_request.query,
                    context=context_text
                )
                response_text = llm_response.content
                
                # Track token usage
                if llm_response.tokens_used:
                    cost_tracker.track_token_usage(
                        input_tokens=llm_response.tokens_used.get('prompt_tokens', 0),
                        output_tokens=llm_response.tokens_used.get('completion_tokens', 0),
                        model=llm_response.model,
                        operation_type="query_response"
                    )
            else:
                # No relevant documents found
                llm_response = llm_service.generate_response(
                    query=query_request.query,
                    context="No relevant documents found in the knowledge base.",
                    system_prompt="You are a helpful AI assistant. The user asked a question but no relevant documents were found in the knowledge base. Politely explain that you don't have information about their specific question in the available documents, and suggest they might want to upload relevant documents or rephrase their question."
                )
                response_text = llm_response.content
            
            # Create citations
            citations = []
            if query_request.include_citations and ranked_chunks:
                for chunk in ranked_chunks:
                    citations.append(CitationResponse(
                        document_id=chunk.document_id,
                        filename=getattr(chunk.metadata, 'filename', 'Unknown'),
                        page_number=chunk.page_number,
                        chunk_content=chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                        relevance_score=chunk.final_score
                    ))
            
            evaluation_score = None
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Track costs and performance
        token_usage = cost_tracker.get_current_usage()
        
        # Store query log in database
        query_log = QueryLog(
            query_id=query_id,
            user_query=query_request.query,
            response=response_text,
            citations=[citation.dict() for citation in citations],
            processing_time=processing_time,
            token_usage=token_usage,
            evaluation_metrics={
                "evaluation_score": evaluation_score,
                "citation_count": len(citations),
                "use_agents": query_request.use_agents
            },
            timestamp=start_time
        )
        
        db_session.add(query_log)
        db_session.commit()
        
        # Update final status
        ongoing_queries[query_id] = QueryStatus(
            query_id=query_id,
            status="completed",
            progress=1.0,
            message="Query processing completed successfully"
        )
        
        logger.info(f"Successfully completed query processing for {query_id}")
        
    except Exception as e:
        logger.error(f"Error processing query {query_id}: {str(e)}")
        
        # Update error status
        ongoing_queries[query_id] = QueryStatus(
            query_id=query_id,
            status="failed",
            progress=0.0,
            message=f"Query processing failed: {str(e)}"
        )
        
        # Store error in database
        try:
            error_log = QueryLog(
                query_id=query_id,
                user_query=query_request.query,
                response=f"Error: {str(e)}",
                citations=[],
                processing_time=0.0,
                token_usage={},
                evaluation_metrics={"error": str(e)},
                timestamp=datetime.now()
            )
            db_session.add(error_log)
            db_session.commit()
        except Exception as db_error:
            logger.error(f"Failed to store error log: {str(db_error)}")
    
    finally:
        db_session.close()


@router.post("/", response_model=Dict[str, str])
async def submit_query(
    query_request: QueryRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Submit a query for processing
    
    - **query**: The question to ask (1-2000 characters)
    - **max_chunks**: Maximum number of document chunks to retrieve (1-20)
    - **use_agents**: Whether to use agent orchestration (default: true)
    - **include_citations**: Whether to include source citations (default: true)
    
    Returns query ID for status tracking
    """
    try:
        print("post query")
        # Generate unique query ID
        query_id = str(uuid.uuid4())
        
        # Initialize status tracking
        ongoing_queries[query_id] = QueryStatus(
            query_id=query_id,
            status="queued",
            progress=0.0,
            message="Query queued for processing"
        )
        
        # Start background processing
        from app.models.database import SessionLocal
        background_db = SessionLocal()
        
        background_tasks.add_task(
            process_query_background,
            query_id,
            query_request,
            background_db
        )
        
        return {
            "query_id": query_id,
            "status": "queued",
            "message": "Query submitted successfully. Use the query ID to check status."
        }
        
    except Exception as e:
        logger.error(f"Error submitting query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/{query_id}/status", response_model=QueryStatus)
async def get_query_status(query_id: str):
    """
    Get query processing status
    
    - **query_id**: Unique query identifier
    """
    try:
        if query_id not in ongoing_queries:
            raise HTTPException(
                status_code=404,
                detail=f"Query {query_id} not found"
            )
        
        return ongoing_queries[query_id]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting query status {query_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/{query_id}", response_model=QueryResponse)
async def get_query_result(
    query_id: str,
    db: Session = Depends(get_db)
):
    """
    Get query result by ID
    
    - **query_id**: Unique query identifier
    """
    try:
        # Get query log from database
        query_log = db.query(QueryLog).filter(
            QueryLog.query_id == query_id
        ).first()
        
        if not query_log:
            raise HTTPException(
                status_code=404,
                detail=f"Query {query_id} not found"
            )
        
        # Convert citations from JSON
        citations = []
        if query_log.citations:
            for citation_data in query_log.citations:
                citations.append(CitationResponse(**citation_data))
        
        # Determine status
        status = "completed"
        if query_id in ongoing_queries:
            status = ongoing_queries[query_id].status
        
        return QueryResponse(
            query_id=query_id,
            query=query_log.user_query,
            response=query_log.response or "",
            citations=citations,
            processing_time=query_log.processing_time or 0.0,
            token_usage=query_log.token_usage,
            evaluation_score=query_log.evaluation_metrics.get('evaluation_score') if query_log.evaluation_metrics else None,
            status=status
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting query result {query_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/direct", response_model=QueryResponse)
async def process_query_direct(
    query_request: QueryRequest,
    db: Session = Depends(get_db)
):
    """
    Process query directly (synchronous) - for simple queries
    
    - **query**: The question to ask (1-2000 characters)
    - **max_chunks**: Maximum number of document chunks to retrieve (1-20)
    - **use_agents**: Whether to use agent orchestration (default: false for direct)
    - **include_citations**: Whether to include source citations (default: true)
    
    Returns immediate response (may timeout for complex queries)
    """
    try:
        print("queries post direct")
        query_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        # Force direct RAG pipeline for synchronous processing
        query_request.use_agents = False
        
        # Retrieve relevant chunks
        retrieved_chunks = rag_pipeline.hybrid_retrieve(
            query=query_request.query,
            top_k=query_request.max_chunks * 2
        )
        
        # Re-rank results
        if retrieved_chunks:
            ranked_chunks = rag_pipeline.rerank_results(
                query=query_request.query,
                chunks=retrieved_chunks
            )[:query_request.max_chunks]
        else:
            ranked_chunks = []
        
        # Generate LLM response
        if ranked_chunks:
            context_window = rag_pipeline.optimize_context(ranked_chunks, max_tokens=2000)
            # Extract the actual text content from the ContextWindow
            context_text = context_window.combined_context if hasattr(context_window, 'combined_context') else str(context_window)
            
            # Use LLM to generate a natural response
            llm_response = llm_service.generate_response(
                query=query_request.query,
                context=context_text
            )
            response_text = llm_response.content
            
            # Track token usage
            if llm_response.tokens_used:
                cost_tracker.track_token_usage(
                    input_tokens=llm_response.tokens_used.get('prompt_tokens', 0),
                    output_tokens=llm_response.tokens_used.get('completion_tokens', 0),
                    model=llm_response.model,
                    operation_type="query_response"
                )
        else:
            # No relevant documents found
            llm_response = llm_service.generate_response(
                query=query_request.query,
                context="No relevant documents found in the knowledge base.",
                system_prompt="You are a helpful AI assistant. The user asked a question but no relevant documents were found in the knowledge base. Politely explain that you don't have information about their specific question in the available documents, and suggest they might want to upload relevant documents or rephrase their question."
            )
            response_text = llm_response.content
        
        # Create citations
        citations = []
        if query_request.include_citations and ranked_chunks:
            for chunk in ranked_chunks:
                citations.append(CitationResponse(
                    document_id=chunk.document_id,
                    filename=getattr(chunk.metadata, 'filename', 'Unknown'),
                    page_number=chunk.page_number,
                    chunk_content=chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                    relevance_score=chunk.final_score
                ))
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Store query log
        query_log = QueryLog(
            query_id=query_id,
            user_query=query_request.query,
            response=response_text,
            citations=[citation.dict() for citation in citations],
            processing_time=processing_time,
            token_usage=cost_tracker.get_current_usage(),
            evaluation_metrics={
                "citation_count": len(citations),
                "use_agents": False,
                "direct_processing": True
            },
            timestamp=start_time
        )
        
        db.add(query_log)
        db.commit()
        
        return QueryResponse(
            query_id=query_id,
            query=query_request.query,
            response=response_text,
            citations=citations,
            processing_time=processing_time,
            token_usage=cost_tracker.get_current_usage(),
            evaluation_score=None,
            status="completed"
        )
        
    except Exception as e:
        logger.error(f"Error processing direct query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/", response_model=QueryHistoryResponse)
async def get_query_history(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    Get query history with pagination
    
    - **skip**: Number of queries to skip (default: 0)
    - **limit**: Maximum number of queries to return (default: 50, max: 200)
    """
    try:
        # Validate parameters
        if limit > 200:
            limit = 200
        if skip < 0:
            skip = 0
        
        # Get query logs
        query_logs = db.query(QueryLog).order_by(
            QueryLog.timestamp.desc()
        ).offset(skip).limit(limit).all()
        
        total = db.query(QueryLog).count()
        
        # Convert to response format
        queries = []
        for log in query_logs:
            citations = []
            if log.citations:
                for citation_data in log.citations:
                    citations.append(CitationResponse(**citation_data))
            
            queries.append(QueryResponse(
                query_id=log.query_id,
                query=log.user_query,
                response=log.response or "",
                citations=citations,
                processing_time=log.processing_time or 0.0,
                token_usage=log.token_usage,
                evaluation_score=log.evaluation_metrics.get('evaluation_score') if log.evaluation_metrics else None,
                status="completed"
            ))
        
        return QueryHistoryResponse(
            queries=queries,
            total=total,
            skip=skip,
            limit=limit
        )
        
    except Exception as e:
        logger.error(f"Error getting query history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.delete("/{query_id}")
async def delete_query(
    query_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete a query from history
    
    - **query_id**: Unique query identifier
    """
    try:
        query_log = db.query(QueryLog).filter(
            QueryLog.query_id == query_id
        ).first()
        
        if not query_log:
            raise HTTPException(
                status_code=404,
                detail=f"Query {query_id} not found"
            )
        
        db.delete(query_log)
        db.commit()
        
        # Remove from ongoing queries if present
        if query_id in ongoing_queries:
            del ongoing_queries[query_id]
        
        return {"message": f"Query {query_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting query {query_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )