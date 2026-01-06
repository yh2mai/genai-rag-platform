"""
Document upload and management API endpoints
"""
import os
import tempfile
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
import logging

from app.models.database import get_db, Document, DocumentChunk
from app.services.document_processor import PDFProcessor
from app.services.document_storage import DocumentStorageService
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

# Initialize services
pdf_processor = PDFProcessor()
storage_service = DocumentStorageService()
embedding_service = EmbeddingService()
vector_store = VectorStore()


class DocumentResponse(BaseModel):
    """Document metadata response model"""
    document_id: str
    filename: str
    file_size: int
    page_count: int
    processing_status: str
    created_at: str
    chunk_count: Optional[int] = None


class DocumentUploadResponse(BaseModel):
    """Document upload response model"""
    success: bool
    message: str
    document_id: Optional[str] = None
    processing_status: str


class DocumentListResponse(BaseModel):
    """Document list response model"""
    documents: List[DocumentResponse]
    total: int
    skip: int
    limit: int


class ProcessingStatusResponse(BaseModel):
    """Processing status response model"""
    document_id: str
    status: str
    message: Optional[str] = None
    progress: Optional[float] = None


def process_document_background(file_path: str, document_id: str, db_session: Session):
    """Background task to process document and generate embeddings"""
    try:
        logger.info(f"Starting background processing for document {document_id}")
        
        # Process the PDF
        processing_result = pdf_processor.process_pdf(file_path)
        
        if not processing_result.success:
            logger.error(f"Failed to process document {document_id}: {processing_result.error_message}")
            # Update status in database
            doc = db_session.query(Document).filter(Document.document_id == document_id).first()
            if doc:
                doc.processing_status = "failed"
                db_session.commit()
            return
        
        # Store document and chunks
        success = storage_service.store_document(file_path, processing_result, db_session)
        
        if not success:
            logger.error(f"Failed to store document {document_id}")
            # Update status in database
            doc = db_session.query(Document).filter(Document.document_id == document_id).first()
            if doc:
                doc.processing_status = "failed"
                db_session.commit()
            return
        
        # Generate embeddings for chunks
        if processing_result.chunks:
            try:
                # Generate embeddings
                embedded_chunks = embedding_service.generate_embeddings(processing_result.chunks)
                
                # Store in vector database
                vector_store.add_embeddings(embedded_chunks)
                logger.info(f"Generated and stored embeddings for {len(embedded_chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Failed to generate embeddings for document {document_id}: {str(e)}")
                # Document is still stored, just without embeddings
        
        # Update final status
        doc = db_session.query(Document).filter(Document.document_id == document_id).first()
        if doc:
            doc.processing_status = "completed"
            db_session.commit()
        
        logger.info(f"Successfully completed processing for document {document_id}")
        
    except Exception as e:
        logger.error(f"Error in background processing for document {document_id}: {str(e)}")
        # Update status to failed
        try:
            doc = db_session.query(Document).filter(Document.document_id == document_id).first()
            if doc:
                doc.processing_status = "failed"
                db_session.commit()
        except Exception as db_error:
            logger.error(f"Failed to update document status: {str(db_error)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(file_path):
            os.unlink(file_path)
        db_session.close()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload a PDF document for processing
    
    - **file**: PDF file to upload (max 50MB)
    - Returns document ID and processing status
    """
    try:
        # Validate file type
        print("start post upload")
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )
        
        # Validate file size (50MB limit)
        max_size = 50 * 1024 * 1024  # 50MB
        file_content = await file.read()
        if len(file_content) > max_size:
            raise HTTPException(
                status_code=413,
                detail="File size exceeds 50MB limit"
            )

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        # Quick metadata extraction to get document ID
        metadata = pdf_processor.extract_metadata(temp_file_path)
        if not metadata:
            os.unlink(temp_file_path)
            raise HTTPException(
                status_code=400,
                detail="Failed to extract document metadata. File may be corrupted."
            )
        
        # Check if document already exists
        existing_doc = db.query(Document).filter(
            Document.checksum == metadata.checksum
        ).first()
        
        if existing_doc:
            os.unlink(temp_file_path)
            return DocumentUploadResponse(
                success=True,
                message="Document already exists in the system",
                document_id=existing_doc.document_id,
                processing_status=existing_doc.processing_status
            )
        
        # Create initial database record with processing status
        doc_record = Document(
            document_id=metadata.document_id,
            filename=metadata.filename,
            file_path="",  # Will be updated after processing
            file_size=metadata.file_size,
            page_count=metadata.page_count,
            creation_date=metadata.creation_date,
            processing_date=metadata.processing_date,
            checksum=metadata.checksum,
            document_metadata={
                "original_filename": metadata.filename,
                "upload_timestamp": metadata.processing_date.isoformat()
            },
            processing_status="processing"
        )
        print("post upload - document created")
        db.add(doc_record)
        db.commit()
        
        # Start background processing
        # Create new session for background task
        from app.models.database import SessionLocal
        background_db = SessionLocal()
        
        background_tasks.add_task(
            process_document_background,
            temp_file_path,
            metadata.document_id,
            background_db
        )
        
        return DocumentUploadResponse(
            success=True,
            message="Document uploaded successfully. Processing started.",
            document_id=metadata.document_id,
            processing_status="processing"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    List all documents with pagination
    
    - **skip**: Number of documents to skip (default: 0)
    - **limit**: Maximum number of documents to return (default: 100, max: 1000)
    """
    try:
        # Validate parameters
        if limit > 1000:
            limit = 1000
        if skip < 0:
            skip = 0
        
        # Get documents
        documents = storage_service.list_documents(db, skip=skip, limit=limit)
        total = db.query(Document).count()
        
        # Convert to response format
        document_responses = []
        for doc in documents:
            # Get chunk count
            chunk_count = db.query(DocumentChunk).filter(
                DocumentChunk.document_id == doc.document_id
            ).count()
            
            document_responses.append(DocumentResponse(
                document_id=doc.document_id,
                filename=doc.filename,
                file_size=doc.file_size,
                page_count=doc.page_count,
                processing_status=doc.processing_status,
                created_at=doc.created_at.isoformat(),
                chunk_count=chunk_count
            ))
        
        return DocumentListResponse(
            documents=document_responses,
            total=total,
            skip=skip,
            limit=limit
        )
        
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Get document details by ID
    
    - **document_id**: Unique document identifier
    """
    try:
        document = storage_service.get_document(document_id, db)
        
        if not document:
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_id} not found"
            )
        
        # Get chunk count
        chunk_count = db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document_id
        ).count()
        
        return DocumentResponse(
            document_id=document.document_id,
            filename=document.filename,
            file_size=document.file_size,
            page_count=document.page_count,
            processing_status=document.processing_status,
            created_at=document.created_at.isoformat(),
            chunk_count=chunk_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/{document_id}/status", response_model=ProcessingStatusResponse)
async def get_processing_status(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Get document processing status
    
    - **document_id**: Unique document identifier
    """
    try:
        document = storage_service.get_document(document_id, db)
        
        if not document:
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_id} not found"
            )
        
        # Determine message based on status
        message = None
        progress = None
        
        if document.processing_status == "processing":
            message = "Document is being processed"
            progress = 0.5  # Indeterminate progress
        elif document.processing_status == "completed":
            message = "Document processing completed successfully"
            progress = 1.0
        elif document.processing_status == "failed":
            message = "Document processing failed"
            progress = 0.0
        
        return ProcessingStatusResponse(
            document_id=document_id,
            status=document.processing_status,
            message=message,
            progress=progress
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting processing status for {document_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete a document and all associated data
    
    - **document_id**: Unique document identifier
    """
    try:
        success = storage_service.delete_document(document_id, db)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_id} not found or could not be deleted"
            )
        
        return {"message": f"Document {document_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/{document_id}/chunks")
async def get_document_chunks(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Get all chunks for a document
    
    - **document_id**: Unique document identifier
    """
    try:
        # Check if document exists
        document = storage_service.get_document(document_id, db)
        if not document:
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_id} not found"
            )
        
        # Get chunks
        chunks = storage_service.get_document_chunks(document_id, db)
        
        # Convert to response format
        chunk_data = []
        for chunk in chunks:
            chunk_data.append({
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "char_count": chunk.char_count,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char
            })
        
        return {
            "document_id": document_id,
            "chunks": chunk_data,
            "total_chunks": len(chunk_data)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chunks for document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )