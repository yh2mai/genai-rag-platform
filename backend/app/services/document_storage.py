"""
Document storage service for managing PDF files and metadata
"""
import os
import shutil
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
from pathlib import Path

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.core.config import settings
from app.models.database import Document, DocumentChunk, get_db
from app.services.document_processor import DocumentMetadata, ProcessingResult
from app.services.semantic_chunker import Chunk

logger = logging.getLogger(__name__)


class DocumentStorageService:
    """Service for storing and managing documents and their metadata"""
    
    def __init__(self):
        self.storage_path = Path(settings.DOCUMENT_STORE_PATH)
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def store_document(self, 
                      file_path: str, 
                      processing_result: ProcessingResult,
                      db: Session) -> bool:
        """
        Store document file and metadata in the storage system
        
        Args:
            file_path: Original file path
            processing_result: Result from document processing
            db: Database session
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not processing_result.success or not processing_result.document_metadata:
                logger.error("Cannot store document: processing failed or no metadata")
                return False
            
            metadata = processing_result.document_metadata
            
            # Create storage path for this document
            doc_storage_path = self.storage_path / metadata.document_id
            doc_storage_path.mkdir(exist_ok=True)
            
            # Copy original file to storage
            stored_file_path = doc_storage_path / metadata.filename
            shutil.copy2(file_path, stored_file_path)
            
            # Store document metadata in database
            success = self._store_document_metadata(
                metadata=metadata,
                stored_file_path=str(stored_file_path),
                db=db
            )
            
            if not success:
                # Clean up file if database storage failed
                if stored_file_path.exists():
                    stored_file_path.unlink()
                return False
            
            # Store document chunks
            success = self._store_document_chunks(
                chunks=processing_result.chunks,
                db=db
            )
            
            if not success:
                # Clean up if chunk storage failed
                self._cleanup_document(metadata.document_id, db)
                return False
            
            logger.info(f"Successfully stored document {metadata.document_id} with {len(processing_result.chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error storing document: {str(e)}")
            return False
    
    def _store_document_metadata(self, 
                                metadata: DocumentMetadata,
                                stored_file_path: str,
                                db: Session) -> bool:
        """Store document metadata in database"""
        try:
            # Check if document already exists (might be in processing state)
            existing_doc = db.query(Document).filter(
                Document.document_id == metadata.document_id
            ).first()
            
            if existing_doc:
                # Update existing document with final information
                existing_doc.file_path = stored_file_path
                existing_doc.processing_status = "completed"
                existing_doc.document_metadata = {
                    "original_filename": metadata.filename,
                    "processing_date": metadata.processing_date.isoformat()
                }
                db.commit()
                logger.info(f"Updated existing document {metadata.document_id} with storage path")
                return True
            else:
                # Create new document record (shouldn't happen in normal flow)
                doc_record = Document(
                    document_id=metadata.document_id,
                    filename=metadata.filename,
                    file_path=stored_file_path,
                    file_size=metadata.file_size,
                    page_count=metadata.page_count,
                    creation_date=metadata.creation_date,
                    processing_date=metadata.processing_date,
                    checksum=metadata.checksum,
                    document_metadata={
                        "original_filename": metadata.filename,
                        "processing_date": metadata.processing_date.isoformat()
                    },
                    processing_status="completed"
                )
                
                db.add(doc_record)
                db.commit()
                logger.info(f"Created new document record {metadata.document_id}")
                return True
            
        except IntegrityError as e:
            logger.error(f"Database integrity error storing document metadata: {str(e)}")
            db.rollback()
            return False
        except Exception as e:
            logger.error(f"Error storing document metadata: {str(e)}")
            db.rollback()
            return False
    
    def _store_document_chunks(self, chunks: List[Chunk], db: Session) -> bool:
        """Store document chunks in database"""
        try:
            chunk_records = []
            
            for chunk in chunks:
                chunk_record = DocumentChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    char_count=len(chunk.content)
                )
                chunk_records.append(chunk_record)
            
            # Batch insert chunks
            db.add_all(chunk_records)
            db.commit()
            
            logger.info(f"Stored {len(chunk_records)} chunks in database")
            return True
            
        except Exception as e:
            logger.error(f"Error storing document chunks: {str(e)}")
            db.rollback()
            return False
    
    def get_document(self, document_id: str, db: Session) -> Optional[Document]:
        """Retrieve document metadata by ID"""
        try:
            return db.query(Document).filter(
                Document.document_id == document_id
            ).first()
        except Exception as e:
            logger.error(f"Error retrieving document {document_id}: {str(e)}")
            return None
    
    def get_document_chunks(self, document_id: str, db: Session) -> List[DocumentChunk]:
        """Retrieve all chunks for a document"""
        try:
            return db.query(DocumentChunk).filter(
                DocumentChunk.document_id == document_id
            ).order_by(DocumentChunk.chunk_index).all()
        except Exception as e:
            logger.error(f"Error retrieving chunks for document {document_id}: {str(e)}")
            return []
    
    def list_documents(self, 
                      db: Session,
                      skip: int = 0,
                      limit: int = 100) -> List[Document]:
        """List all documents with pagination"""
        try:
            return db.query(Document).offset(skip).limit(limit).all()
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []
    
    def delete_document(self, document_id: str, db: Session) -> bool:
        """Delete document and all associated data"""
        try:
            # Get document record
            document = self.get_document(document_id, db)
            if not document:
                logger.warning(f"Document {document_id} not found for deletion")
                return False
            
            # Delete physical file
            file_path = Path(document.file_path)
            if file_path.exists():
                file_path.unlink()
            
            # Delete document directory if empty
            doc_dir = file_path.parent
            if doc_dir.exists() and not any(doc_dir.iterdir()):
                doc_dir.rmdir()
            
            # Delete from database (chunks will be deleted via cascade)
            db.delete(document)
            db.commit()
            
            logger.info(f"Successfully deleted document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            db.rollback()
            return False
    
    def _cleanup_document(self, document_id: str, db: Session):
        """Clean up document data after failed storage"""
        try:
            # Remove from database
            document = self.get_document(document_id, db)
            if document:
                db.delete(document)
                db.commit()
            
            # Remove physical files
            doc_storage_path = self.storage_path / document_id
            if doc_storage_path.exists():
                shutil.rmtree(doc_storage_path)
                
        except Exception as e:
            logger.error(f"Error during cleanup of document {document_id}: {str(e)}")
    
    def get_storage_stats(self, db: Session) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            total_documents = db.query(Document).count()
            total_chunks = db.query(DocumentChunk).count()
            
            # Calculate total storage size
            total_size = 0
            for doc in db.query(Document).all():
                if os.path.exists(doc.file_path):
                    total_size += os.path.getsize(doc.file_path)
            
            return {
                "total_documents": total_documents,
                "total_chunks": total_chunks,
                "total_storage_bytes": total_size,
                "storage_path": str(self.storage_path)
            }
            
        except Exception as e:
            logger.error(f"Error getting storage stats: {str(e)}")
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "total_storage_bytes": 0,
                "storage_path": str(self.storage_path)
            }