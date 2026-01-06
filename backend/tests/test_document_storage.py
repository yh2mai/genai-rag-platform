"""
Tests for document storage functionality
"""
import os
import tempfile
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.database import Base, Document, DocumentChunk
from app.services.document_storage import DocumentStorageService
from app.services.document_processor import DocumentMetadata, ProcessingResult, PageContent
from app.services.semantic_chunker import Chunk


class TestDocumentStorageService:
    """Test cases for document storage functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Create in-memory SQLite database for testing
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=self.engine)
        
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.db = SessionLocal()
        
        # Create temporary storage directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock settings
        with patch('app.services.document_storage.settings') as mock_settings:
            mock_settings.DOCUMENT_STORE_PATH = self.temp_dir
            self.storage_service = DocumentStorageService()
    
    def teardown_method(self):
        """Clean up test fixtures"""
        self.db.close()
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_processing_result(self) -> ProcessingResult:
        """Create a test processing result"""
        metadata = DocumentMetadata(
            document_id="doc_test123",
            filename="test.pdf",
            file_size=1024,
            page_count=2,
            creation_date=datetime.now(),
            processing_date=datetime.now(),
            checksum="abc123def456"
        )
        
        pages = [
            PageContent(1, "Page 1 content", 15),
            PageContent(2, "Page 2 content", 15)
        ]
        
        chunks = [
            Chunk(
                chunk_id="doc_test123_chunk_0000",
                document_id="doc_test123",
                content="Page 1 content",
                page_number=1,
                start_char=0,
                end_char=15,
                chunk_index=0
            ),
            Chunk(
                chunk_id="doc_test123_chunk_0001",
                document_id="doc_test123",
                content="Page 2 content",
                page_number=2,
                start_char=0,
                end_char=15,
                chunk_index=1
            )
        ]
        
        return ProcessingResult(
            success=True,
            document_metadata=metadata,
            pages=pages,
            chunks=chunks
        )
    
    def test_store_document_success(self):
        """Test successful document storage"""
        # Create test file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(b"test pdf content")
            tmp_file_path = tmp_file.name
        
        try:
            processing_result = self.create_test_processing_result()
            
            success = self.storage_service.store_document(
                file_path=tmp_file_path,
                processing_result=processing_result,
                db=self.db
            )
            
            assert success
            
            # Verify document in database
            doc = self.db.query(Document).filter(
                Document.document_id == "doc_test123"
            ).first()
            
            assert doc is not None
            assert doc.filename == "test.pdf"
            assert doc.checksum == "abc123def456"
            
            # Verify chunks in database
            chunks = self.db.query(DocumentChunk).filter(
                DocumentChunk.document_id == "doc_test123"
            ).all()
            
            assert len(chunks) == 2
            assert chunks[0].content == "Page 1 content"
            assert chunks[1].content == "Page 2 content"
            
            # Verify file was copied to storage
            stored_file_path = Path(self.temp_dir) / "doc_test123" / "test.pdf"
            assert stored_file_path.exists()
            
        finally:
            os.unlink(tmp_file_path)
    
    def test_store_document_failed_processing(self):
        """Test storing document with failed processing result"""
        failed_result = ProcessingResult(
            success=False,
            error_message="Processing failed"
        )
        
        success = self.storage_service.store_document(
            file_path="dummy_path",
            processing_result=failed_result,
            db=self.db
        )
        
        assert not success
    
    def test_store_duplicate_document(self):
        """Test storing document with duplicate checksum"""
        # Create test file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(b"test pdf content")
            tmp_file_path = tmp_file.name
        
        try:
            processing_result = self.create_test_processing_result()
            
            # Store document first time
            success1 = self.storage_service.store_document(
                file_path=tmp_file_path,
                processing_result=processing_result,
                db=self.db
            )
            assert success1
            
            # Try to store same document again
            success2 = self.storage_service.store_document(
                file_path=tmp_file_path,
                processing_result=processing_result,
                db=self.db
            )
            assert not success2  # Should fail due to duplicate checksum
            
        finally:
            os.unlink(tmp_file_path)
    
    def test_get_document(self):
        """Test retrieving document by ID"""
        # First store a document
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(b"test pdf content")
            tmp_file_path = tmp_file.name
        
        try:
            processing_result = self.create_test_processing_result()
            self.storage_service.store_document(tmp_file_path, processing_result, self.db)
            
            # Retrieve document
            doc = self.storage_service.get_document("doc_test123", self.db)
            
            assert doc is not None
            assert doc.document_id == "doc_test123"
            assert doc.filename == "test.pdf"
            
            # Test non-existent document
            non_existent = self.storage_service.get_document("non_existent", self.db)
            assert non_existent is None
            
        finally:
            os.unlink(tmp_file_path)
    
    def test_get_document_chunks(self):
        """Test retrieving document chunks"""
        # First store a document
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(b"test pdf content")
            tmp_file_path = tmp_file.name
        
        try:
            processing_result = self.create_test_processing_result()
            self.storage_service.store_document(tmp_file_path, processing_result, self.db)
            
            # Retrieve chunks
            chunks = self.storage_service.get_document_chunks("doc_test123", self.db)
            
            assert len(chunks) == 2
            assert chunks[0].chunk_index == 0
            assert chunks[1].chunk_index == 1
            assert chunks[0].content == "Page 1 content"
            
        finally:
            os.unlink(tmp_file_path)
    
    def test_list_documents(self):
        """Test listing documents with pagination"""
        # Store multiple documents
        for i in range(3):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                tmp_file.write(f"test pdf content {i}".encode())
                tmp_file_path = tmp_file.name
            
            try:
                processing_result = self.create_test_processing_result()
                processing_result.document_metadata.document_id = f"doc_test{i}"
                processing_result.document_metadata.checksum = f"checksum{i}"
                processing_result.document_metadata.filename = f"test{i}.pdf"
                
                # Update chunk IDs to be unique
                for j, chunk in enumerate(processing_result.chunks):
                    chunk.document_id = f"doc_test{i}"
                    chunk.chunk_id = f"doc_test{i}_chunk_{j:04d}"
                
                self.storage_service.store_document(tmp_file_path, processing_result, self.db)
            finally:
                os.unlink(tmp_file_path)
        
        # List all documents
        docs = self.storage_service.list_documents(self.db)
        assert len(docs) == 3
        
        # Test pagination
        docs_page1 = self.storage_service.list_documents(self.db, skip=0, limit=2)
        assert len(docs_page1) == 2
        
        docs_page2 = self.storage_service.list_documents(self.db, skip=2, limit=2)
        assert len(docs_page2) == 1
    
    def test_delete_document(self):
        """Test document deletion"""
        # First store a document
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(b"test pdf content")
            tmp_file_path = tmp_file.name
        
        try:
            processing_result = self.create_test_processing_result()
            self.storage_service.store_document(tmp_file_path, processing_result, self.db)
            
            # Verify document exists
            doc = self.storage_service.get_document("doc_test123", self.db)
            assert doc is not None
            
            # Delete document
            success = self.storage_service.delete_document("doc_test123", self.db)
            assert success
            
            # Verify document is gone
            doc = self.storage_service.get_document("doc_test123", self.db)
            assert doc is None
            
            # Verify chunks are gone
            chunks = self.storage_service.get_document_chunks("doc_test123", self.db)
            assert len(chunks) == 0
            
            # Test deleting non-existent document
            success = self.storage_service.delete_document("non_existent", self.db)
            assert not success
            
        finally:
            os.unlink(tmp_file_path)
    
    def test_get_storage_stats(self):
        """Test storage statistics"""
        # Initially empty
        stats = self.storage_service.get_storage_stats(self.db)
        assert stats["total_documents"] == 0
        assert stats["total_chunks"] == 0
        
        # Store a document
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(b"test pdf content")
            tmp_file_path = tmp_file.name
        
        try:
            processing_result = self.create_test_processing_result()
            self.storage_service.store_document(tmp_file_path, processing_result, self.db)
            
            # Check updated stats
            stats = self.storage_service.get_storage_stats(self.db)
            assert stats["total_documents"] == 1
            assert stats["total_chunks"] == 2
            assert stats["total_storage_bytes"] > 0
            
        finally:
            os.unlink(tmp_file_path)