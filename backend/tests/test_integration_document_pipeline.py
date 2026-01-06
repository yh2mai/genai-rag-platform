"""
Integration tests for the complete document processing pipeline
"""
import os
import tempfile
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.database import Base
from app.services.document_processor import PDFProcessor
from app.services.document_storage import DocumentStorageService


class TestDocumentProcessingPipeline:
    """Integration tests for the complete document processing pipeline"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Create in-memory SQLite database for testing
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=self.engine)
        
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.db = SessionLocal()
        
        # Create temporary storage directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize services
        self.processor = PDFProcessor()
        
        # Mock settings for storage service
        with pytest.MonkeyPatch().context() as m:
            m.setattr('app.services.document_storage.settings.DOCUMENT_STORE_PATH', self.temp_dir)
            self.storage_service = DocumentStorageService()
    
    def teardown_method(self):
        """Clean up test fixtures"""
        self.db.close()
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_document_processing_pipeline(self):
        """Test the complete pipeline from PDF file to stored chunks"""
        # This test would require a real PDF file, so we'll mock the processing
        # In a real scenario, you would use an actual PDF file
        
        # Create a mock PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(b"Mock PDF content for testing")
            tmp_file_path = tmp_file.name
        
        try:
            # Mock the PDF processing since we don't have a real PDF
            from unittest.mock import patch
            
            with patch.object(self.processor, 'extract_metadata') as mock_metadata, \
                 patch.object(self.processor, 'extract_text_by_pages') as mock_pages:
                
                # Mock metadata extraction
                from app.services.document_processor import DocumentMetadata
                from datetime import datetime
                
                mock_metadata.return_value = DocumentMetadata(
                    document_id="integration_test_doc",
                    filename="test_integration.pdf",
                    file_size=1024,
                    page_count=1,
                    creation_date=datetime.now(),
                    processing_date=datetime.now(),
                    checksum="integration_test_checksum"
                )
                
                # Mock page extraction
                from app.services.document_processor import PageContent
                mock_pages.return_value = [
                    PageContent(1, "This is a test document for integration testing. It contains multiple sentences to test the semantic chunking functionality.", 120)
                ]
                
                # Step 1: Process the PDF
                processing_result = self.processor.process_pdf(tmp_file_path)
                
                # Verify processing succeeded
                assert processing_result.success
                assert processing_result.document_metadata is not None
                assert len(processing_result.pages) == 1
                assert len(processing_result.chunks) >= 1
                
                # Step 2: Store the processed document
                storage_success = self.storage_service.store_document(
                    file_path=tmp_file_path,
                    processing_result=processing_result,
                    db=self.db
                )
                
                # Verify storage succeeded
                assert storage_success
                
                # Step 3: Verify document can be retrieved
                stored_doc = self.storage_service.get_document("integration_test_doc", self.db)
                assert stored_doc is not None
                assert stored_doc.filename == "test_integration.pdf"
                
                # Step 4: Verify chunks can be retrieved
                stored_chunks = self.storage_service.get_document_chunks("integration_test_doc", self.db)
                assert len(stored_chunks) >= 1
                assert all(chunk.document_id == "integration_test_doc" for chunk in stored_chunks)
                
                # Step 5: Verify storage stats
                stats = self.storage_service.get_storage_stats(self.db)
                assert stats["total_documents"] == 1
                assert stats["total_chunks"] >= 1
                
                print(f"✅ Integration test passed: Processed document with {len(stored_chunks)} chunks")
                
        finally:
            os.unlink(tmp_file_path)