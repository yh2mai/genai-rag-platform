"""
Tests for document processing functionality
"""
import os
import tempfile
import pytest
from datetime import datetime
from unittest.mock import patch, mock_open

from app.services.document_processor import PDFProcessor, DocumentMetadata, PageContent, ProcessingResult


class TestPDFProcessor:
    """Test cases for PDF processing functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.processor = PDFProcessor()
    
    def test_process_pdf_file_not_found(self):
        """Test processing non-existent file"""
        result = self.processor.process_pdf("nonexistent.pdf")
        
        assert not result.success
        assert "File not found" in result.error_message
        assert result.document_metadata is None
        assert len(result.pages) == 0
    
    def test_process_pdf_unsupported_file_type(self):
        """Test processing non-PDF file"""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_file:
            tmp_file.write(b"test content")
            tmp_file_path = tmp_file.name
        
        try:
            result = self.processor.process_pdf(tmp_file_path)
            
            assert not result.success
            assert "Unsupported file type" in result.error_message
        finally:
            os.unlink(tmp_file_path)
    
    def test_supported_extensions(self):
        """Test that processor recognizes PDF extension"""
        assert '.pdf' in self.processor.supported_extensions
    
    def test_calculate_checksum(self):
        """Test checksum calculation"""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            test_content = b"test content for checksum"
            tmp_file.write(test_content)
            tmp_file_path = tmp_file.name
        
        try:
            checksum1 = self.processor._calculate_checksum(tmp_file_path)
            checksum2 = self.processor._calculate_checksum(tmp_file_path)
            
            # Same file should produce same checksum
            assert checksum1 == checksum2
            assert len(checksum1) == 64  # SHA-256 produces 64 character hex string
        finally:
            os.unlink(tmp_file_path)
    
    def test_clean_text(self):
        """Test text cleaning functionality"""
        # Test multiple whitespace removal
        dirty_text = "This  has   multiple    spaces\n\nand\t\ttabs"
        clean_text = self.processor._clean_text(dirty_text)
        assert clean_text == "This has multiple spaces and tabs"
        
        # Test empty text
        assert self.processor._clean_text("") == ""
        assert self.processor._clean_text(None) == ""
        
        # Test whitespace stripping
        assert self.processor._clean_text("  text  ") == "text"
    
    @patch('app.services.document_processor.pdfplumber')
    @patch('app.services.document_processor.PyPDF2')
    def test_extract_metadata_success(self, mock_pypdf2, mock_pdfplumber):
        """Test successful metadata extraction"""
        # Mock file operations
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(b"fake pdf content")
            tmp_file_path = tmp_file.name
        
        try:
            # Mock PyPDF2 reader
            mock_reader = mock_pypdf2.PdfReader.return_value
            mock_reader.pages = [None, None, None]  # 3 pages
            mock_reader.metadata = {'/CreationDate': 'D:20231201120000'}
            
            metadata = self.processor.extract_metadata(tmp_file_path)
            
            assert metadata is not None
            assert metadata.filename == os.path.basename(tmp_file_path)
            assert metadata.page_count == 3
            assert metadata.file_size > 0
            assert metadata.checksum is not None
            assert metadata.document_id.startswith("doc_")
            assert isinstance(metadata.processing_date, datetime)
        finally:
            os.unlink(tmp_file_path)
    
    @patch('app.services.document_processor.pdfplumber')
    def test_extract_text_by_pages_success(self, mock_pdfplumber):
        """Test successful text extraction by pages"""
        # Create mock pages
        mock_page1 = type('MockPage', (), {})()
        mock_page1.extract_text = lambda: "Page 1 content"
        
        mock_page2 = type('MockPage', (), {})()
        mock_page2.extract_text = lambda: "Page 2 content"
        
        # Mock pdfplumber context manager
        mock_pdf = mock_pdfplumber.open.return_value.__enter__.return_value
        mock_pdf.pages = [mock_page1, mock_page2]
        
        pages = self.processor.extract_text_by_pages("fake_path.pdf")
        
        assert len(pages) == 2
        assert pages[0].page_number == 1
        assert pages[0].text == "Page 1 content"
        assert pages[0].char_count == len("Page 1 content")
        assert pages[1].page_number == 2
        assert pages[1].text == "Page 2 content"
    
    @patch('app.services.document_processor.pdfplumber')
    def test_extract_text_empty_page(self, mock_pdfplumber):
        """Test handling of empty pages"""
        # Create mock page that returns None
        mock_page = type('MockPage', (), {})()
        mock_page.extract_text = lambda: None
        
        mock_pdf = mock_pdfplumber.open.return_value.__enter__.return_value
        mock_pdf.pages = [mock_page]
        
        pages = self.processor.extract_text_by_pages("fake_path.pdf")
        
        assert len(pages) == 1
        assert pages[0].page_number == 1
        assert pages[0].text == ""
        assert pages[0].char_count == 0
    
    @patch('app.services.document_processor.pdfplumber')
    @patch('app.services.document_processor.PyPDF2')
    def test_process_pdf_success_with_chunks(self, mock_pypdf2, mock_pdfplumber):
        """Test successful PDF processing with chunking"""
        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(b"fake pdf content")
            tmp_file_path = tmp_file.name
        
        try:
            # Mock PyPDF2 reader for metadata
            mock_reader = mock_pypdf2.PdfReader.return_value
            mock_reader.pages = [None, None]  # 2 pages
            mock_reader.metadata = {'/CreationDate': 'D:20231201120000'}
            
            # Mock pdfplumber for text extraction
            mock_page1 = type('MockPage', (), {})()
            mock_page1.extract_text = lambda: "This is page one content."
            
            mock_page2 = type('MockPage', (), {})()
            mock_page2.extract_text = lambda: "This is page two content."
            
            mock_pdf = mock_pdfplumber.open.return_value.__enter__.return_value
            mock_pdf.pages = [mock_page1, mock_page2]
            
            # Process the PDF
            result = self.processor.process_pdf(tmp_file_path)
            
            assert result.success
            assert result.document_metadata is not None
            assert len(result.pages) == 2
            assert len(result.chunks) >= 1  # Should create at least one chunk
            assert result.error_message is None
            
            # Check chunk properties
            for chunk in result.chunks:
                assert chunk.document_id == result.document_metadata.document_id
                assert chunk.page_number in [1, 2]
                assert len(chunk.content) > 0
                
        finally:
            os.unlink(tmp_file_path)