"""
Tests for semantic chunking functionality
"""
import pytest
from dataclasses import dataclass

from app.services.semantic_chunker import SemanticChunker, Chunk


@dataclass
class MockPageContent:
    """Mock page content for testing"""
    page_number: int
    text: str
    char_count: int


class TestSemanticChunker:
    """Test cases for semantic chunking functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.chunker = SemanticChunker(
            max_chunk_size=100,  # Small size for testing
            min_chunk_size=20,
            overlap_size=10
        )
    
    def test_chunk_empty_document(self):
        """Test chunking empty document"""
        pages = [MockPageContent(1, "", 0)]
        chunks = self.chunker.chunk_document("doc_123", pages)
        
        assert len(chunks) == 0
    
    def test_chunk_single_short_page(self):
        """Test chunking single page with short content"""
        text = "This is a short paragraph that fits in one chunk."
        pages = [MockPageContent(1, text, len(text))]
        
        chunks = self.chunker.chunk_document("doc_123", pages)
        
        assert len(chunks) == 1
        assert chunks[0].document_id == "doc_123"
        assert chunks[0].page_number == 1
        assert chunks[0].content == text
        assert chunks[0].chunk_id == "doc_123_chunk_0000"
        assert chunks[0].chunk_index == 0
    
    def test_chunk_long_paragraph(self):
        """Test chunking long paragraph that needs splitting"""
        # Create text longer than max_chunk_size (100)
        text = "This is a very long paragraph. " * 10  # ~310 characters
        pages = [MockPageContent(1, text, len(text))]
        
        chunks = self.chunker.chunk_document("doc_123", pages)
        
        assert len(chunks) > 1
        assert all(chunk.document_id == "doc_123" for chunk in chunks)
        assert all(chunk.page_number == 1 for chunk in chunks)
        
        # Check that chunks respect sentence boundaries
        for chunk in chunks:
            assert len(chunk.content) <= self.chunker.max_chunk_size + 50  # Allow some flexibility
    
    def test_chunk_multiple_paragraphs(self):
        """Test chunking multiple paragraphs"""
        text = """First paragraph is here.

Second paragraph is here.

Third paragraph is here."""
        
        pages = [MockPageContent(1, text, len(text))]
        chunks = self.chunker.chunk_document("doc_123", pages)
        
        assert len(chunks) >= 1
        # Should maintain paragraph structure
        assert "First paragraph" in chunks[0].content
    
    def test_chunk_multiple_pages(self):
        """Test chunking across multiple pages"""
        pages = [
            MockPageContent(1, "Content from page one.", 22),
            MockPageContent(2, "Content from page two.", 22),
            MockPageContent(3, "Content from page three.", 24)
        ]
        
        chunks = self.chunker.chunk_document("doc_123", pages)
        
        # Should have chunks from different pages
        page_numbers = {chunk.page_number for chunk in chunks}
        assert 1 in page_numbers
        assert 2 in page_numbers
        assert 3 in page_numbers
    
    def test_split_by_paragraphs(self):
        """Test paragraph splitting functionality"""
        text = "Para 1.\n\nPara 2.\n\nPara 3."
        paragraphs = self.chunker._split_by_paragraphs(text)
        
        assert len(paragraphs) == 3
        assert "Para 1." in paragraphs[0]
        assert "Para 2." in paragraphs[1]
        assert "Para 3." in paragraphs[2]
    
    def test_split_by_sentences(self):
        """Test sentence splitting functionality"""
        text = "First sentence. Second sentence! Third sentence?"
        sentences = self.chunker._split_by_sentences(text)
        
        assert len(sentences) >= 2  # Should split on sentence boundaries
        assert any("First sentence" in s for s in sentences)
    
    def test_get_overlap_text(self):
        """Test overlap text extraction"""
        text = "This is a long text that should be used for overlap testing."
        overlap = self.chunker._get_overlap_text(text)
        
        assert len(overlap) <= self.chunker.overlap_size
        assert overlap in text
    
    def test_create_chunk(self):
        """Test chunk creation with metadata"""
        chunk = self.chunker._create_chunk(
            document_id="doc_123",
            page_number=2,
            content="Test content",
            start_char=10,
            chunk_index=5
        )
        
        assert chunk.chunk_id == "doc_123_chunk_0005"
        assert chunk.document_id == "doc_123"
        assert chunk.page_number == 2
        assert chunk.content == "Test content"
        assert chunk.start_char == 10
        assert chunk.end_char == 10 + len("Test content")
        assert chunk.chunk_index == 5
    
    def test_chunk_preserves_metadata(self):
        """Test that chunking preserves page and position metadata"""
        text = "Page content that will be chunked."
        pages = [MockPageContent(3, text, len(text))]
        
        chunks = self.chunker.chunk_document("doc_456", pages)
        
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.page_number == 3
        assert chunk.start_char >= 0
        assert chunk.end_char > chunk.start_char
        assert chunk.document_id == "doc_456"