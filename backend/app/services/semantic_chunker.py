"""
Semantic chunking service for intelligent text splitting
"""
import re
from typing import List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Text chunk with metadata"""
    chunk_id: str
    document_id: str
    content: str
    page_number: int
    start_char: int
    end_char: int
    chunk_index: int


class SemanticChunker:
    """
    Intelligent text chunker that maintains semantic coherence
    using sentence boundaries and semantic relationships
    """
    
    def __init__(self, 
                 max_chunk_size: int = 1000,
                 min_chunk_size: int = 100,
                 overlap_size: int = 50):
        """
        Initialize semantic chunker
        
        Args:
            max_chunk_size: Maximum characters per chunk
            min_chunk_size: Minimum characters per chunk
            overlap_size: Character overlap between chunks
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size
        
        # Sentence boundary patterns
        self.sentence_endings = re.compile(r'[.!?]+\s+')
        self.paragraph_breaks = re.compile(r'\n\s*\n')
    
    def chunk_document(self, 
                      document_id: str,
                      pages: List['PageContent']) -> List[Chunk]:
        """
        Chunk document pages into semantically coherent chunks
        
        Args:
            document_id: Unique document identifier
            pages: List of PageContent objects
            
        Returns:
            List of Chunk objects with semantic boundaries
        """
        all_chunks = []
        chunk_counter = 0
        
        for page in pages:
            if not page.text.strip():
                continue
                
            page_chunks = self._chunk_page_text(
                document_id=document_id,
                page_number=page.page_number,
                text=page.text,
                start_chunk_index=chunk_counter
            )
            
            all_chunks.extend(page_chunks)
            chunk_counter += len(page_chunks)
        
        logger.info(f"Created {len(all_chunks)} semantic chunks for document {document_id}")
        return all_chunks
    
    def _chunk_page_text(self, 
                        document_id: str,
                        page_number: int,
                        text: str,
                        start_chunk_index: int) -> List[Chunk]:
        """
        Chunk text from a single page using semantic boundaries
        
        Args:
            document_id: Document identifier
            page_number: Page number
            text: Text content to chunk
            start_chunk_index: Starting index for chunk numbering
            
        Returns:
            List of chunks for this page
        """
        if not text.strip():
            return []
        
        # First, try to split by paragraphs
        paragraphs = self._split_by_paragraphs(text)
        
        chunks = []
        current_chunk = ""
        current_start_char = 0
        chunk_index = start_chunk_index
        
        for paragraph in paragraphs:
            # If paragraph alone exceeds max size, split by sentences
            if len(paragraph) > self.max_chunk_size:
                # Process any accumulated chunk first
                if current_chunk.strip():
                    chunk = self._create_chunk(
                        document_id=document_id,
                        page_number=page_number,
                        content=current_chunk.strip(),
                        start_char=current_start_char,
                        chunk_index=chunk_index
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    current_chunk = ""
                
                # Split large paragraph by sentences
                sentence_chunks = self._split_large_paragraph(
                    document_id=document_id,
                    page_number=page_number,
                    paragraph=paragraph,
                    start_char=text.find(paragraph),
                    start_chunk_index=chunk_index
                )
                chunks.extend(sentence_chunks)
                chunk_index += len(sentence_chunks)
                current_start_char = text.find(paragraph) + len(paragraph)
                
            else:
                # Check if adding this paragraph would exceed max size
                potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
                
                if len(potential_chunk) > self.max_chunk_size and current_chunk:
                    # Create chunk with current content
                    chunk = self._create_chunk(
                        document_id=document_id,
                        page_number=page_number,
                        content=current_chunk.strip(),
                        start_char=current_start_char,
                        chunk_index=chunk_index
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    
                    # Start new chunk with overlap
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + paragraph
                    current_start_char = text.find(paragraph) - len(overlap_text)
                else:
                    # Add paragraph to current chunk
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
                        current_start_char = text.find(paragraph)
        
        # Handle remaining content
        if current_chunk.strip():
            chunk = self._create_chunk(
                document_id=document_id,
                page_number=page_number,
                content=current_chunk.strip(),
                start_char=current_start_char,
                chunk_index=chunk_index
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split text by paragraph breaks"""
        paragraphs = self.paragraph_breaks.split(text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_large_paragraph(self, 
                              document_id: str,
                              page_number: int,
                              paragraph: str,
                              start_char: int,
                              start_chunk_index: int) -> List[Chunk]:
        """Split large paragraph by sentences"""
        sentences = self._split_by_sentences(paragraph)
        chunks = []
        current_chunk = ""
        current_start = start_char
        chunk_index = start_chunk_index
        
        for sentence in sentences:
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) > self.max_chunk_size and current_chunk:
                # Create chunk with current sentences
                chunk = self._create_chunk(
                    document_id=document_id,
                    page_number=page_number,
                    content=current_chunk.strip(),
                    start_char=current_start,
                    chunk_index=chunk_index
                )
                chunks.append(chunk)
                chunk_index += 1
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + sentence
                current_start = start_char + paragraph.find(sentence) - len(overlap_text)
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                    current_start = start_char + paragraph.find(sentence)
        
        # Handle remaining content
        if current_chunk.strip():
            chunk = self._create_chunk(
                document_id=document_id,
                page_number=page_number,
                content=current_chunk.strip(),
                start_char=current_start,
                chunk_index=chunk_index
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text by sentence boundaries"""
        sentences = self.sentence_endings.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from end of current chunk"""
        if len(text) <= self.overlap_size:
            return text
        
        # Try to find sentence boundary for clean overlap
        overlap_candidate = text[-self.overlap_size:]
        sentence_match = self.sentence_endings.search(overlap_candidate)
        
        if sentence_match:
            # Use text after last sentence boundary
            return overlap_candidate[sentence_match.end():]
        else:
            # Use last N characters
            return overlap_candidate
    
    def _create_chunk(self, 
                     document_id: str,
                     page_number: int,
                     content: str,
                     start_char: int,
                     chunk_index: int) -> Chunk:
        """Create a chunk object with metadata"""
        chunk_id = f"{document_id}_chunk_{chunk_index:04d}"
        
        return Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            content=content,
            page_number=page_number,
            start_char=start_char,
            end_char=start_char + len(content),
            chunk_index=chunk_index
        )