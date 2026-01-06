"""
Document processing service for PDF text extraction and metadata handling
"""
import os
import hashlib
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging

import PyPDF2
import pdfplumber
from pydantic import BaseModel

from .semantic_chunker import SemanticChunker, Chunk

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Document metadata structure"""
    document_id: str
    filename: str
    file_size: int
    page_count: int
    creation_date: Optional[datetime]
    processing_date: datetime
    checksum: str


@dataclass
class PageContent:
    """Page content with metadata"""
    page_number: int
    text: str
    char_count: int


class ProcessingResult(BaseModel):
    """Result of document processing"""
    success: bool
    document_metadata: Optional[DocumentMetadata] = None
    pages: List[PageContent] = []
    chunks: List[Chunk] = []
    error_message: Optional[str] = None


class PDFProcessor:
    """PDF document processor for text extraction and metadata handling"""
    
    def __init__(self, 
                 max_chunk_size: int = 1000,
                 min_chunk_size: int = 100,
                 overlap_size: int = 50):
        self.supported_extensions = {'.pdf'}
        self.chunker = SemanticChunker(
            max_chunk_size=max_chunk_size,
            min_chunk_size=min_chunk_size,
            overlap_size=overlap_size
        )
    
    def process_pdf(self, file_path: str) -> ProcessingResult:
        """
        Process a PDF file and extract text content with metadata
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            ProcessingResult with extracted content and metadata
        """
        try:
            if not os.path.exists(file_path):
                return ProcessingResult(
                    success=False,
                    error_message=f"File not found: {file_path}"
                )
            
            if not file_path.lower().endswith('.pdf'):
                return ProcessingResult(
                    success=False,
                    error_message=f"Unsupported file type. Expected PDF, got: {file_path}"
                )
            
            # Extract metadata
            metadata = self.extract_metadata(file_path)
            if not metadata:
                return ProcessingResult(
                    success=False,
                    error_message="Failed to extract document metadata"
                )
            
            # Extract text content page by page
            pages = self.extract_text_by_pages(file_path)
            if not pages:
                return ProcessingResult(
                    success=False,
                    error_message="Failed to extract text content from PDF"
                )
            
            # Create semantic chunks
            chunks = self.chunker.chunk_document(metadata.document_id, pages)
            
            logger.info(f"Successfully processed PDF: {metadata.filename}, {len(pages)} pages, {len(chunks)} chunks")
            
            return ProcessingResult(
                success=True,
                document_metadata=metadata,
                pages=pages,
                chunks=chunks
            )
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            return ProcessingResult(
                success=False,
                error_message=f"Processing error: {str(e)}"
            )
    
    def extract_metadata(self, file_path: str) -> Optional[DocumentMetadata]:
        """
        Extract metadata from PDF file
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            DocumentMetadata object or None if extraction fails
        """
        try:
            # Get file stats
            file_stats = os.stat(file_path)
            file_size = file_stats.st_size
            filename = os.path.basename(file_path)
            
            # Calculate file checksum
            checksum = self._calculate_checksum(file_path)
            
            # Generate document ID from checksum
            document_id = f"doc_{checksum[:16]}"
            
            # Extract PDF metadata using PyPDF2
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                page_count = len(pdf_reader.pages)
                
                # Try to get creation date from PDF metadata
                creation_date = None
                if pdf_reader.metadata:
                    if '/CreationDate' in pdf_reader.metadata:
                        try:
                            # PDF dates are in format D:YYYYMMDDHHmmSSOHH'mm'
                            date_str = str(pdf_reader.metadata['/CreationDate'])
                            if date_str.startswith('D:'):
                                date_str = date_str[2:16]  # Extract YYYYMMDDHHMMSS
                                creation_date = datetime.strptime(date_str, '%Y%m%d%H%M%S')
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Could not parse creation date: {e}")
                
                # Fallback to file modification time if no creation date
                if not creation_date:
                    creation_date = datetime.fromtimestamp(file_stats.st_mtime)
            
            return DocumentMetadata(
                document_id=document_id,
                filename=filename,
                file_size=file_size,
                page_count=page_count,
                creation_date=creation_date,
                processing_date=datetime.now(),
                checksum=checksum
            )
            
        except Exception as e:
            logger.error(f"Error extracting metadata from {file_path}: {str(e)}")
            return None
    
    def extract_text_by_pages(self, file_path: str) -> List[PageContent]:
        """
        Extract text content from PDF with page-level granularity
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of PageContent objects with text and metadata
        """
        pages = []
        
        try:
            # Use pdfplumber for better text extraction
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        # Extract text from page
                        text = page.extract_text()
                        
                        # Handle empty pages
                        if not text:
                            text = ""
                        
                        # Clean up text (remove excessive whitespace)
                        text = self._clean_text(text)
                        
                        pages.append(PageContent(
                            page_number=page_num,
                            text=text,
                            char_count=len(text)
                        ))
                        
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num}: {str(e)}")
                        # Add empty page content to maintain page numbering
                        pages.append(PageContent(
                            page_number=page_num,
                            text="",
                            char_count=0
                        ))
            
            return pages
            
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            return []
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA-256 checksum of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text by removing excessive whitespace"""
        if not text:
            return ""
        
        # Replace multiple whitespace characters with single space
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text