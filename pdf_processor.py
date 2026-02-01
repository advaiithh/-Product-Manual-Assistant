"""
PDF Processor Module for ManualQ
================================
Handles PDF extraction, cleaning, and preprocessing for RAG pipeline.
"""

import PyPDF2
import pymupdf
import pdfplumber
import re
from typing import List, Dict, Tuple
from pathlib import Path


class PDFProcessor:
    """Extract and clean text from PDF documents."""
    
    def __init__(self):
        """Initialize PDF processor."""
        self.text_content = ""
        self.metadata = {}
        self.page_contents = {}
    
    def extract_from_pdf(self, pdf_path: str) -> Dict:
        """
        Extract text from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with extracted content and metadata
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Extract using multiple methods for robustness
        text_content = self._extract_with_pymupdf(str(pdf_path))
        
        # Get metadata
        self.metadata = {
            'filename': pdf_path.name,
            'file_size': pdf_path.stat().st_size,
            'total_pages': self._get_page_count(str(pdf_path)),
        }
        
        self.text_content = text_content
        
        return {
            'text': text_content,
            'metadata': self.metadata,
            'page_contents': self.page_contents
        }
    
    def _extract_with_pymupdf(self, pdf_path: str) -> str:
        """Extract text using pymupdf (fast and reliable)."""
        text_content = ""
        try:
            doc = pymupdf.open(pdf_path)
            
            for page_num, page in enumerate(doc, 1):
                text = page.get_text()
                self.page_contents[page_num] = text
                text_content += text + f"\n--- Page {page_num} ---\n"
            
            doc.close()
        except Exception as e:
            print(f"Error with pymupdf: {e}. Falling back to PyPDF2.")
            text_content = self._extract_with_pypdf2(pdf_path)
        
        return text_content
    
    def _extract_with_pypdf2(self, pdf_path: str) -> str:
        """Fallback: Extract text using PyPDF2."""
        text_content = ""
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text()
                    self.page_contents[page_num] = text
                    text_content += text + f"\n--- Page {page_num} ---\n"
        except Exception as e:
            print(f"Error with PyPDF2: {e}")
            raise
        
        return text_content
    
    def _get_page_count(self, pdf_path: str) -> int:
        """Get number of pages in PDF."""
        try:
            doc = pymupdf.open(pdf_path)
            count = len(doc)
            doc.close()
            return count
        except:
            return 0
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remove page markers
        text = re.sub(r'--- Page \d+ ---', '', text)
        
        # Remove common artifacts
        text = re.sub(r'©.*?\d{4}', '', text)
        
        return text.strip()
    
    def extract_page_ranges(self, text: str) -> Dict[int, str]:
        """Extract content by page number."""
        return self.page_contents


class TextCleaner:
    """Advanced text cleaning and preprocessing."""
    
    @staticmethod
    def remove_headers_footers(text: str) -> str:
        """Remove headers and footers from text."""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip page numbers and headers
            if re.match(r'^\d+\s*$', line.strip()):
                continue
            if re.match(r'^[A-Z\s]{5,}$', line.strip()) and len(line.strip()) < 30:
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text formatting."""
        # Fix common encoding issues
        text = text.replace('â€™', "'")
        text = text.replace('â€œ', '"')
        text = text.replace('â€\x9d', '"')
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()


if __name__ == "__main__":
    # Example usage
    processor = PDFProcessor()
    print("PDF Processor initialized and ready for use.")
