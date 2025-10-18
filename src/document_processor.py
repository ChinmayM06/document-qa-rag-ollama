# src/document_processor.py
from typing import List, Optional
from pathlib import Path
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS


class DocumentProcessor:
    """Handles document loading and chunking."""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=SEPARATORS,
            length_function=len,
        )
    
    def load_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error loading PDF: {str(e)}")
    
    def load_text(self, file_path: str) -> str:
        """Load text from .txt file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            raise Exception(f"Error loading text file: {str(e)}")
    
    def load_document(self, file_path: str) -> str:
        """Load document based on file extension."""
        path = Path(file_path)
        
        if path.suffix.lower() == '.pdf':
            return self.load_pdf(file_path)
        elif path.suffix.lower() == '.txt':
            return self.load_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
    
    def chunk_text(self, text: str, metadata: Optional[dict] = None) -> List[Document]:
        """Split text into chunks."""
        if not text.strip():
            raise ValueError("Empty document provided")
        
        # Create LangChain documents
        chunks = self.text_splitter.create_documents(
            texts=[text],
            metadatas=[metadata] if metadata else None
        )
        
        return chunks
    
    def process_document(self, file_path: str) -> List[Document]:
        """Complete pipeline: load and chunk document."""
        # Load document
        text = self.load_document(file_path)
        
        # Create metadata
        metadata = {
            "source": Path(file_path).name,
            "file_path": str(file_path)
        }
        
        # Chunk text
        chunks = self.chunk_text(text, metadata)
        
        return chunks


# Test function
if __name__ == "__main__":
    processor = DocumentProcessor()
    
    # Test with a sample text
    sample_text = "This is a test document. " * 200
    chunks = processor.chunk_text(sample_text)
    
    print(f"Created {len(chunks)} chunks")
    print(f"First chunk preview: {chunks[0].page_content[:100]}...")