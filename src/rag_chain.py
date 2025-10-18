# src/rag_chain.py
from typing import List, Optional, Dict, Any
from langchain.schema import Document
from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.llm_handler import LLMHandler
from utils.prompt_templates import get_qa_prompt_template, format_context
from config.settings import TOP_K


class RAGChain:
    """Complete RAG (Retrieval-Augmented Generation) pipeline."""
    
    def __init__(self):
        """Initialize all components of the RAG chain."""
        print("Initializing RAG Chain...")
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.embedding_generator = EmbeddingGenerator()
        self.llm_handler = LLMHandler()
        self.vector_store = None
        self.retriever = None
        
        print("✓ RAG Chain initialized")
    
    def load_document(self, file_path: str) -> int:
        """
        Load and process a document into the vector store.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Number of chunks created
        """
        print(f"\nProcessing document: {file_path}")
        
        # Process document into chunks
        chunks = self.document_processor.process_document(file_path)
        print(f"✓ Created {len(chunks)} chunks")
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embedding_generator.generate_embeddings_from_documents(
            chunks, 
            show_progress=True
        )
        print(f"✓ Generated embeddings of shape {embeddings.shape}")
        
        # Create or update vector store
        if self.vector_store is None:
            print("Creating new vector store...")
            self.vector_store = VectorStore(
                self.embedding_generator.get_embedding_dimension()
            )
        
        # Add documents to vector store
        self.vector_store.add_documents(chunks, embeddings)
        
        # Create/update retriever
        self.retriever = Retriever(
            self.vector_store,
            self.embedding_generator,
            top_k=TOP_K
        )
        
        print(f"✓ Document loaded successfully")
        return len(chunks)
    
    def load_multiple_documents(self, file_paths: List[str]) -> Dict[str, int]:
        """
        Load multiple documents.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Dictionary mapping file paths to number of chunks
        """
        results = {}
        
        for file_path in file_paths:
            try:
                num_chunks = self.load_document(file_path)
                results[file_path] = num_chunks
            except Exception as e:
                print(f"✗ Error loading {file_path}: {e}")
                results[file_path] = 0
        
        return results
    
    def query(self, 
              question: str, 
              top_k: Optional[int] = None,
              return_sources: bool = True) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: User's question
            top_k: Number of documents to retrieve (override default)
            return_sources: Whether to return source documents
            
        Returns:
            Dictionary with answer and metadata
        """
        if self.retriever is None:
            raise ValueError("No documents loaded. Please load documents first.")
        
        if not question or not question.strip():
            raise ValueError("Empty question provided")
        
        # Retrieve relevant documents
        print(f"\nQuery: {question}")
        print("Retrieving relevant context...")
        retrieved_docs = self.retriever.retrieve(question, top_k=top_k)
        
        if not retrieved_docs:
            return {
                'question': question,
                'answer': "I couldn't find any relevant information in the documents to answer your question.",
                'sources': [],
                'num_sources': 0
            }
        
        print(f"✓ Retrieved {len(retrieved_docs)} relevant chunks")
        
        # Format context
        context = format_context(retrieved_docs, include_metadata=True)
        
        # Get prompt template
        prompt_template = get_qa_prompt_template()
        prompt = prompt_template.format(context=context, question=question)
        
        # Generate answer
        print("Generating answer...")
        answer = self.llm_handler.generate(prompt)
        print("✓ Answer generated")
        
        # Prepare response
        response = {
            'question': question,
            'answer': answer.strip(),
            'num_sources': len(retrieved_docs)
        }
        
        if return_sources:
            response['sources'] = [
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': score
                }
                for doc, score in retrieved_docs
            ]
        
        return response
    
    def chat(self, question: str, top_k: Optional[int] = None) -> str:
        """
        Simple chat interface - returns just the answer.
        
        Args:
            question: User's question
            top_k: Number of documents to retrieve
            
        Returns:
            Answer string
        """
        result = self.query(question, top_k, return_sources=False)
        return result['answer']
    
    def save_vector_store(self, path: Optional[str] = None):
        """Save the vector store to disk."""
        if self.vector_store is None:
            raise ValueError("No vector store to save")
        
        self.vector_store.save(path)
    
    def load_vector_store(self, path: Optional[str] = None):
        """Load a vector store from disk."""
        self.vector_store = VectorStore.load(path)
        self.retriever = Retriever(
            self.vector_store,
            self.embedding_generator,
            top_k=TOP_K
        )
        print("✓ Vector store loaded and retriever initialized")
    
    def clear(self):
        """Clear all loaded documents."""
        if self.vector_store:
            self.vector_store.clear()
        self.retriever = None
        print("✓ All documents cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the current state."""
        if self.vector_store is None:
            return {
                'total_documents': 0,
                'embedding_dimension': self.embedding_generator.get_embedding_dimension(),
                'model_name': self.embedding_generator.model_name,
                'llm_model': self.llm_handler.model_name
            }
        
        return {
            'total_documents': self.vector_store.get_total_documents(),
            'embedding_dimension': self.embedding_generator.get_embedding_dimension(),
            'model_name': self.embedding_generator.model_name,
            'llm_model': self.llm_handler.model_name
        }


# Test function
if __name__ == "__main__":
    import tempfile
    import os
    
    print("Testing RAG Chain...\n")
    
    # Initialize RAG chain
    rag = RAGChain()
    
    # Create a temporary test document
    test_content = """
    Python Programming Language
    
    Python is a high-level, interpreted programming language created by Guido van Rossum.
    It was first released in 1991 and has since become one of the most popular programming languages.
    
    Python is known for its simplicity and readability, making it an excellent choice for beginners.
    It supports multiple programming paradigms including procedural, object-oriented, and functional programming.
    
    Key Features:
    - Easy to learn and read
    - Extensive standard library
    - Strong community support
    - Great for web development, data science, AI, and automation
    
    Popular Python frameworks include Django for web development, NumPy and Pandas for data analysis,
    and TensorFlow and PyTorch for machine learning.
    """
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        temp_file = f.name
    
    try:
        # Load document
        print("="*60)
        print("Step 1: Loading Document")
        print("="*60)
        num_chunks = rag.load_document(temp_file)
        print(f"\nDocument loaded with {num_chunks} chunks")
        
        # Get stats
        print("\n" + "="*60)
        print("Step 2: System Statistics")
        print("="*60)
        stats = rag.get_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Test queries
        test_questions = [
            "Who created Python?",
            "What are the key features of Python?",
            "What frameworks are mentioned for machine learning?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print("\n" + "="*60)
            print(f"Step {i+2}: Query {i}")
            print("="*60)
            
            result = rag.query(question, return_sources=True)
            
            print(f"\nQuestion: {result['question']}")
            print(f"\nAnswer: {result['answer']}")
            print(f"\nNumber of sources used: {result['num_sources']}")
            
            if result.get('sources'):
                print("\nTop source:")
                print(f"- {result['sources'][0]['content'][:150]}...")
        
        # Test save/load
        print("\n" + "="*60)
        print("Step 6: Testing Save/Load")
        print("="*60)
        rag.save_vector_store()
        print("✓ Vector store saved")
        
        # Clear and reload
        rag.clear()
        rag.load_vector_store()
        print("✓ Vector store reloaded")
        
        # Test simple chat interface
        print("\n" + "="*60)
        print("Step 7: Testing Simple Chat Interface")
        print("="*60)
        answer = rag.chat("What year was Python released?")
        print(f"Answer: {answer}")
        
    finally:
        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    print("\n" + "="*60)
    print("✓ All tests completed successfully!")
    print("="*60)