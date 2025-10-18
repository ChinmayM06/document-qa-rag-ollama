# src/retriever.py
from typing import List, Tuple, Optional
from langchain.schema import Document
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from config.settings import TOP_K, SIMILARITY_THRESHOLD


class Retriever:
    """Handles document retrieval from vector store."""
    
    def __init__(self, 
                 vector_store: VectorStore,
                 embedding_generator: EmbeddingGenerator,
                 top_k: int = TOP_K,
                 similarity_threshold: Optional[float] = SIMILARITY_THRESHOLD):
        """
        Initialize retriever.
        
        Args:
            vector_store: Vector store instance
            embedding_generator: Embedding generator instance
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score (None to disable)
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Override default top_k
            
        Returns:
            List of (Document, score) tuples
        """
        if not query or not query.strip():
            raise ValueError("Empty query provided")
        
        # Use provided top_k or default
        k = top_k if top_k is not None else self.top_k
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, top_k=k)
        
        # Filter by similarity threshold if set
        if self.similarity_threshold is not None:
            results = [(doc, score) for doc, score in results 
                      if score <= self.similarity_threshold]  # Lower L2 distance = more similar
        
        return results
    
    def retrieve_with_scores(self, query: str, top_k: Optional[int] = None) -> dict:
        """
        Retrieve documents with detailed score information.
        
        Args:
            query: Search query
            top_k: Override default top_k
            
        Returns:
            Dictionary with documents, scores, and metadata
        """
        results = self.retrieve(query, top_k)
        
        return {
            'query': query,
            'num_results': len(results),
            'results': [
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': score
                }
                for doc, score in results
            ]
        }
    
    def get_context_string(self, query: str, top_k: Optional[int] = None) -> str:
        """
        Get formatted context string for a query.
        
        Args:
            query: Search query
            top_k: Override default top_k
            
        Returns:
            Formatted context string
        """
        results = self.retrieve(query, top_k)
        
        if not results:
            return "No relevant context found."
        
        context_parts = []
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get('source', 'Unknown')
            context_parts.append(
                f"[Source {i}: {source} | Relevance: {1/(1+score):.2f}]\n{doc.page_content}"
            )
        
        return "\n\n".join(context_parts)


# Test function
if __name__ == "__main__":
    from src.document_processor import DocumentProcessor
    
    print("Testing Retriever...\n")
    
    # Initialize components
    print("Initializing components...")
    embedder = EmbeddingGenerator()
    processor = DocumentProcessor()
    
    # Create sample documents
    sample_texts = [
        "Python is a high-level programming language created by Guido van Rossum in 1991.",
        "Machine learning is a subset of artificial intelligence that focuses on learning from data.",
        "FAISS is a library developed by Facebook AI for efficient similarity search and clustering.",
        "Vector databases store data as high-dimensional vectors for semantic search.",
        "Natural language processing (NLP) is a field of AI that deals with text and language data.",
        "Deep learning uses neural networks with multiple layers to learn complex patterns.",
        "Transformers are a type of neural network architecture used in modern NLP models.",
        "Embeddings represent text as dense vectors in a continuous space."
    ]
    
    documents = [Document(page_content=text, metadata={"source": f"doc_{i}.txt"}) 
                 for i, text in enumerate(sample_texts)]
    
    # Generate embeddings and create vector store
    print("Creating vector store...")
    embeddings = embedder.generate_embeddings_from_documents(documents, show_progress=False)
    vector_store = VectorStore(embedder.get_embedding_dimension())
    vector_store.add_documents(documents, embeddings)
    
    # Create retriever
    retriever = Retriever(vector_store, embedder, top_k=3)
    
    # Test queries
    test_queries = [
        "What is Python?",
        "Tell me about machine learning",
        "How do vector databases work?",
        "Explain neural networks"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        # Get results
        results = retriever.retrieve(query)
        
        print(f"\nFound {len(results)} relevant documents:\n")
        for i, (doc, score) in enumerate(results, 1):
            print(f"{i}. [Score: {score:.4f}] {doc.page_content[:100]}...")
        
        # Get formatted context
        print("\n--- Formatted Context ---")
        context = retriever.get_context_string(query)
        print(context[:300] + "...")