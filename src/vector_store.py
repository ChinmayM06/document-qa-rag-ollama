from typing import List, Tuple, Optional
import numpy as np
import faiss
import pickle
from pathlib import Path
from langchain.schema import Document
from config.settings import VECTOR_DB_DIR


class VectorStore:
    """Manages FAISS vector database for document retrieval."""
    
    def __init__(self, embedding_dimension: int):
        """
        Initialize FAISS vector store.
        
        Args:
            embedding_dimension: Dimension of the embedding vectors
        """
        self.embedding_dimension = embedding_dimension
        self.index = faiss.IndexFlatL2(embedding_dimension)  # L2 distance
        self.documents = []  # Store original documents
        self.doc_id_to_index = {}  # Map document IDs to index positions
        
    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the vector store.
        
        Args:
            documents: List of LangChain Document objects
            embeddings: Numpy array of embeddings (n_docs, embedding_dim)
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match")
        
        # Ensure embeddings are float32 (required by FAISS)
        embeddings = embeddings.astype('float32')
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store documents
        start_id = len(self.documents)
        for i, doc in enumerate(documents):
            self.documents.append(doc)
            self.doc_id_to_index[start_id + i] = start_id + i
        
        print(f"Added {len(documents)} documents. Total documents: {len(self.documents)}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of tuples (Document, distance_score)
        """
        if self.index.ntotal == 0:
            raise ValueError("Vector store is empty. Add documents first.")
        
        # Ensure query embedding is the right shape and type
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        # Retrieve documents
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):  # Valid index
                results.append((self.documents[idx], float(dist)))
        
        return results
    
    def save(self, path: Optional[str] = None):
        """
        Save vector store to disk.
        
        Args:
            path: Directory path to save the store
        """
        if path is None:
            path = VECTOR_DB_DIR
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = path / "faiss_index.bin"
        faiss.write_index(self.index, str(index_path))
        
        # Save documents
        docs_path = path / "documents.pkl"
        with open(docs_path, 'wb') as f:
            pickle.dump(self.documents, f)
        
        # Save metadata
        metadata_path = path / "metadata.pkl"
        metadata = {
            'embedding_dimension': self.embedding_dimension,
            'doc_id_to_index': self.doc_id_to_index
        }
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Vector store saved to {path}")
    
    @classmethod
    def load(cls, path: Optional[str] = None):
        """
        Load vector store from disk.
        
        Args:
            path: Directory path to load from
            
        Returns:
            VectorStore instance
        """
        if path is None:
            path = VECTOR_DB_DIR
        
        path = Path(path)
        
        # Load metadata
        metadata_path = path / "metadata.pkl"
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Create instance
        store = cls(metadata['embedding_dimension'])
        
        # Load FAISS index
        index_path = path / "faiss_index.bin"
        store.index = faiss.read_index(str(index_path))
        
        # Load documents
        docs_path = path / "documents.pkl"
        with open(docs_path, 'rb') as f:
            store.documents = pickle.load(f)
        
        store.doc_id_to_index = metadata['doc_id_to_index']
        
        print(f"Vector store loaded from {path}")
        print(f"Total documents: {len(store.documents)}")
        
        return store
    
    def clear(self):
        """Clear all data from the vector store."""
        self.index.reset()
        self.documents = []
        self.doc_id_to_index = {}
        print("Vector store cleared")
    
    def get_total_documents(self) -> int:
        """Return total number of documents in the store."""
        return len(self.documents)


# Test function
if __name__ == "__main__":
    from src.embeddings import EmbeddingGenerator
    from src.document_processor import DocumentProcessor
    
    print("Testing Vector Store...")
    
    # Initialize components
    embedder = EmbeddingGenerator()
    processor = DocumentProcessor()
    
    # Create sample documents
    sample_texts = [
        "Python is a high-level programming language.",
        "Machine learning is a subset of artificial intelligence.",
        "FAISS is a library for efficient similarity search.",
        "Vector databases are used for semantic search.",
        "Natural language processing deals with text data."
    ]
    
    documents = [Document(page_content=text, metadata={"source": "test"}) 
                 for text in sample_texts]
    
    # Generate embeddings
    print("\nGenerating embeddings...")
    embeddings = embedder.generate_embeddings_from_documents(documents)
    
    # Create vector store
    print("\nCreating vector store...")
    vector_store = VectorStore(embedder.get_embedding_dimension())
    vector_store.add_documents(documents, embeddings)
    
    # Test search
    query = "What is machine learning?"
    print(f"\nQuery: {query}")
    query_embedding = embedder.generate_embedding(query)
    
    results = vector_store.search(query_embedding, top_k=3)
    
    print("\nTop 3 results:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. [Score: {score:.4f}] {doc.page_content}")
    
    # Test save/load
    print("\nTesting save/load...")
    vector_store.save()
    
    loaded_store = VectorStore.load()
    print(f"Loaded store has {loaded_store.get_total_documents()} documents")