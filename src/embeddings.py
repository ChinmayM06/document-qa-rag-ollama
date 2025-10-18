from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from config.settings import EMBEDDING_MODEL


class EmbeddingGenerator:
    """Handles text embedding generation using SentenceTransformers."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize the embedding model.  
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        self.model_name = model_name
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dimension}")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text string
            
        Returns:
            numpy array of embeddings
        """
        if not text or not text.strip():
            raise ValueError("Empty text provided for embedding")
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def generate_embeddings(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of text strings
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if not texts:
            raise ValueError("Empty text list provided")
        
        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        
        if not valid_texts:
            raise ValueError("All texts are empty")
        
        embeddings = self.model.encode(
            valid_texts,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            batch_size=32
        )
        
        return embeddings
    
    def generate_embeddings_from_documents(self, documents: List[Document], 
                                          show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings from LangChain Document objects.
        
        Args:
            documents: List of LangChain Document objects
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of embeddings
        """
        texts = [doc.page_content for doc in documents]
        return self.generate_embeddings(texts, show_progress)
    
    def get_embedding_dimension(self) -> int:
        """Return the dimension of the embeddings."""
        return self.embedding_dimension


# Test function
if __name__ == "__main__":
    # Initialize embedding generator
    embedder = EmbeddingGenerator()
    
    # Test with sample texts
    sample_texts = [
        "This is the first test sentence.",
        "This is another test sentence with different content.",
        "Machine learning is a subset of artificial intelligence."
    ]
    
    print("\nGenerating embeddings for sample texts...")
    embeddings = embedder.generate_embeddings(sample_texts)
    
    print(f"\nGenerated embeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {embedder.get_embedding_dimension()}")
    print(f"First embedding preview: {embeddings[0][:5]}...")
    
    # Test similarity
    from numpy.linalg import norm
    
    def cosine_similarity(a, b):
        return np.dot(a, b) / (norm(a) * norm(b))
    
    sim_1_2 = cosine_similarity(embeddings[0], embeddings[1])
    sim_1_3 = cosine_similarity(embeddings[0], embeddings[2])
    
    print(f"\nSimilarity between text 1 and 2: {sim_1_2:.4f}")
    print(f"Similarity between text 1 and 3: {sim_1_3:.4f}")