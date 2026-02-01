"""
Vector Store Module for ManualQ
==============================
Manages FAISS vector index creation, storage, and similarity search.
"""

import numpy as np
import faiss
import pickle
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json


class VectorStore:
    """FAISS-based vector storage and retrieval."""
    
    def __init__(self, embedding_dim: int = 384):
        """
        Initialize vector store.
        
        Args:
            embedding_dim: Dimension of embeddings (default for sentence-transformers)
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.chunks = {}
        self.chunk_id = 0
        self.metadata = {}
    
    def add_chunks(
        self,
        chunks: List[Dict],
        embeddings: np.ndarray
    ) -> None:
        """
        Add chunks and their embeddings to vector store.
        
        Args:
            chunks: List of chunk dictionaries with text and metadata
            embeddings: Numpy array of embedding vectors
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match embeddings")
        
        # Add to FAISS index
        embeddings = embeddings.astype('float32')
        self.index.add(embeddings)
        
        # Store chunk metadata
        for chunk in chunks:
            self.chunks[self.chunk_id] = chunk
            self.chunk_id += 1
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> Tuple[List[int], List[float], List[Dict]]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            Tuple of (chunk_ids, distances, chunk_metadata)
        """
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)
        
        retrieved_chunks = []
        for idx, distance in zip(indices[0], distances[0]):
            idx = int(idx)
            if idx in self.chunks:
                chunk = self.chunks[idx].copy()
                chunk['distance'] = float(distance)
                retrieved_chunks.append(chunk)
        
        return indices[0], distances[0], retrieved_chunks
    
    def save(self, save_path: str) -> None:
        """Save vector store to disk."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(save_path / "faiss.index"))
        
        # Save metadata
        with open(save_path / "chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)
        
        with open(save_path / "metadata.json", 'w') as f:
            json.dump({
                'embedding_dim': self.embedding_dim,
                'total_chunks': len(self.chunks)
            }, f)
    
    def load(self, load_path: str) -> None:
        """Load vector store from disk."""
        load_path = Path(load_path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(load_path / "faiss.index"))
        
        # Load metadata
        with open(load_path / "chunks.pkl", 'rb') as f:
            self.chunks = pickle.load(f)
        
        with open(load_path / "metadata.json", 'r') as f:
            metadata = json.load(f)
            self.embedding_dim = metadata['embedding_dim']
            self.chunk_id = metadata['total_chunks']
    
    def get_stats(self) -> Dict:
        """Get vector store statistics."""
        return {
            'total_chunks': len(self.chunks),
            'embedding_dimension': self.embedding_dim,
            'index_size': self.index.ntotal,
        }


class EmbeddingModel:
    """Wrapper for sentence-transformers embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model.
        
        Args:
            model_name: Name of sentence-transformers model
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
        except ImportError:
            print("sentence-transformers not installed. Using dummy embeddings.")
            self.model = None
            self.model_name = model_name
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if self.model is None:
            # Dummy embedding for testing
            return np.random.randn(384).astype('float32')
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype('float32')
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            Array of embedding vectors
        """
        if self.model is None:
            # Dummy embeddings for testing
            return np.random.randn(len(texts), 384).astype('float32')
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        return embeddings.astype('float32')
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        if self.model is None:
            return 384
        return self.model.get_sentence_embedding_dimension()


if __name__ == "__main__":
    print("Vector Store module initialized.")
