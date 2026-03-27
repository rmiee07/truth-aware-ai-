"""
embedder.py
===========
Converts document chunks into dense vector embeddings and stores them
in a FAISS index for fast nearest-neighbour retrieval.

Two embedding providers are supported (set in config.py):
  - "huggingface" : sentence-transformers (runs locally, no API key needed)
  - "openai"      : text-embedding-3-small (best quality, requires API key)

FAISS index is saved to disk so you don't re-embed on every run.
"""

import os
import pickle
import numpy as np
from typing import List, Tuple

from config import EMBEDDING_PROVIDER, EMBEDDING_MODELS, RETRIEVAL
from document_loader import DocumentChunk


# ---------------------------------------------------------------------------
# Embedding backend wrappers
# ---------------------------------------------------------------------------

class HuggingFaceEmbedder:
    """
    Uses sentence-transformers to embed text locally.
    No API key or internet connection required after initial model download.

    Install: pip install sentence-transformers
    """
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Run: pip install sentence-transformers")
        print(f"[Embedder] Loading HuggingFace model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"[Embedder] Embedding dimension: {self.dimension}")

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of strings.
        Returns: float32 numpy array of shape (N, dimension)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,   # unit vectors → cosine sim = dot product
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)


class OpenAIEmbedder:
    """
    Uses OpenAI's text-embedding-3-small model via the API.
    Requires OPENAI_API_KEY set in environment.

    Install: pip install openai
    """
    def __init__(self, model_name: str):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Run: pip install openai")
        from config import OPENAI_API_KEY
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = model_name
        # Dimension for text-embedding-3-small is 1536
        self.dimension = 1536
        print(f"[Embedder] Using OpenAI model: {model_name}")

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed texts using OpenAI API (batched to respect token limits).
        Returns: float32 numpy array of shape (N, dimension)
        """
        all_embeddings = []
        batch_size = 100  # OpenAI allows up to 2048 inputs, but 100 is safe

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.model_name,
            )
            batch_vectors = [item.embedding for item in response.data]
            all_embeddings.extend(batch_vectors)

        arr = np.array(all_embeddings, dtype=np.float32)
        # Normalize for cosine similarity
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / (norms + 1e-10)
        return arr


# ---------------------------------------------------------------------------
# Factory: pick the right embedder based on config
# ---------------------------------------------------------------------------

def get_embedder():
    """Return the configured embedding model instance."""
    model_name = EMBEDDING_MODELS[EMBEDDING_PROVIDER]
    if EMBEDDING_PROVIDER == "huggingface":
        return HuggingFaceEmbedder(model_name)
    elif EMBEDDING_PROVIDER == "openai":
        return OpenAIEmbedder(model_name)
    else:
        raise ValueError(f"Unknown embedding provider: {EMBEDDING_PROVIDER}")


# ---------------------------------------------------------------------------
# FAISS index builder
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Wraps a FAISS flat index with chunk metadata storage.

    Usage:
        store = VectorStore()
        store.build(chunks)          # embed and index all chunks
        store.save()                 # persist to disk
        store.load()                 # reload from disk
        results = store.search(query_text, top_k=5)
    """

    def __init__(self):
        self.embedder = get_embedder()
        self.index = None          # FAISS index object
        self.chunks: List[DocumentChunk] = []   # parallel list of chunks

    def build(self, chunks: List[DocumentChunk]) -> None:
        """
        Embed all chunks and build a FAISS IndexFlatIP (inner product = cosine
        similarity when vectors are L2-normalised, which our embedders do).

        Args:
            chunks : list of DocumentChunk objects to index
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "FAISS not installed.\n"
                "CPU only:  pip install faiss-cpu\n"
                "With GPU:  pip install faiss-gpu"
            )

        if not chunks:
            raise ValueError("No chunks provided to build the index.")

        print(f"[VectorStore] Embedding {len(chunks)} chunks...")
        texts = [c.text for c in chunks]
        vectors = self.embedder.embed(texts)   # shape: (N, dim)

        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)    # Inner Product index
        self.index.add(vectors)                # add all vectors
        self.chunks = chunks

        print(f"[VectorStore] Index built: {self.index.ntotal} vectors, dim={dim}")

    def save(
        self,
        index_path: str   = RETRIEVAL.index_path,
        meta_path: str    = RETRIEVAL.metadata_path,
    ) -> None:
        """Persist the FAISS index and chunk metadata to disk."""
        try:
            import faiss
        except ImportError:
            raise ImportError("Run: pip install faiss-cpu")

        faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"[VectorStore] Saved index → {index_path}, metadata → {meta_path}")

    def load(
        self,
        index_path: str   = RETRIEVAL.index_path,
        meta_path: str    = RETRIEVAL.metadata_path,
    ) -> None:
        """Load a previously saved FAISS index and chunk metadata."""
        try:
            import faiss
        except ImportError:
            raise ImportError("Run: pip install faiss-cpu")

        if not os.path.isfile(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.chunks = pickle.load(f)
        print(f"[VectorStore] Loaded {self.index.ntotal} vectors from {index_path}")

    def search(
        self,
        query: str,
        top_k: int            = RETRIEVAL.top_k,
        threshold: float      = RETRIEVAL.similarity_threshold,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Embed the query and return the top-k most similar chunks.

        Args:
            query     : natural language question / query string
            top_k     : max number of results to return
            threshold : minimum similarity score (0..1) to include a result

        Returns:
            List of (DocumentChunk, similarity_score) tuples, sorted by score desc.
        """
        if self.index is None:
            raise RuntimeError("Index not built or loaded. Call build() or load() first.")

        query_vec = self.embedder.embed([query])   # shape: (1, dim)

        scores, indices = self.index.search(query_vec, top_k)
        # scores[0] and indices[0] are the results for our single query

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:                  # FAISS returns -1 for empty slots
                continue
            if float(score) < threshold:
                continue
            results.append((self.chunks[idx], float(score)))

        return results   # already sorted by FAISS (highest score first)


# ---------------------------------------------------------------------------
# Convenience: build from a list of sources and save
# ---------------------------------------------------------------------------

def build_index_from_sources(sources: List[str]) -> VectorStore:
    """
    Full pipeline: load documents → chunk → embed → index → save.

    Args:
        sources : list of file paths, URLs, or "text:..." strings

    Returns:
        A ready-to-query VectorStore
    """
    from document_loader import load_multiple

    chunks = load_multiple(sources)
    store = VectorStore()
    store.build(chunks)
    store.save()
    return store


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from document_loader import load_and_chunk

    sample_texts = [
        "text:Retrieval-Augmented Generation (RAG) is a technique that grounds "
        "LLM outputs in external knowledge by retrieving relevant documents.",

        "text:FAISS (Facebook AI Similarity Search) is a library for efficient "
        "similarity search and clustering of dense vectors.",

        "text:Natural Language Inference (NLI) classifies the relationship between "
        "a premise and a hypothesis as entailment, contradiction, or neutral.",

        "text:Hallucination in LLMs refers to the generation of factually incorrect "
        "or unverifiable content presented with high confidence.",
    ]

    all_chunks = []
    for src in sample_texts:
        all_chunks.extend(load_and_chunk(src))

    store = VectorStore()
    store.build(all_chunks)
    store.save()

    # Test retrieval
    query = "What is hallucination in language models?"
    results = store.search(query, top_k=3)
    print(f"\nQuery: {query}")
    for chunk, score in results:
        print(f"  Score: {score:.4f} | {chunk.text[:100]}")
