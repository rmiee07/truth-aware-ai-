"""
retriever.py
============
The retriever takes a natural language query, embeds it, searches the FAISS
index, and returns the top-k relevant chunks formatted as a context string
ready to be injected into the LLM prompt.

This is the "R" in RAG — the component that prevents hallucination
by giving the LLM verified facts to reason from.
"""

from typing import List, Tuple
from embedder import VectorStore
from document_loader import DocumentChunk
from config import RETRIEVAL


# ---------------------------------------------------------------------------
# RetrievalResult: bundles a chunk with its similarity score
# ---------------------------------------------------------------------------

class RetrievalResult:
    """Lightweight container for a retrieved chunk and its relevance score."""

    def __init__(self, chunk: DocumentChunk, score: float):
        self.chunk = chunk
        self.score = score
        self.text  = chunk.text
        self.source = chunk.source

    def __repr__(self):
        preview = self.text[:80].replace("\n", " ")
        return f"<RetrievalResult score={self.score:.4f} | '{preview}...'>"


# ---------------------------------------------------------------------------
# Retriever class
# ---------------------------------------------------------------------------

class Retriever:
    """
    Wraps the VectorStore to provide a clean .retrieve(query) interface.

    After retrieval, it also:
      - Deduplicates near-identical chunks (exact text match)
      - Formats chunks into a single context string for the LLM prompt
    """

    def __init__(self, store: VectorStore):
        self.store = store

    def retrieve(
        self,
        query: str,
        top_k: int       = RETRIEVAL.top_k,
        threshold: float = RETRIEVAL.similarity_threshold,
    ) -> List[RetrievalResult]:
        """
        Retrieve the most relevant chunks for a given query.

        Args:
            query     : the user's question
            top_k     : number of chunks to fetch from FAISS
            threshold : minimum similarity score to keep a result

        Returns:
            Sorted list of RetrievalResult objects (best match first)
        """
        raw_results = self.store.search(query, top_k=top_k, threshold=threshold)

        results = [RetrievalResult(chunk, score) for chunk, score in raw_results]
        results = self._deduplicate(results)

        print(f"[Retriever] Query: '{query[:60]}...' → {len(results)} chunks retrieved")
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate(results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Remove duplicate chunks (same text appearing from multiple sources)."""
        seen_texts = set()
        unique = []
        for r in results:
            # Normalize: lowercase + strip whitespace for comparison
            key = r.text.lower().strip()
            if key not in seen_texts:
                seen_texts.add(key)
                unique.append(r)
        return unique

    @staticmethod
    def format_context(results: List[RetrievalResult]) -> str:
        """
        Format retrieved chunks into a numbered context block for injection
        into the LLM system/user prompt.

        Example output:
            [Source 1 | score: 0.87 | file.pdf]
            RAG combines retrieval with generation...

            [Source 2 | score: 0.74 | wiki.txt]
            FAISS is a library for...
        """
        if not results:
            return "No relevant context was found in the knowledge base."

        parts = []
        for i, r in enumerate(results, start=1):
            header = f"[Source {i} | score: {r.score:.2f} | {r.source}]"
            parts.append(f"{header}\n{r.text}")

        return "\n\n".join(parts)

    def retrieve_and_format(
        self,
        query: str,
        top_k: int       = RETRIEVAL.top_k,
        threshold: float = RETRIEVAL.similarity_threshold,
    ) -> Tuple[List[RetrievalResult], str]:
        """
        Convenience method: retrieve + format in one call.

        Returns:
            (results, context_string)
            results        → list of RetrievalResult (for scoring, provenance)
            context_string → formatted string to inject into the LLM prompt
        """
        results = self.retrieve(query, top_k=top_k, threshold=threshold)
        context = self.format_context(results)
        return results, context


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Assumes you have already run embedder.py to build and save the index.
    store = VectorStore()
    store.load()

    retriever = Retriever(store)

    test_queries = [
        "What is hallucination in AI?",
        "How does FAISS work?",
        "Explain retrieval augmented generation",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        results, context = retriever.retrieve_and_format(q)
        print(f"Query: {q}")
        print(f"Context:\n{context[:400]}...")
