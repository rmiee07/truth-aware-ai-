"""
pipeline.py
===========
Orchestrates the full RAG pipeline:

    User Query
       ↓
    Retriever  →  top-k chunks from FAISS
       ↓
    RAGGenerator  →  grounded LLM response
       ↓
    PipelineResult  (answer + retrieved sources)

The NLI-based truth verification and confidence scoring will be added
in the next phase of the project (truth_verifier.py).
"""

from dataclasses import dataclass, field
from typing import List, Optional

from embedder import VectorStore
from retriever import Retriever, RetrievalResult
from generator import RAGGenerator
from config import RETRIEVAL


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """
    Encapsulates everything the pipeline produces for a single query.
    This object is what gets displayed to the user and logged for evaluation.
    """
    query: str                              # original user question
    answer: str                             # LLM-generated answer
    retrieved_chunks: List[RetrievalResult] # evidence used
    context_used: str                       # formatted context string sent to LLM

    # These will be populated by the truth verifier (Phase 2)
    confidence_score: Optional[float] = None
    verdict: Optional[str]            = None   # "verified" | "uncertain" | "refused"
    claim_scores: List[dict]          = field(default_factory=list)

    def display(self) -> str:
        """Human-readable summary of the pipeline result."""
        lines = [
            "=" * 60,
            f"QUERY:  {self.query}",
            "=" * 60,
            f"ANSWER:\n{self.answer}",
            "",
        ]

        if self.confidence_score is not None:
            score_pct = f"{self.confidence_score * 100:.1f}%"
            lines.append(f"TRUTH CONFIDENCE: {score_pct}  [{self.verdict}]")

        lines.append(f"\nSOURCES ({len(self.retrieved_chunks)} chunks):")
        for i, r in enumerate(self.retrieved_chunks, start=1):
            lines.append(f"  [{i}] score={r.score:.3f} | {r.source}")
            lines.append(f"      {r.text[:120]}...")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize to a dictionary (useful for JSON logging / evaluation)."""
        return {
            "query":            self.query,
            "answer":           self.answer,
            "confidence_score": self.confidence_score,
            "verdict":          self.verdict,
            "sources": [
                {"source": r.source, "score": r.score, "text": r.text}
                for r in self.retrieved_chunks
            ],
            "claim_scores":     self.claim_scores,
        }


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline.

    Initialization options:
        A) Pass an already-built VectorStore:
               pipeline = RAGPipeline(store=my_store)

        B) Load a previously saved index from disk:
               pipeline = RAGPipeline()   # auto-loads from RETRIEVAL.index_path

        C) Build from source files:
               pipeline = RAGPipeline.from_sources(["doc1.pdf", "doc2.txt"])
    """

    def __init__(self, store: Optional[VectorStore] = None):
        # Load or use the provided vector store
        if store is not None:
            self.store = store
        else:
            self.store = VectorStore()
            self.store.load()  # load from disk (index_path in config.py)

        self.retriever = Retriever(self.store)
        self.generator = RAGGenerator()

    # ------------------------------------------------------------------
    # Factory: build from source documents
    # ------------------------------------------------------------------

    @classmethod
    def from_sources(cls, sources: List[str]) -> "RAGPipeline":
        """
        Build the full pipeline (load docs → embed → index) from a list of
        file paths, URLs, or "text:..." strings.

        Args:
            sources : list of document sources

        Returns:
            Ready-to-query RAGPipeline instance
        """
        from embedder import build_index_from_sources
        store = build_index_from_sources(sources)
        return cls(store=store)

    # ------------------------------------------------------------------
    # Core query method
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        top_k: int       = RETRIEVAL.top_k,
        threshold: float = RETRIEVAL.similarity_threshold,
        verbose: bool    = False,
    ) -> PipelineResult:
        """
        Run the full RAG pipeline for a single question.

        Steps:
          1. Retrieve relevant chunks from FAISS
          2. Format chunks as context
          3. Generate an LLM response grounded in that context
          4. Return a PipelineResult (truth verification added in Phase 2)

        Args:
            question  : user's natural language question
            top_k     : number of chunks to retrieve
            threshold : minimum similarity score for retrieval
            verbose   : if True, print the full formatted result

        Returns:
            PipelineResult object
        """
        print(f"\n[Pipeline] Running query: '{question[:70]}...'")

        # Step 1 & 2: Retrieve + format context
        retrieved, context = self.retriever.retrieve_and_format(
            query=question, top_k=top_k, threshold=threshold
        )

        # Step 3: Generate
        answer = self.generator.generate(query=question, context=context)

        # Step 4: Package result
        result = PipelineResult(
            query=question,
            answer=answer,
            retrieved_chunks=retrieved,
            context_used=context,
        )

        if verbose:
            print(result.display())

        return result

    def batch_query(
        self,
        questions: List[str],
        verbose: bool = False,
    ) -> List[PipelineResult]:
        """
        Run the pipeline for multiple questions and return all results.
        Useful for bulk evaluation.
        """
        results = []
        for q in questions:
            result = self.query(q, verbose=verbose)
            results.append(result)
        return results


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_pipeline(
    pipeline: RAGPipeline,
    eval_set: List[dict],
) -> dict:
    """
    Run evaluation on a labeled Q&A dataset.

    Args:
        pipeline : initialized RAGPipeline
        eval_set : list of dicts with keys "question" and "expected_answer"

    Returns:
        dict with summary statistics
    """
    total = len(eval_set)
    results = []

    for item in eval_set:
        result = pipeline.query(item["question"])
        results.append({
            "question":        item["question"],
            "expected":        item.get("expected_answer", "N/A"),
            "generated":       result.answer,
            "num_sources":     len(result.retrieved_chunks),
            "confidence":      result.confidence_score,
        })

    # Very basic evaluation: were any sources retrieved?
    retrieved_count = sum(1 for r in results if r["num_sources"] > 0)

    summary = {
        "total_questions":     total,
        "questions_with_sources": retrieved_count,
        "retrieval_rate":      retrieved_count / total if total else 0,
        "results":             results,
    }
    return summary


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Demo using raw text sources (no files needed)
    sources = [
        "text:Retrieval-Augmented Generation (RAG) is an architecture that grounds "
        "LLM responses in external knowledge by first retrieving relevant documents.",

        "text:FAISS (Facebook AI Similarity Search) is an open-source library for "
        "efficient similarity search over large collections of dense vectors.",

        "text:Hallucination in large language models (LLMs) is the phenomenon where "
        "the model generates confident but factually incorrect or unverifiable text.",

        "text:Natural Language Inference (NLI) is the task of determining whether a "
        "hypothesis is entailed by, contradicted by, or neutral with respect to a premise.",

        "text:The DeBERTa model by Microsoft achieves state-of-the-art performance on "
        "the NLI task and is commonly used for fact verification in RAG systems.",
    ]

    print("[Main] Building pipeline from raw text sources...")
    pipeline = RAGPipeline.from_sources(sources)

    questions = [
        "What is hallucination in LLMs?",
        "How does FAISS enable retrieval in RAG?",
        "What is Natural Language Inference used for?",
    ]

    for q in questions:
        result = pipeline.query(q, verbose=True)
