"""
main.py
=======
Entry point for the Truth-Aware Generative AI System.

Usage examples:
  # Interactive mode (ask questions after building the index):
  python main.py --sources data/doc1.pdf data/doc2.txt

  # Single query:
  python main.py --sources data/doc1.pdf --query "What is RAG?"

  # Load pre-built index and query:
  python main.py --query "What is hallucination?"

  # Rebuild index then query:
  python main.py --sources data/ --rebuild --query "Explain NLI"
"""

import argparse
import json
import os
import sys
from pathlib import Path

from pipeline import RAGPipeline, PipelineResult
from config import RETRIEVAL


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def gather_sources(paths: list) -> list:
    """
    Expand a list of paths: if a path is a directory, find all
    .pdf, .txt, .docx files inside it (non-recursive).
    """
    SUPPORTED = {".pdf", ".txt", ".docx", ".doc"}
    sources = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if child.suffix.lower() in SUPPORTED:
                    sources.append(str(child))
        elif path.is_file():
            sources.append(str(path))
        elif p.startswith("http://") or p.startswith("https://"):
            sources.append(p)
        else:
            print(f"[Warning] Skipping unrecognized source: {p}")
    return sources


def print_result(result: PipelineResult):
    """Pretty-print a pipeline result to stdout."""
    print(result.display())


def save_result(result: PipelineResult, output_path: str):
    """Save result as JSON for logging / evaluation."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"[Main] Result saved to: {output_path}")


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

def run_interactive(pipeline: RAGPipeline):
    """
    Drop into an interactive question-answering loop.
    Type 'exit' or 'quit' to stop.
    """
    print("\n" + "=" * 60)
    print("  Truth-Aware Generative AI — Interactive Mode")
    print("  Type your question and press Enter.")
    print("  Commands: 'exit' to quit, 'save' to save last result")
    print("=" * 60 + "\n")

    last_result: PipelineResult = None

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[Exiting]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print("[Exiting]")
            break

        if user_input.lower() == "save" and last_result:
            save_result(last_result, "last_result.json")
            continue

        last_result = pipeline.query(user_input, verbose=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Truth-Aware Generative AI System — RAG Pipeline"
    )
    parser.add_argument(
        "--sources", nargs="*", default=[],
        help="Document sources: file paths, directory, or URLs"
    )
    parser.add_argument(
        "--query", "-q", type=str, default=None,
        help="Single question to answer (skips interactive mode)"
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Force rebuilding the index even if one exists on disk"
    )
    parser.add_argument(
        "--top-k", type=int, default=RETRIEVAL.top_k,
        help=f"Number of chunks to retrieve (default: {RETRIEVAL.top_k})"
    )
    parser.add_argument(
        "--threshold", type=float, default=RETRIEVAL.similarity_threshold,
        help=f"Minimum similarity score (default: {RETRIEVAL.similarity_threshold})"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Save the result to a JSON file"
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Build or load the pipeline
    # ------------------------------------------------------------------
    index_exists = (
        os.path.isfile(RETRIEVAL.index_path) and
        os.path.isfile(RETRIEVAL.metadata_path)
    )

    if args.sources:
        sources = gather_sources(args.sources)
        if not sources:
            print("[Error] No valid sources found. Provide .pdf, .txt, or .docx files.")
            sys.exit(1)

        if index_exists and not args.rebuild:
            print(f"[Main] Existing index found. Use --rebuild to overwrite.")
            pipeline = RAGPipeline()
        else:
            print(f"[Main] Building index from {len(sources)} source(s)...")
            pipeline = RAGPipeline.from_sources(sources)

    elif index_exists:
        print(f"[Main] Loading existing index from disk...")
        pipeline = RAGPipeline()
    else:
        # No sources and no saved index: demo mode with built-in examples
        print("[Main] No sources provided and no saved index found.")
        print("[Main] Running demo with built-in sample texts...\n")
        demo_sources = [
            "text:Retrieval-Augmented Generation (RAG) is an AI framework that "
            "retrieves relevant documents from a knowledge base and uses them as "
            "context for a language model to generate grounded, factual responses.",

            "text:Hallucination in LLMs is the generation of confident but "
            "factually incorrect or unverifiable information. It is one of the "
            "primary reliability challenges in production AI systems.",

            "text:FAISS (Facebook AI Similarity Search) is a library that allows "
            "efficient nearest-neighbor search over large collections of dense vectors.",

            "text:Natural Language Inference (NLI) determines whether a hypothesis "
            "is entailed by, contradicted by, or neutral with respect to a given premise.",
        ]
        pipeline = RAGPipeline.from_sources(demo_sources)

    # ------------------------------------------------------------------
    # Single query or interactive mode
    # ------------------------------------------------------------------
    if args.query:
        result = pipeline.query(
            args.query,
            top_k=args.top_k,
            threshold=args.threshold,
            verbose=True,
        )
        if args.output:
            save_result(result, args.output)
    else:
        run_interactive(pipeline)


if __name__ == "__main__":
    main()
