"""
generator.py
============
LLM-backed RAG response generation using only free HuggingFace models.
No API keys required — all inference runs locally in Colab.

Default model: google/flan-t5-base  (fast, free, works on Colab CPU/GPU)
Upgrade path:  google/flan-t5-large → flan-t5-xl → Mistral-7B-Instruct
"""

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from config import LLM_MODEL_NAME, GENERATION


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_rag_prompt(query: str, context: str) -> str:
    """
    Flan-T5 works best with explicit instruction-style prompts.
    Keep it concise — flan-t5-base has a 512-token input limit.
    """
    # Truncate context if too long (flan-t5-base limit is 512 tokens ~ 1800 chars)
    max_context_chars = 1600
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "..."

    prompt = (
        f"{GENERATION.system_prompt}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
    return prompt


# ---------------------------------------------------------------------------
# HuggingFace local generator
# ---------------------------------------------------------------------------

class LocalHFGenerator:
    """
    Loads a seq2seq (T5-family) or causal LM from HuggingFace and runs
    inference locally. Works on both CPU and GPU — Colab T4 GPU is ideal.

    Supported model families:
      - google/flan-t5-*       (seq2seq, recommended for free Colab)
      - mistralai/Mistral-7B-* (causal, needs T4 GPU + ~14 GB)
      - TinyLlama/*            (causal, lightweight alternative)
    """

    def __init__(self, model_name: str = LLM_MODEL_NAME):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Generator] Device: {self.device}")
        print(f"[Generator] Loading model: {model_name}  (this may take 1-2 min first time)")

        # Detect model family to pick the right pipeline task
        is_causal = any(x in model_name.lower() for x in ["mistral", "llama", "gpt", "bloom", "tinyllama"])
        task = "text-generation" if is_causal else "text2text-generation"

        self.pipe = pipeline(
            task=task,
            model=model_name,
            device=0 if self.device == "cuda" else -1,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.is_causal = is_causal
        print(f"[Generator] Model loaded successfully.")

    def generate(self, query: str, context: str) -> str:
        """
        Generate an answer grounded in the retrieved context.

        Args:
            query   : user's question
            context : formatted retrieved chunks from Retriever

        Returns:
            Answer string
        """
        if not context.strip():
            return "I could not find relevant information in the knowledge base."

        prompt = build_rag_prompt(query, context)

        output = self.pipe(
            prompt,
            max_new_tokens=GENERATION.max_new_tokens,
            temperature=GENERATION.temperature if GENERATION.do_sample else 1.0,
            do_sample=GENERATION.do_sample,
            # For causal models, strip the input prompt from output
            return_full_text=False if self.is_causal else None,
        )

        # Extract text depending on pipeline output format
        if self.is_causal:
            answer = output[0]["generated_text"].strip()
        else:
            answer = output[0]["generated_text"].strip()

        return answer if answer else "I was unable to generate a response."


# ---------------------------------------------------------------------------
# RAGGenerator: thin wrapper for clean pipeline interface
# ---------------------------------------------------------------------------

class RAGGenerator:
    """
    High-level generator used by pipeline.py.

    Usage:
        gen = RAGGenerator()
        answer = gen.generate(query="What is RAG?", context="RAG stands for...")
    """

    def __init__(self):
        self._backend = LocalHFGenerator()

    def generate(self, query: str, context: str) -> str:
        print(f"[Generator] Generating for: '{query[:60]}'")
        answer = self._backend.generate(query, context)
        print(f"[Generator] Done. ({len(answer)} chars)")
        return answer


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_context = (
        "[Source 1 | score: 0.91]\n"
        "Retrieval-Augmented Generation (RAG) is an AI framework that combines "
        "a retrieval system with a generative model. It retrieves relevant passages "
        "from a knowledge base and uses them as context to generate grounded answers, "
        "significantly reducing hallucinations.\n\n"
        "[Source 2 | score: 0.76]\n"
        "Hallucination in LLMs refers to the generation of fluent but factually "
        "incorrect or unverifiable content presented with high confidence."
    )

    gen = RAGGenerator()
    answer = gen.generate(
        query="How does RAG reduce hallucinations?",
        context=test_context,
    )
    print(f"\nAnswer:\n{answer}")
