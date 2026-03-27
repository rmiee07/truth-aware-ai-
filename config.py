"""
config.py
=========
Central configuration for the Truth-Aware Generative AI System.
100% free — no API keys required. All models run locally via HuggingFace.
"""

import os
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# LLM Configuration (fully local via HuggingFace Transformers)
# ---------------------------------------------------------------------------
# Options (all free, no API key):
#   "google/flan-t5-base"          → very fast, low RAM, good for Colab free tier
#   "google/flan-t5-large"         → better quality, needs ~4GB VRAM
#   "mistralai/Mistral-7B-Instruct-v0.2" → best quality, needs Colab T4 GPU + ~14GB
#
# Recommendation: start with flan-t5-base, upgrade once pipeline works
LLM_MODEL_NAME = "google/flan-t5-base"


# ---------------------------------------------------------------------------
# Embedding Model (local, free, fast)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# NLI Model for Truth Verification (Phase 2)
# ---------------------------------------------------------------------------
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"


# ---------------------------------------------------------------------------
# Google Drive paths (persistent storage across Colab sessions)
# ---------------------------------------------------------------------------
DRIVE_BASE    = "/content/drive/MyDrive/truth_aware_ai"
INDEX_PATH    = os.path.join(DRIVE_BASE, "faiss_index.bin")
METADATA_PATH = os.path.join(DRIVE_BASE, "faiss_metadata.pkl")


# ---------------------------------------------------------------------------
# Chunking parameters
# ---------------------------------------------------------------------------
@dataclass
class ChunkingConfig:
    chunk_size: int    = 512
    chunk_overlap: int = 64
    min_chunk_len: int = 50


# ---------------------------------------------------------------------------
# Retrieval parameters
# ---------------------------------------------------------------------------
@dataclass
class RetrievalConfig:
    top_k: int                  = 5
    similarity_threshold: float = 0.30
    index_path: str             = INDEX_PATH
    metadata_path: str          = METADATA_PATH


# ---------------------------------------------------------------------------
# Generation parameters
# ---------------------------------------------------------------------------
@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float  = 0.1
    do_sample: bool     = False
    system_prompt: str  = (
        "Answer the question using only the context below. "
        "If the context does not contain the answer, say: "
        "'I do not have enough information to answer this.' "
        "Do not make up facts."
    )


# ---------------------------------------------------------------------------
# Truth scoring thresholds
# ---------------------------------------------------------------------------
@dataclass
class TruthConfig:
    nli_model: str              = NLI_MODEL_NAME
    high_confidence: float      = 0.70
    low_confidence: float       = 0.35
    entailment_weight: float    = 1.0
    neutral_weight: float       = 0.0
    contradiction_weight: float = -1.0


# ---------------------------------------------------------------------------
# Instantiate (import these in other modules)
# ---------------------------------------------------------------------------
CHUNKING   = ChunkingConfig()
RETRIEVAL  = RetrievalConfig()
GENERATION = GenerationConfig()
TRUTH      = TruthConfig()
