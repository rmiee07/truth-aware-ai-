# Truth-Aware Generative AI System

**Major Project | Manipal University Jaipur**
**Student:** Ramya Mishra (229309090)
**Guide:** Shweta Redkar
**Department:** Data Science and Engineering

---

## Overview

A Retrieval-Augmented Generation (RAG) system with NLI-based truth verification and confidence scoring. Tackles the hallucination problem in LLMs by grounding every response in verified knowledge sources.

**100% free — no API keys required.** All models run locally via HuggingFace.

---

## Tech Stack

| Component | Tool |
|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Search | FAISS-CPU |
| LLM Generation | `google/flan-t5-base` |
| NLI Verification | `cross-encoder/nli-deberta-v3-small` |
| Notebook | Google Colab (T4 GPU) |
| Storage | Google Drive |

---

## Project Structure

```
truth_aware_ai/
├── config.py              # Central config (models, thresholds, paths)
├── document_loader.py     # Load PDF/TXT/DOCX/URL → chunks
├── embedder.py            # Embed chunks → FAISS index
├── retriever.py           # Query → top-k relevant chunks
├── generator.py           # RAG-grounded LLM response
├── pipeline.py            # Orchestrates all stages
├── main.py                # CLI entry point
├── TruthAwareAI_Colab.ipynb  # ← Start here
└── requirements.txt
```

---

## Quick Start (Google Colab)

1. Open `TruthAwareAI_Colab.ipynb` in Google Colab
2. Set runtime to **T4 GPU**
3. Run cells top to bottom
4. No API keys needed

---

## Pipeline Stages

```
User Query
   ↓
Embed query (MiniLM)
   ↓
FAISS similarity search → top-k chunks
   ↓
Inject chunks into prompt → flan-t5-base
   ↓
[Phase 2] NLI fact verification (DeBERTa)
   ↓
[Phase 2] Confidence score (0.0 → 1.0)
   ↓
Response + verdict + sources
```

---

## Model Upgrade Path

| Model | Quality | Colab RAM needed |
|---|---|---|
| `flan-t5-base` | Good | ~2 GB (CPU works) |
| `flan-t5-large` | Better | ~4 GB GPU |
| `Mistral-7B-Instruct` | Best | ~14 GB GPU |

Change `LLM_MODEL_NAME` in `config.py` to upgrade.
