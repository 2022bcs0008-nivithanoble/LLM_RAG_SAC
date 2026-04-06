# Summary-Augmented Chunking (SAC) for RAG Systems - Implementation

Complete implementation of the paper: **"Towards Reliable Retrieval in RAG Systems for Large Legal Datasets"** (NLLP 2025)

## Overview

This repository contains a full implementation of Summary-Augmented Chunking (SAC), a technique to improve retrieval quality in RAG systems for legal documents by reducing Document-Level Retrieval Mismatch (DRM).

### The Problem

Standard RAG systems struggle with large databases of structurally similar legal documents (e.g., NDAs, privacy policies). Retrievers often select text from entirely wrong documents because:
- Documents share boilerplate language
- Chunks are embedded without document context
- Similarity search can't distinguish between documents

This is **Document-Level Retrieval Mismatch (DRM)**: retrieving chunks from the wrong source document.

### The Solution: Summary-Augmented Chunking (SAC)

1. Generate ONE summary per document (~150 characters)
2. Prepend this summary to EVERY chunk from that document
3. Embed the augmented chunks (summary + content)
4. Retrieve as usual

**Result**: DRM cuts in half, precision and recall improve significantly!

## 🚀 Quick Start

### Installation

```bash
# Install required packages
pip install sentence-transformers faiss-cpu numpy matplotlib --break-system-packages

# For generating summaries (optional - uses mock summaries by default)
pip install openai anthropic --break-system-packages
```

### Run the Interactive Tutorial

```bash
python interactive_tutorial.py
```

This walks you through:
1. Understanding the DRM problem
2. Implementing baseline RAG
3. Implementing SAC
4. Comparing results
5. Generating visualizations

### Run the Full Experiment

```bash
python rag_sac_implementation.py
```

This runs all three configurations and generates comparison results.

## File Structure

```
├── rag_sac_implementation.py    # Core implementation
│   ├── Document, Chunk, Query classes
│   ├── RecursiveCharacterSplitter
│   ├── SummaryGenerator (generic & expert)
│   ├── RAGRetriever (with optional SAC)
│   └── RAGEvaluator (DRM, precision, recall)
│
├── visualization_utils.py       # Plotting and analysis
│   ├── ResultsVisualizer
│   ├── ErrorAnalyzer
│   └── Plotting functions
│
├── interactive_tutorial.py      # Step-by-step walkthrough
│
├── STEP_BY_STEP_GUIDE.md       # Detailed implementation guide
│
└── README.md                    # This file
```

## Experiments

### 1. Baseline RAG

```python
from rag_sac_implementation import RAGRetriever

retriever = RAGRetriever(
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    use_sac=False,  # No summary augmentation
    chunk_size=500
)

retriever.index_documents(documents)
results = retriever.retrieve("your query here", top_k=5)
```

### 2. SAC with Generic Summaries

```python
retriever = RAGRetriever(
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    use_sac=True,  # Enable SAC
    use_expert_summary=False,  # Use generic summaries
    summary_length=150,
    chunk_size=500
)

retriever.index_documents(documents)
results = retriever.retrieve("your query here", top_k=5)
```

## Evaluation Metrics

### 1. Document-Level Retrieval Mismatch (DRM)

Percentage of retrieved chunks from wrong documents:

```python
from rag_sac_implementation import RAGEvaluator

evaluator = RAGEvaluator()
results = evaluator.evaluate_dataset(
    queries,
    retriever,
    top_k_values=[1, 2, 4, 8, 16, 32, 64]
)

print(f"DRM at k=8: {results[8]['drm']:.2f}%")
```

**Lower is better!** DRM = 0% means all chunks from correct document.

### 2. Character-Level Precision

Fraction of retrieved text that overlaps with ground truth:

```python
print(f"Precision at k=8: {results[8]['precision']:.4f}")
```

### 3. Character-Level Recall

Fraction of ground truth text that was retrieved:

```python
print(f"Recall at k=8: {results[8]['recall']:.4f}")
```


## Key Results (from paper)

### On LegalBench-RAG Dataset

| Metric | Baseline | SAC-Generic | Improvement |
|--------|----------|-------------|-------------|
| DRM (k=8) | ~60% | ~30% | **-50%** |
| Precision | 0.11 | 0.25 | **+127%** |
| Recall | 0.35 | 0.55 | **+57%** |

### Most Challenging Dataset: ContractNLI (362 NDAs)

- Baseline DRM: **95%** (nearly all wrong!)
- SAC DRM: **40%** (still challenging but much better)


## Implementation Details

### Recursive Character Splitting

The paper uses recursive splitting with priority separators:

1. `\n\n` (paragraph breaks) - preferred
2. `\n` (line breaks)
3. `. ` (sentence ends)
4. ` ` (word breaks)
5. `` (character - last resort)

This preserves semantic coherence better than fixed-length splitting.

### FAISS Indexing

Uses Inner Product (cosine similarity) with normalized embeddings:

```python
import faiss

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product
faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
index.add(embeddings)
```


## 📖 Citation

```bibtex
@inproceedings{reuter2025reliable,
  title={Towards Reliable Retrieval in RAG Systems for Large Legal Datasets},
  author={Reuter, Markus and Lingenberg, Tobias and others},
  booktitle={Proceedings of the Natural Legal Language Processing Workshop 2025},
  pages={17--30},
  year={2025}
}
```


Based on the paper by Reuter, Lingenberg, et al. (NLLP 2025). Implementation follows the methodology described in the paper.

---

