# Step-by-Step Implementation Guide
## "Towards Reliable Retrieval in RAG Systems for Large Legal Datasets"

This guide provides complete instructions for implementing and reproducing the experiments from the research paper on Summary-Augmented Chunking (SAC) for RAG systems.

---

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Step 1: Understanding the Problem](#step-1-understanding-the-problem)
4. [Step 2: Baseline RAG Implementation](#step-2-baseline-rag-implementation)
5. [Step 3: Summary-Augmented Chunking (SAC)](#step-3-summary-augmented-chunking-sac)
6. [Step 4: Expert-Guided SAC](#step-4-expert-guided-sac)
7. [Step 5: Evaluation Metrics](#step-5-evaluation-metrics)
8. [Step 6: Running Experiments](#step-6-running-experiments)
9. [Step 7: Analyzing Results](#step-7-analyzing-results)
10. [Advanced Topics](#advanced-topics)

---

## Overview

### What is the Paper About?

The paper addresses a critical problem in Retrieval-Augmented Generation (RAG) systems for legal documents: **retrievers often select text from entirely wrong documents** due to high structural similarity in legal corpora.

### Key Innovation: Summary-Augmented Chunking (SAC)

Instead of embedding chunks in isolation, **prepend a document-level summary to each chunk**. This injects global context that helps the retriever identify the correct source document.

### Main Metrics

1. **Document-Level Retrieval Mismatch (DRM)**: % of retrieved chunks from wrong documents
2. **Character-Level Precision**: Fraction of retrieved text that's relevant
3. **Character-Level Recall**: Fraction of relevant text that was retrieved

---

## Prerequisites

### Required Libraries

```bash
pip install sentence-transformers faiss-cpu numpy --break-system-packages
# For actual implementation, you'd also need:
pip install openai anthropic  # For generating summaries
```

### System Requirements
- Python 3.8+
- 8GB+ RAM recommended
- GPU optional but helpful for faster embeddings

---

## Step 1: Understanding the Problem

### The Document-Level Retrieval Mismatch (DRM) Problem

**Scenario**: You have 100 NDAs (Non-Disclosure Agreements) in your database. They're 95% identical - same structure, similar clauses, different only in party names and dates.

**What happens**:
1. User asks: "Can Evelozcity independently develop similar information?"
2. Standard RAG chunks all documents into 500-character pieces
3. Retriever finds chunks that mention "independently develop" 
4. **Problem**: It retrieves from the WRONG NDA because chunks look identical!

**Why this matters in legal contexts**:
- Wrong document = wrong parties = invalid legal answer
- Could give advice about Company A's contract when asked about Company B
- Undermines trust and reliability

### Paper's Key Insight

The problem is **loss of global context**. Each chunk is embedded alone, without knowing:
- Which document it came from
- What the document is about
- Who the parties are
- What makes THIS document unique

---

## Step 2: Baseline RAG Implementation

### 2.1 Document Representation

```python
@dataclass
class Document:
    doc_id: str          # Unique identifier
    filename: str        # Original filename
    content: str         # Full text content
    doc_type: Optional[str] = None  # NDA, Privacy Policy, etc.
```

### 2.2 Recursive Character Splitting

The paper uses **recursive character splitting** - it tries to split on natural boundaries:

```
Priority of separators:
1. "\n\n" (paragraph breaks)
2. "\n"   (line breaks)
3. ". "   (sentence ends)
4. " "    (word breaks)
5. ""     (character breaks - last resort)
```

**Algorithm**:
```python
def recursive_split(text, separators, chunk_size=500):
    if no separators left:
        return split_by_fixed_length(text, chunk_size)
    
    separator = separators[0]
    splits = text.split(separator)
    
    chunks = []
    current_chunk = ""
    
    for split in splits:
        if len(current_chunk + split) <= chunk_size:
            current_chunk += separator + split
        else:
            chunks.append(current_chunk)
            if len(split) > chunk_size:
                # Recursively split this piece
                sub_chunks = recursive_split(split, separators[1:], chunk_size)
                chunks.extend(sub_chunks)
            else:
                current_chunk = split
    
    return chunks
```

### 2.3 Creating Embeddings

```python
# For each chunk:
1. Convert text to embedding vector using SentenceTransformer
2. Normalize for cosine similarity
3. Add to FAISS index

# Retrieval:
1. Embed the query
2. Find top-k most similar chunk embeddings
3. Return corresponding chunks
```

### 2.4 Baseline Pipeline

```
Input Document
    ↓
[Chunk into 500-char pieces]
    ↓
[Embed each chunk independently]
    ↓
[Index in FAISS vector DB]
    ↓
Query → [Retrieve top-k] → Results
```

**Problem**: No document-level context! Chunks from different documents look identical.

---

## Step 3: Summary-Augmented Chunking (SAC)

### 3.1 Core Idea

For EACH document:
1. Generate ONE summary (~150 characters)
2. Prepend this summary to EVERY chunk from that document
3. Embed the augmented chunk (summary + chunk content)

### 3.2 Generic Summarization Prompt

```
System: You are an expert legal document summarizer.

User: Summarize the following legal document text.
Focus on extracting the most important entities, core purpose, 
and key legal topics. The summary must be concise, maximum 
150 characters long, and optimized for providing context to 
smaller text chunks. Output only the summary text.

Document: {document_content}
```

### 3.3 SAC Pipeline

```
Input Document
    ↓
[Generate Document Summary]  ← NEW STEP
    ↓
[Chunk into 500-char pieces]
    ↓
[Prepend summary to each chunk]  ← NEW STEP
    ↓
Chunk format: "{summary}\n\n{chunk_content}"
    ↓
[Embed augmented chunks]
    ↓
[Index in FAISS vector DB]
    ↓
Query → [Retrieve top-k] → Results
```

### 3.4 Example

**Original Chunk** (no context):
```
"(d) is independently developed by or for the Recipient 
by persons who have had no access to or been informed 
of the existence or substance of the Confidential Information."
```

**With SAC** (global context added):
```
"Non-Disclosure Agreement between Evelozcity and Recipient 
to protect confidential information shared during a meeting.

(d) is independently developed by or for the Recipient 
by persons who have had no access to or been informed 
of the existence or substance of the Confidential Information."
```

Now the retriever knows:
- This is an NDA
- It's specifically Evelozcity's NDA
- It's about confidential information from a meeting

---

## Step 4: Expert-Guided SAC

### 4.1 Motivation

Generic summaries help, but can we do better by incorporating **legal domain knowledge**?

Legal experts know that certain elements REALLY matter for distinguishing documents:
- For NDAs: parties, definition of confidential info, exclusions, term
- For Privacy Policies: data types collected, legal basis, purposes, retention

### 4.2 Expert Prompt Structure

```
1. Classify document type (NDA, Privacy Policy, Other)
2. Apply type-specific template
3. Extract legally-relevant differentiating features
```

### 4.3 NDA Template

```
Extract and summarize:
- **Parties**: Who is disclosing? Who is receiving?
- **Confidential Information Definition**: What types of data?
- **Obligations**: What must recipient do?
- **Exclusions**: What's NOT covered?
- **Term**: How long does it last?
- **Jurisdiction**: Which laws apply?
```

### 4.4 Privacy Policy Template

```
Extract and summarize:
- **Data Collected**: What personal data is gathered?
- **Controller**: Who is responsible?
- **Purposes**: Why is data processed?
- **Legal Basis**: Consent? Contract? Legitimate interest?
- **Recipients**: Who gets the data?
- **Retention**: How long is data kept?
- **User Rights**: What rights do users have?
```

### 4.5 Example Expert Summary

**Generic SAC Summary**:
```
"Non-Disclosure Agreement between Evelozcity and Recipient 
to protect confidential information shared during a meeting."
```

**Expert-Guided SAC Summary**:
```
"NDA between Evelozcity and Recipient; covers vehicle prototypes, 
confidentiality obligations, exclusions, 5-yr term, CA governing law."
```

More specific → Should better distinguish between similar NDAs

---

## Step 5: Evaluation Metrics

### 5.1 Document-Level Retrieval Mismatch (DRM)

**Definition**: Proportion of top-k retrieved chunks from WRONG documents

**Formula**:
```
DRM = (# chunks from wrong doc) / (total # chunks retrieved)
```

**Example**:
```
Query about "Evelozcity NDA"
Ground truth document: nda_evelozcity

Retrieved top-5 chunks:
1. From nda_evelozcity ✓
2. From nda_roi_corp   ✗ (mismatch!)
3. From nda_evelozcity ✓
4. From nda_roi_corp   ✗ (mismatch!)
5. From nda_evelozcity ✓

DRM = 2/5 = 40%
```

**Lower is better!** DRM = 0% means perfect document selection.

### 5.2 Character-Level Precision

**Definition**: What fraction of RETRIEVED text actually overlaps with ground truth?

**Formula**:
```
Precision = |retrieved_chars ∩ ground_truth_chars| / |retrieved_chars|
```

**Example**:
```
Ground truth span: characters 1000-1500 (500 chars)
Retrieved chunks cover: characters 950-1200, 1400-1600 (450 chars)

Overlap: 1000-1200, 1400-1500 = 200 + 100 = 300 chars
Precision = 300 / 450 = 0.667 (66.7%)
```

### 5.3 Character-Level Recall

**Definition**: What fraction of GROUND TRUTH text was retrieved?

**Formula**:
```
Recall = |retrieved_chars ∩ ground_truth_chars| / |ground_truth_chars|
```

**Same example**:
```
Overlap: 300 chars
Ground truth: 500 chars
Recall = 300 / 500 = 0.60 (60%)
```

### 5.4 Precision vs Recall Trade-off

- **High Precision, Low Recall**: Retrieved text is relevant but incomplete
- **Low Precision, High Recall**: Retrieved lots of text but much is irrelevant
- **Ideal**: Both high - retrieved all relevant text and nothing else

### 5.5 Why Character-Level Metrics?

Legal text is precise. Even if you get the right document, you need the RIGHT CLAUSE.

Word-level or sentence-level metrics might miss that you retrieved the wrong subsection.

---

## Step 6: Running Experiments

### 6.1 Experimental Design

The paper compares three configurations:

| Configuration | Use SAC? | Summary Type |
|--------------|----------|--------------|
| Baseline     | No       | N/A          |
| SAC-Generic  | Yes      | Generic      |
| SAC-Expert   | Yes      | Expert       |

### 6.2 Variables to Test

**Hyperparameters**:
- Chunk size: 200, 500, 800 characters
- Summary length: 150, 300 characters
- Top-k: 1, 2, 4, 8, 16, 32, 64

**Models**:
- Embedding model: thenlper/gte-large (paper's choice)
- Summarization model: gpt-4o-mini

### 6.3 Running the Experiment

```python
# 1. Load documents
documents = load_legalbench_rag_dataset()

# 2. Load queries with ground truth
queries = load_queries()

# 3. For each configuration:
for config in [baseline, sac_generic, sac_expert]:
    # Initialize retriever
    retriever = RAGRetriever(**config)
    
    # Index documents
    retriever.index_documents(documents)
    
    # Evaluate on all queries
    for k in [1, 2, 4, 8, 16, 32, 64]:
        for query in queries:
            results = retriever.retrieve(query.text, top_k=k)
            
            # Calculate metrics
            drm = calculate_drm(results, query.ground_truth_doc)
            prec, rec = calculate_precision_recall(results, query.ground_truth_span)
            
            # Store results
            log_metrics(config, k, drm, prec, rec)

# 4. Aggregate and plot
plot_results()
```

### 6.4 What to Expect

**Paper's findings**:
1. Baseline DRM: 50-95% (terrible!)
2. SAC DRM: 20-40% (much better!)
3. Generic SAC > Expert SAC (surprising!)

**Why expert summaries didn't win**:
- Too dense/structured for embedding models
- May overfit to narrow features
- Generic summaries strike better balance

---

## Step 7: Analyzing Results

### 7.1 Plotting DRM Curves

Create plots showing DRM vs top-k for each method:

```python
import matplotlib.pyplot as plt

top_k_values = [1, 2, 4, 8, 16, 32, 64]

plt.figure(figsize=(10, 6))
plt.plot(top_k_values, baseline_drm, label='Baseline', marker='o')
plt.plot(top_k_values, sac_generic_drm, label='SAC (Generic)', marker='s')
plt.plot(top_k_values, sac_expert_drm, label='SAC (Expert)', marker='^')

plt.xlabel('Top-K')
plt.ylabel('DRM (%)')
plt.title('Document-Level Retrieval Mismatch')
plt.legend()
plt.grid(True)
plt.show()
```

### 7.2 Interpreting Results

**Good signs**:
- DRM decreases with SAC
- Precision/recall increase with SAC
- Improvement consistent across different k values

**What if results are bad?**:
- Check summary quality (are they informative?)
- Try different embedding models
- Verify ground truth annotations
- Check for data leakage

### 7.3 Statistical Significance

Run multiple random seeds and compute confidence intervals:

```python
results = []
for seed in range(5):
    set_random_seed(seed)
    result = run_experiment()
    results.append(result)

mean_drm = np.mean(results)
std_drm = np.std(results)
ci_95 = 1.96 * std_drm / np.sqrt(len(results))

print(f"DRM: {mean_drm:.2f}% ± {ci_95:.2f}%")
```

---

## Advanced Topics

### A. Using Real LLM APIs for Summaries

```python
import openai

def generate_summary_with_gpt4(document):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert legal document summarizer."},
            {"role": "user", "content": f"Summarize this legal document in 150 characters:\n\n{document.content}"}
        ],
        max_tokens=100,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()
```

### B. Integrating with LegalBench-RAG Dataset

```python
import datasets

# Load from HuggingFace
dataset = datasets.load_dataset("legalbench/legalbench_rag")

# Convert to our format
documents = []
queries = []

for split in ['contractnli', 'cuad', 'maud', 'privacy_qa']:
    data = dataset[split]
    
    for example in data:
        # Extract document
        doc = Document(
            doc_id=example['doc_id'],
            filename=example['filename'],
            content=example['content']
        )
        documents.append(doc)
        
        # Extract query
        query = Query(
            query_id=example['query_id'],
            query_text=example['query'],
            ground_truth_doc_id=example['doc_id'],
            ground_truth_start=example['span_start'],
            ground_truth_end=example['span_end']
        )
        queries.append(query)
```

### C. Hybrid Dense + Sparse Retrieval

The paper tested BM25 (sparse) + semantic search (dense):

```python
from rank_bm25 import BM25Okapi

class HybridRetriever:
    def __init__(self, alpha=0.5):
        self.dense_retriever = DenseRetriever()
        self.bm25 = None
        self.alpha = alpha  # Weight for dense vs sparse
    
    def index(self, documents):
        # Index for dense retrieval
        self.dense_retriever.index(documents)
        
        # Index for BM25
        tokenized_docs = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def retrieve(self, query, top_k=10):
        # Get dense scores
        dense_results = self.dense_retriever.retrieve(query, top_k=100)
        
        # Get sparse scores
        sparse_scores = self.bm25.get_scores(query.split())
        
        # Combine scores
        combined_scores = {}
        for i, result in enumerate(dense_results):
            dense_score = result.score
            sparse_score = sparse_scores[i]
            combined_scores[i] = self.alpha * dense_score + (1 - self.alpha) * sparse_score
        
        # Re-rank and return top-k
        sorted_indices = sorted(combined_scores, key=combined_scores.get, reverse=True)
        return [dense_results[i] for i in sorted_indices[:top_k]]
```

**Paper's finding**: Hybrid improved DRM slightly but reduced precision/recall. Not worth the complexity.

### D. Analyzing Embedding Space

Visualize how SAC affects the embedding space:

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Get embeddings for chunks from 3 different documents
baseline_embeddings = get_baseline_chunk_embeddings()
sac_embeddings = get_sac_chunk_embeddings()

# Reduce to 2D
tsne = TSNE(n_components=2, random_state=42)
baseline_2d = tsne.fit_transform(baseline_embeddings)
sac_2d = tsne.fit_transform(sac_embeddings)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Baseline: chunks from different docs overlap
ax1.scatter(baseline_2d[:, 0], baseline_2d[:, 1], c=doc_labels, cmap='tab10')
ax1.set_title('Baseline: Chunks Overlap (High DRM)')

# SAC: chunks cluster by document
ax2.scatter(sac_2d[:, 0], sac_2d[:, 1], c=doc_labels, cmap='tab10')
ax2.set_title('SAC: Chunks Cluster by Document (Low DRM)')

plt.show()
```

### E. Error Analysis

Qualitative analysis of failure cases:

```python
def analyze_errors(queries, results):
    """Analyze queries where DRM > 50%"""
    
    errors = []
    for query, result in zip(queries, results):
        if result['drm'] > 0.5:
            errors.append({
                'query': query.text,
                'expected_doc': query.ground_truth_doc_id,
                'retrieved_docs': [r.chunk.doc_id for r in result['retrieved']],
                'summary': documents[query.ground_truth_doc_id].summary
            })
    
    # Categorize errors
    categories = {
        'insufficient_summary': [],  # Summary didn't capture key differentiators
        'structural_similarity': [],  # Documents too similar
        'query_ambiguity': []  # Query could match multiple docs
    }
    
    for error in errors:
        # Manual categorization or automated heuristics
        if len(set(error['retrieved_docs'])) == 1:
            categories['structural_similarity'].append(error)
        # ... etc
    
    return categories
```

---

## Common Pitfalls and Solutions

### Pitfall 1: Summary Too Generic
**Problem**: Summary says "This is a legal contract" for everything
**Solution**: Prompt LLM to focus on DIFFERENTIATING features

### Pitfall 2: Summary Too Long
**Problem**: Summary dominates the chunk, local content gets ignored
**Solution**: Keep summaries concise (150 chars max per paper)

### Pitfall 3: Wrong Embedding Model
**Problem**: Model not trained on legal text
**Solution**: Use thenlper/gte-large or legal-specific models

### Pitfall 4: Not Normalizing Embeddings
**Problem**: FAISS returns wrong results
**Solution**: Always normalize embeddings for cosine similarity

### Pitfall 5: Ignoring Document Boundaries
**Problem**: Chunks span multiple documents
**Solution**: Chunk each document separately

---

## Reproducing Paper Results

### Full Reproduction Checklist

- [ ] Use LegalBench-RAG dataset (exact same data)
- [ ] Use thenlper/gte-large embeddings
- [ ] Use gpt-4o-mini for summaries
- [ ] Chunk size: 500 characters
- [ ] Summary length: 150 characters
- [ ] Recursive character splitting
- [ ] FAISS with cosine similarity
- [ ] Evaluate on all 4 subsets: ContractNLI, CUAD, MAUD, PrivacyQA
- [ ] Run multiple random seeds
- [ ] Report mean ± std for all metrics

### Expected Results (from paper)

**ContractNLI (362 NDAs - hardest dataset)**:
- Baseline DRM: ~95%
- SAC DRM: ~40%
- Precision improvement: +15-20%
- Recall improvement: +20-30%

**Overall (weighted average)**:
- Baseline DRM: ~50-60%
- SAC DRM: ~20-30%
- Expert SAC: Similar to generic SAC

---

## Conclusion

This implementation demonstrates:

1. **The Problem**: Standard RAG fails on similar legal documents
2. **The Solution**: Summary-Augmented Chunking injects global context
3. **The Result**: DRM cuts in half, precision/recall improve significantly
4. **The Surprise**: Simple generic summaries work better than expert-crafted ones

### Key Takeaways

- **Context matters**: Don't embed chunks in isolation
- **Simple works**: One summary per document is enough
- **Generic beats specific**: For retrieval, broad semantics > dense legal detail
- **Practical**: SAC is cheap, modular, and easy to integrate

### Next Steps

1. Test on your own legal documents
2. Experiment with different summary lengths
3. Try hierarchical SAC (paragraph, section, document summaries)
4. Combine with query optimization and reranking
5. Measure end-to-end RAG performance (retrieval + generation)

---

## References

Pipitone & Alami (2024). LegalBench-RAG: A benchmark for retrieval-augmented generation in the legal domain.

Paper: "Towards Reliable Retrieval in RAG Systems for Large Legal Datasets" (NLLP 2025)
