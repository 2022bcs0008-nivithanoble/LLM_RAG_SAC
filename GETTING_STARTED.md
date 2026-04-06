# 🚀 GETTING STARTED - Quick Guide

## What You Have

A complete implementation of **Summary-Augmented Chunking (SAC) for RAG Systems** from the paper "Towards Reliable Retrieval in RAG Systems for Large Legal Datasets" (NLLP 2025).

## 3-Minute Quick Start

### Step 1: Install Dependencies

```bash
pip install sentence-transformers faiss-cpu numpy matplotlib --break-system-packages
```

### Step 2: Choose Your Path

#### Option A: Interactive Tutorial (Recommended for Learning)
```bash
python interactive_tutorial.py
```
This walks you through every step with explanations!

#### Option B: Run All Experiments
```bash
python run_experiments.py all
```
This runs all experiments and generates visualization plots.

#### Option C: Run Individual Experiments
```bash
python run_experiments.py baseline   # Baseline RAG
python run_experiments.py sac        # SAC with generic summaries
python run_experiments.py expert     # SAC with expert summaries
```

### Step 3: View Results

After running experiments, you'll get:
- `drm_comparison.png` - Document-Level Retrieval Mismatch curves
- `precision_recall.png` - Precision and Recall curves
- `improvements.png` - Bar chart showing improvements
- Console output with detailed metrics

## What Each File Does

| File | Purpose | When to Use |
|------|---------|-------------|
| `interactive_tutorial.py` | Step-by-step walkthrough | **Start here!** Learn the concepts |
| `run_experiments.py` | Quick experiment runner | Run pre-configured experiments |
| `rag_sac_implementation.py` | Core implementation | Import for custom code |
| `visualization_utils.py` | Plotting tools | Analyze and visualize results |
| `STEP_BY_STEP_GUIDE.md` | Detailed documentation | Deep dive into implementation |
| `README.md` | Full documentation | API reference and usage |
| `IMPLEMENTATION_SUMMARY.md` | What was built | Overview of components |

## Quick Code Example

```python
from rag_sac_implementation import RAGRetriever, Document

# 1. Create a document
doc = Document(
    doc_id="my_nda",
    filename="nda.txt",
    content="Your legal document text here...",
    doc_type="NDA"
)

# 2. Initialize retriever with SAC
retriever = RAGRetriever(
    use_sac=True,              # Enable Summary-Augmented Chunking
    summary_length=150,         # 150 character summaries
    chunk_size=500              # 500 character chunks
)

# 3. Index documents
retriever.index_documents([doc])

# 4. Query
results = retriever.retrieve(
    "Can party independently develop similar information?",
    top_k=5
)

# 5. View results
for i, result in enumerate(results, 1):
    print(f"{i}. Score: {result.score:.4f}")
    print(f"   From: {result.chunk.doc_id}")
    print(f"   Text: {result.chunk.content[:100]}...\n")
```

## Understanding the Key Metrics

### DRM (Document-Level Retrieval Mismatch)
- **What it is**: % of retrieved chunks from wrong documents
- **Good value**: < 30%
- **Bad value**: > 60%
- **Paper's finding**: Baseline ~60%, SAC ~30%

### Precision
- **What it is**: How much retrieved text is actually relevant
- **Good value**: > 0.20
- **Interpretation**: If 0.25, then 25% of retrieved text is relevant

### Recall
- **What it is**: How much of the relevant text was retrieved
- **Good value**: > 0.50
- **Interpretation**: If 0.55, then 55% of relevant text was found

## Common Use Cases

### 1. Test with Your Own Documents

```python
# Load your documents
documents = []
for filename in ["doc1.txt", "doc2.txt", "doc3.txt"]:
    with open(filename) as f:
        doc = Document(
            doc_id=filename,
            filename=filename,
            content=f.read()
        )
        documents.append(doc)

# Use SAC
retriever = RAGRetriever(use_sac=True)
retriever.index_documents(documents)

# Query
results = retriever.retrieve("your query", top_k=5)
```

### 2. Compare Baseline vs SAC

```python
# Baseline
baseline = RAGRetriever(use_sac=False)
baseline.index_documents(documents)
baseline_results = baseline.retrieve(query, top_k=5)

# SAC
sac = RAGRetriever(use_sac=True)
sac.index_documents(documents)
sac_results = sac.retrieve(query, top_k=5)

# Compare
print("Baseline:", [r.chunk.doc_id for r in baseline_results])
print("SAC:", [r.chunk.doc_id for r in sac_results])
```

### 3. Evaluate on Multiple Queries

```python
from rag_sac_implementation import Query, RAGEvaluator

queries = [
    Query(
        query_id="q1",
        query_text="Your question here",
        ground_truth_doc_id="correct_doc_id",
        ground_truth_start=0,
        ground_truth_end=500
    ),
    # Add more queries...
]

evaluator = RAGEvaluator()
results = evaluator.evaluate_dataset(
    queries,
    retriever,
    top_k_values=[1, 2, 4, 8, 16]
)

# View metrics
for k, metrics in results.items():
    print(f"Top-{k}: DRM={metrics['drm']:.2f}%, "
          f"Precision={metrics['precision']:.4f}, "
          f"Recall={metrics['recall']:.4f}")
```

## Troubleshooting

### Issue: Installation errors
**Solution**: Make sure Python 3.8+ is installed. Use `--break-system-packages` flag for pip on newer systems.

### Issue: Out of memory
**Solution**: Use a smaller embedding model or reduce batch size:
```python
retriever = RAGRetriever(
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

### Issue: Slow performance
**Solution**: 
1. Use GPU if available
2. Reduce number of documents
3. Use smaller chunk sizes

### Issue: Poor results
**Solution**:
1. Check that summaries are informative
2. Try different chunk sizes (200, 500, 800)
3. Try different summary lengths (150, 300)
4. Use better embedding model (thenlper/gte-large)

## Next Steps

### For Learning:
1. ✅ Run `interactive_tutorial.py` 
2. ✅ Read `STEP_BY_STEP_GUIDE.md`
3. ✅ Experiment with your own data
4. ✅ Read the original paper

### For Research:
1. ✅ Use `run_experiments.py all` to reproduce results
2. ✅ Modify hyperparameters in the code
3. ✅ Add your own evaluation metrics
4. ✅ Compare with other methods

### For Production:
1. ✅ Integrate `RAGRetriever` into your pipeline
2. ✅ Use real LLM API for summaries (OpenAI, Anthropic)
3. ✅ Add reranking and query optimization
4. ✅ Monitor DRM and other metrics

## Key Takeaway

**Summary-Augmented Chunking (SAC) solves a critical problem in RAG systems: retrieving information from the wrong source document.**

By adding a simple document-level summary to each chunk before embedding, you can:
- ✅ Reduce Document-Level Retrieval Mismatch by ~50%
- ✅ Improve precision by ~100%
- ✅ Improve recall by ~50%

**And it's incredibly simple to implement!**

## Help & Support

- Read `README.md` for full API documentation
- Check `STEP_BY_STEP_GUIDE.md` for detailed explanations
- See `IMPLEMENTATION_SUMMARY.md` for what's included
- Look at code comments for inline documentation

## Paper Reference

Reuter, M., Lingenberg, T., et al. (2025). "Towards Reliable Retrieval in RAG Systems for Large Legal Datasets." *Proceedings of the Natural Legal Language Processing Workshop 2025*, pages 17-30.

---

**Ready to get started?** Run:
```bash
python interactive_tutorial.py
```

**Happy coding!** 🚀
