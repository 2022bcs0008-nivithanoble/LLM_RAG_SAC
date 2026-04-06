"""
Interactive Tutorial: Summary-Augmented Chunking (SAC) for RAG

This tutorial walks through implementing the key experiments from the paper
step by step with explanations and visualizations.
"""

print("""
================================================================================
          TUTORIAL: Summary-Augmented Chunking for RAG Systems
================================================================================

This tutorial implements the paper:
"Towards Reliable Retrieval in RAG Systems for Large Legal Datasets"

We'll cover:
1. The problem: Document-Level Retrieval Mismatch (DRM)
2. The solution: Summary-Augmented Chunking (SAC)
3. Running experiments
4. Analyzing results

Let's get started!
================================================================================
""")

import time
from rag_sac_implementation import *
from visualization_utils import *

def pause_for_user():
    """Helper to pause between tutorial steps"""
    input("\nPress Enter to continue...")

# =============================================================================
# STEP 1: Understanding the Problem
# =============================================================================

print("\n" + "="*80)
print("STEP 1: Understanding the Document-Level Retrieval Mismatch Problem")
print("="*80)

print("""
Imagine you have 100 Non-Disclosure Agreements (NDAs) in your database.
They all look very similar:
- Same structure
- Same legal clauses  
- Only differ in: party names, dates, specific terms

USER QUERY: "Can Evelozcity independently develop similar information?"

WHAT HAPPENS IN STANDARD RAG:
1. Documents are chunked into 500-character pieces
2. Each chunk is embedded INDEPENDENTLY (no document context!)
3. Query: "independently develop" searches for similar chunks
4. Result: Finds chunks about "independent development" from ANY NDA
5. PROBLEM: Returns the WRONG NDA because chunks look identical!

This is Document-Level Retrieval Mismatch (DRM).
""")

pause_for_user()

# =============================================================================
# STEP 2: Create Sample Dataset
# =============================================================================

print("\n" + "="*80)
print("STEP 2: Creating Sample Legal Dataset")
print("="*80)

print("\nLoading sample legal documents...")
documents, queries = create_sample_legal_dataset()

print(f"\n✓ Loaded {len(documents)} documents:")
for doc in documents:
    print(f"  - {doc.filename} ({doc.doc_type})")

print(f"\n✓ Loaded {len(queries)} test queries")
for query in queries:
    print(f"  - {query.query_text[:60]}...")
    print(f"    Expected document: {query.ground_truth_doc_id}")

pause_for_user()

# =============================================================================
# STEP 3: Baseline RAG (No SAC)
# =============================================================================

print("\n" + "="*80)
print("STEP 3: Running Baseline RAG (No Summary Augmentation)")
print("="*80)

print("""
BASELINE PIPELINE:
1. Chunk each document into 500-character pieces
2. Embed each chunk INDEPENDENTLY (no document context)
3. Index in FAISS vector database
4. For queries: retrieve top-k most similar chunks

Let's see how it performs...
""")

print("\nInitializing baseline retriever...")
baseline_retriever = RAGRetriever(
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    use_sac=False,  # ← NO summary augmentation
    chunk_size=500
)

print("Indexing documents...")
baseline_retriever.index_documents(documents)

print("\nTesting on a query...")
test_query = queries[0]
print(f"\nQuery: {test_query.query_text}")
print(f"Expected document: {test_query.ground_truth_doc_id}")

results = baseline_retriever.retrieve(test_query.query_text, top_k=5)

print("\nTop 5 Retrieved Chunks:")
for i, result in enumerate(results, 1):
    is_correct = "✓" if result.chunk.doc_id == test_query.ground_truth_doc_id else "✗"
    print(f"{i}. {is_correct} From: {result.chunk.doc_id}")
    print(f"   Score: {result.score:.4f}")
    print(f"   Text: {result.chunk.content[:80]}...")

# Calculate DRM for this query
drm = sum(1 for r in results if r.chunk.doc_id != test_query.ground_truth_doc_id) / len(results)
print(f"\nDocument-Level Retrieval Mismatch (DRM): {drm*100:.1f}%")
print("(This is the % of retrieved chunks from the WRONG document)")

pause_for_user()

# =============================================================================
# STEP 4: Summary-Augmented Chunking (SAC)
# =============================================================================

print("\n" + "="*80)
print("STEP 4: Running RAG with Summary-Augmented Chunking (SAC)")
print("="*80)

print("""
SAC PIPELINE:
1. For EACH document, generate a 150-character summary
2. Chunk each document into 500-character pieces
3. PREPEND the summary to each chunk
4. Embed the augmented chunks (summary + chunk content)
5. Index in FAISS vector database

Now each chunk has GLOBAL CONTEXT about which document it came from!

Let's see the improvement...
""")

print("\nInitializing SAC retriever...")
sac_retriever = RAGRetriever(
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    use_sac=True,  # ← WITH summary augmentation
    use_expert_summary=False,  # Generic summaries
    summary_length=150,
    chunk_size=500
)

print("Indexing documents with SAC...")
print("(This will generate summaries for each document)")
sac_retriever.index_documents(documents)

print("\n" + "="*60)
print("GENERATED SUMMARIES:")
print("="*60)
for doc_id, doc in sac_retriever.documents.items():
    # Get summary from first chunk
    first_chunk = [c for c in sac_retriever.chunks if c.doc_id == doc_id][0]
    if first_chunk.summary:
        print(f"\n{doc.filename}:")
        print(f"  Summary: {first_chunk.summary}")
print("="*60)

print("\nTesting the SAME query with SAC...")
print(f"\nQuery: {test_query.query_text}")
print(f"Expected document: {test_query.ground_truth_doc_id}")

sac_results = sac_retriever.retrieve(test_query.query_text, top_k=5)

print("\nTop 5 Retrieved Chunks (WITH SAC):")
for i, result in enumerate(sac_results, 1):
    is_correct = "✓" if result.chunk.doc_id == test_query.ground_truth_doc_id else "✗"
    print(f"{i}. {is_correct} From: {result.chunk.doc_id}")
    print(f"   Score: {result.score:.4f}")
    # Show the summary that was prepended
    if result.chunk.summary:
        print(f"   Summary: {result.chunk.summary}")
    print(f"   Chunk: {result.chunk.content[:60]}...")

# Calculate DRM with SAC
sac_drm = sum(1 for r in sac_results if r.chunk.doc_id != test_query.ground_truth_doc_id) / len(sac_results)

print(f"\n{'='*60}")
print("COMPARISON:")
print(f"{'='*60}")
print(f"Baseline DRM:  {drm*100:.1f}%")
print(f"SAC DRM:       {sac_drm*100:.1f}%")
print(f"Improvement:   {(drm-sac_drm)*100:.1f} percentage points")
print(f"{'='*60}")

pause_for_user()

# =============================================================================
# STEP 5: Full Evaluation on All Queries
# =============================================================================

print("\n" + "="*80)
print("STEP 5: Comprehensive Evaluation on All Queries")
print("="*80)

print("""
Now let's evaluate both systems on ALL queries across different top-k values.

We'll measure:
- DRM: % of chunks from wrong documents
- Precision: Fraction of retrieved text that's relevant
- Recall: Fraction of relevant text that was retrieved
""")

evaluator = RAGEvaluator()

print("\nEvaluating Baseline...")
baseline_results = evaluator.evaluate_dataset(
    queries,
    baseline_retriever,
    top_k_values=[1, 2, 4, 8]
)

print("\nEvaluating SAC...")
sac_results = evaluator.evaluate_dataset(
    queries,
    sac_retriever,
    top_k_values=[1, 2, 4, 8]
)

# Organize results for visualization
all_results = {
    'Baseline': baseline_results,
    'SAC-Generic': sac_results
}

pause_for_user()

# =============================================================================
# STEP 6: Visualization and Analysis
# =============================================================================

print("\n" + "="*80)
print("STEP 6: Visualizing Results")
print("="*80)

print("\nGenerating visualizations matching the paper's figures...")

viz = ResultsVisualizer(all_results)
viz.generate_all_visualizations(output_dir='/home/claude')

print("""
✓ Created visualizations:
  - drm_comparison.png: DRM curves (like Figure 2 in paper)
  - precision_recall.png: Precision/Recall curves (like Figure 3)
  - improvements.png: Bar chart of improvements
  
Check the current directory for the PNG files!
""")

pause_for_user()

# =============================================================================
# STEP 7: Key Findings
# =============================================================================

print("\n" + "="*80)
print("STEP 7: Key Findings and Conclusions")
print("="*80)

print("""
WHAT WE LEARNED:

1. THE PROBLEM:
   - Standard RAG suffers from high DRM (50-95%) on similar legal docs
   - Chunks lose document context when embedded independently
   - Retrievers can't distinguish between structurally similar documents

2. THE SOLUTION:
   - Summary-Augmented Chunking (SAC) adds document-level context
   - Just ONE summary per document, prepended to all its chunks
   - Simple, cheap, and highly effective

3. THE RESULTS:
   - DRM cuts in half with SAC
   - Precision and recall improve significantly
   - Generic summaries work as well as expert-crafted ones

4. WHY IT WORKS:
   - Summaries inject "document fingerprints" into embeddings
   - Retriever can now distinguish "Evelozcity NDA" from "ROI Corp NDA"
   - Global context preserved without architectural changes

5. PRACTICAL IMPLICATIONS:
   - Easy to integrate into existing RAG pipelines
   - Minimal computational overhead (one LLM call per document)
   - Scalable to large, dynamic legal databases
   - Works with any embedding model and vector database
""")

print("\n" + "="*80)
print("EXPERIMENT COMPLETE!")
print("="*80)

print("""
NEXT STEPS:

1. Try on your own legal documents
2. Experiment with different summary lengths (100, 150, 300 chars)
3. Test different chunk sizes (200, 500, 800 chars)
4. Try expert-guided summaries for your specific document types
5. Combine SAC with other techniques (reranking, query optimization)

For full implementation details, see:
- rag_sac_implementation.py: Complete implementation
- STEP_BY_STEP_GUIDE.md: Detailed explanations
- visualization_utils.py: Plotting and analysis tools

Thank you for following along!
""")

# =============================================================================
# BONUS: Interactive Exploration
# =============================================================================

print("\n" + "="*80)
print("BONUS: Interactive Exploration")
print("="*80)

def interactive_retrieval():
    """Let user try custom queries"""
    print("\nTry your own queries!")
    print("Type 'quit' to exit")
    
    while True:
        user_query = input("\nEnter a query: ").strip()
        
        if user_query.lower() == 'quit':
            break
        
        if not user_query:
            continue
        
        print("\n--- BASELINE RESULTS ---")
        baseline_res = baseline_retriever.retrieve(user_query, top_k=3)
        for i, r in enumerate(baseline_res, 1):
            print(f"{i}. {r.chunk.doc_id}: {r.chunk.content[:100]}...")
        
        print("\n--- SAC RESULTS ---")
        sac_res = sac_retriever.retrieve(user_query, top_k=3)
        for i, r in enumerate(sac_res, 1):
            print(f"{i}. {r.chunk.doc_id}: {r.chunk.content[:100]}...")

print("\nWould you like to try interactive retrieval? (yes/no)")
response = input("> ").strip().lower()

if response == 'yes':
    interactive_retrieval()

print("\n" + "="*80)
print("Tutorial complete! Check the output files for visualizations.")
print("="*80)
