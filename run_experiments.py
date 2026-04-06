#!/usr/bin/env python3
"""
Quick runner script for SAC experiments
Usage: python run_experiments.py [baseline|sac|expert|all|tutorial]
"""

import sys
import os

def run_baseline():
    """Run baseline RAG experiment"""
    print("Running Baseline RAG Experiment...")
    from rag_sac_implementation import (
        create_sample_legal_dataset,
        RAGRetriever,
        RAGEvaluator
    )
    
    documents, queries = create_sample_legal_dataset()
    
    retriever = RAGRetriever(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        use_sac=False,
        chunk_size=500
    )
    
    retriever.index_documents(documents)
    
    evaluator = RAGEvaluator()
    results = evaluator.evaluate_dataset(queries, retriever, top_k_values=[1, 2, 4, 8])
    
    print("\n✓ Baseline experiment complete!")
    return results

def run_sac():
    """Run SAC with generic summaries"""
    print("Running SAC with Generic Summaries...")
    from rag_sac_implementation import (
        create_sample_legal_dataset,
        RAGRetriever,
        RAGEvaluator
    )
    
    documents, queries = create_sample_legal_dataset()
    
    retriever = RAGRetriever(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        use_sac=True,
        use_expert_summary=False,
        summary_length=150,
        chunk_size=500
    )
    
    retriever.index_documents(documents)
    
    evaluator = RAGEvaluator()
    results = evaluator.evaluate_dataset(queries, retriever, top_k_values=[1, 2, 4, 8])
    
    print("\n✓ SAC experiment complete!")
    return results

def run_expert():
    """Run SAC with expert-guided summaries"""
    print("Running SAC with Expert-Guided Summaries...")
    from rag_sac_implementation import (
        create_sample_legal_dataset,
        RAGRetriever,
        RAGEvaluator
    )
    
    documents, queries = create_sample_legal_dataset()
    
    retriever = RAGRetriever(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        use_sac=True,
        use_expert_summary=True,
        summary_length=150,
        chunk_size=500
    )
    
    retriever.index_documents(documents)
    
    evaluator = RAGEvaluator()
    results = evaluator.evaluate_dataset(queries, retriever, top_k_values=[1, 2, 4, 8])
    
    print("\n✓ Expert-SAC experiment complete!")
    return results

def run_all():
    """Run all experiments and generate visualizations"""
    print("="*80)
    print("Running ALL Experiments")
    print("="*80)
    
    from visualization_utils import ResultsVisualizer
    
    baseline_results = run_baseline()
    sac_results = run_sac()
    expert_results = run_expert()
    
    all_results = {
        'Baseline': baseline_results,
        'SAC-Generic': sac_results,
        'SAC-Expert': expert_results
    }
    
    print("\nGenerating visualizations...")
    viz = ResultsVisualizer(all_results)
    viz.generate_all_visualizations(output_dir='.')
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - drm_comparison.png")
    print("  - precision_recall.png")
    print("  - improvements.png")

def run_tutorial():
    """Run interactive tutorial"""
    print("Starting Interactive Tutorial...")
    os.system("python interactive_tutorial.py")

def print_usage():
    """Print usage information"""
    print("""
Usage: python run_experiments.py [mode]

Modes:
  baseline   - Run baseline RAG (no SAC)
  sac        - Run SAC with generic summaries
  expert     - Run SAC with expert-guided summaries
  all        - Run all experiments and generate plots (default)
  tutorial   - Run interactive tutorial

Examples:
  python run_experiments.py baseline
  python run_experiments.py all
  python run_experiments.py tutorial
    """)

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "all":
        run_all()
    elif sys.argv[1] == "baseline":
        run_baseline()
    elif sys.argv[1] == "sac":
        run_sac()
    elif sys.argv[1] == "expert":
        run_expert()
    elif sys.argv[1] == "tutorial":
        run_tutorial()
    elif sys.argv[1] in ["-h", "--help", "help"]:
        print_usage()
    else:
        print(f"Unknown mode: {sys.argv[1]}")
        print_usage()
        sys.exit(1)
