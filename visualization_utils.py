"""
Visualization and Analysis Utilities for RAG-SAC Experiments
Creates plots and analyses matching the paper's figures and tables
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import json


class ResultsVisualizer:
    """Create visualizations matching the paper's figures"""
    
    def __init__(self, results_dict: Dict):
        """
        results_dict format:
        {
            'Baseline': {k: {'drm': float, 'precision': float, 'recall': float}},
            'SAC-Generic': {...},
            'SAC-Expert': {...}
        }
        """
        self.results = results_dict
        self.colors = {
            'Baseline': '#1f77b4',
            'SAC-Generic': '#ff7f0e',
            'SAC-Expert': '#2ca02c'
        }
    
    def plot_drm_comparison(self, save_path='drm_comparison.png'):
        """
        Recreate Figure 2 from the paper:
        DRM curves for Baseline vs SAC
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Baseline
        ax1.set_title('DRM for Standard RAG', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Top-K', fontsize=12)
        ax1.set_ylabel('DRM (%)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 100])
        
        if 'Baseline' in self.results:
            k_values = sorted(self.results['Baseline'].keys())
            drm_values = [self.results['Baseline'][k]['drm'] for k in k_values]
            ax1.plot(k_values, drm_values, 
                    marker='o', linewidth=2, markersize=8,
                    color=self.colors['Baseline'],
                    label='Baseline')
            ax1.fill_between(k_values, drm_values, alpha=0.2, 
                           color=self.colors['Baseline'])
        
        ax1.legend(fontsize=10)
        
        # Plot 2: All methods including SAC
        ax2.set_title('DRM for RAG using SAC', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Top-K', fontsize=12)
        ax2.set_ylabel('DRM (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 100])
        
        for method, color in self.colors.items():
            if method in self.results:
                k_values = sorted(self.results[method].keys())
                drm_values = [self.results[method][k]['drm'] for k in k_values]
                ax2.plot(k_values, drm_values,
                        marker='o', linewidth=2, markersize=8,
                        color=color, label=method)
        
        ax2.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved DRM comparison to {save_path}")
        plt.close()
    
    def plot_precision_recall(self, save_path='precision_recall.png'):
        """
        Recreate Figure 3 from the paper:
        Precision and Recall curves
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Precision plot
        ax1.set_title('Text-Level Precision', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Top-K', fontsize=12)
        ax1.set_ylabel('Precision', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        for method, color in self.colors.items():
            if method in self.results:
                k_values = sorted(self.results[method].keys())
                prec_values = [self.results[method][k]['precision'] for k in k_values]
                ax1.plot(k_values, prec_values,
                        marker='s', linewidth=2, markersize=8,
                        color=color, label=method)
        
        ax1.legend(fontsize=10)
        
        # Recall plot
        ax2.set_title('Text-Level Recall', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Top-K', fontsize=12)
        ax2.set_ylabel('Recall', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        for method, color in self.colors.items():
            if method in self.results:
                k_values = sorted(self.results[method].keys())
                rec_values = [self.results[method][k]['recall'] for k in k_values]
                ax2.plot(k_values, rec_values,
                        marker='^', linewidth=2, markersize=8,
                        color=color, label=method)
        
        ax2.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved precision/recall to {save_path}")
        plt.close()
    
    def create_results_table(self, k_value=8):
        """
        Create a formatted results table for a specific k value
        """
        print(f"\n{'='*80}")
        print(f"RESULTS TABLE (Top-K = {k_value})")
        print(f"{'='*80}")
        print(f"{'Method':<25} {'DRM (%)':<15} {'Precision':<15} {'Recall':<15}")
        print(f"{'-'*80}")
        
        for method in ['Baseline', 'SAC-Generic', 'SAC-Expert']:
            if method in self.results and k_value in self.results[method]:
                r = self.results[method][k_value]
                print(f"{method:<25} {r['drm']:>10.2f}% {r['precision']:>14.4f} {r['recall']:>14.4f}")
        
        print(f"{'='*80}\n")
    
    def plot_improvement_bar_chart(self, save_path='improvements.png'):
        """
        Bar chart showing % improvement of SAC over Baseline
        """
        if 'Baseline' not in self.results or 'SAC-Generic' not in self.results:
            print("Need both Baseline and SAC-Generic results")
            return
        
        # Calculate improvements at k=8 (representative value)
        k = 8
        baseline = self.results['Baseline'][k]
        sac = self.results['SAC-Generic'][k]
        
        # DRM reduction (lower is better, so negative improvement is good)
        drm_reduction = ((baseline['drm'] - sac['drm']) / baseline['drm']) * 100
        
        # Precision increase
        prec_increase = ((sac['precision'] - baseline['precision']) / baseline['precision']) * 100
        
        # Recall increase
        rec_increase = ((sac['recall'] - baseline['recall']) / baseline['recall']) * 100
        
        metrics = ['DRM Reduction', 'Precision Increase', 'Recall Increase']
        improvements = [drm_reduction, prec_increase, rec_increase]
        colors_list = ['#d62728', '#2ca02c', '#1f77b4']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(metrics, improvements, color=colors_list, alpha=0.7, edgecolor='black')
        
        ax.set_ylabel('Improvement (%)', fontsize=12)
        ax.set_title('SAC Improvements over Baseline (at Top-K=8)', 
                    fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, improvements):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}%', ha='center', va='bottom' if val > 0 else 'top',
                   fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved improvement chart to {save_path}")
        plt.close()
    
    def generate_all_visualizations(self, output_dir='.'):
        """Generate all paper-style visualizations"""
        import os
        
        print("\nGenerating visualizations...")
        
        self.plot_drm_comparison(os.path.join(output_dir, 'drm_comparison.png'))
        self.plot_precision_recall(os.path.join(output_dir, 'precision_recall.png'))
        self.plot_improvement_bar_chart(os.path.join(output_dir, 'improvements.png'))
        
        # Print tables for different k values
        for k in [1, 4, 8, 16]:
            self.create_results_table(k_value=k)
        
        print("All visualizations complete!")


class ErrorAnalyzer:
    """Analyze failure cases and error patterns"""
    
    def __init__(self, retriever, queries, documents):
        self.retriever = retriever
        self.queries = queries
        self.documents = documents
    
    def analyze_high_drm_cases(self, threshold=0.5):
        """Find and analyze queries with high DRM"""
        high_drm_cases = []
        
        for query in self.queries:
            results = self.retriever.retrieve(query.query_text, top_k=5)
            
            # Calculate DRM for this query
            mismatches = sum(
                1 for r in results 
                if r.chunk.doc_id != query.ground_truth_doc_id
            )
            drm = mismatches / len(results) if results else 1.0
            
            if drm >= threshold:
                high_drm_cases.append({
                    'query': query,
                    'drm': drm,
                    'expected_doc': query.ground_truth_doc_id,
                    'retrieved_docs': [r.chunk.doc_id for r in results],
                    'retrieved_chunks': results
                })
        
        return high_drm_cases
    
    def print_error_analysis(self, high_drm_cases):
        """Print detailed analysis of error cases"""
        print("\n" + "="*80)
        print("ERROR ANALYSIS: High DRM Cases")
        print("="*80)
        
        for i, case in enumerate(high_drm_cases[:5], 1):  # Show first 5
            print(f"\nCase {i}:")
            print(f"Query: {case['query'].query_text[:100]}...")
            print(f"DRM: {case['drm']*100:.1f}%")
            print(f"Expected Document: {case['expected_doc']}")
            print(f"Retrieved Documents: {set(case['retrieved_docs'])}")
            
            # Show if summaries would help
            expected_doc = self.documents.get(case['expected_doc'])
            if expected_doc and hasattr(expected_doc, 'summary'):
                print(f"Expected Doc Summary: {expected_doc.summary}")
            
            wrong_docs = [d for d in case['retrieved_docs'] if d != case['expected_doc']]
            if wrong_docs:
                wrong_doc = self.documents.get(wrong_docs[0])
                if wrong_doc and hasattr(wrong_doc, 'summary'):
                    print(f"Wrong Doc Summary: {wrong_doc.summary}")
            
            print("-" * 80)
    
    def compare_summary_effectiveness(self, baseline_results, sac_results):
        """Compare which queries benefit most from SAC"""
        improvements = []
        
        for query in self.queries:
            # Find DRM for this query in both systems
            baseline_drm = self._get_query_drm(query, baseline_results)
            sac_drm = self._get_query_drm(query, sac_results)
            
            improvement = baseline_drm - sac_drm
            
            improvements.append({
                'query': query.query_text,
                'improvement': improvement,
                'baseline_drm': baseline_drm,
                'sac_drm': sac_drm
            })
        
        # Sort by improvement
        improvements.sort(key=lambda x: x['improvement'], reverse=True)
        
        print("\n" + "="*80)
        print("QUERIES WITH BIGGEST IMPROVEMENT FROM SAC")
        print("="*80)
        
        for i, imp in enumerate(improvements[:10], 1):
            print(f"\n{i}. {imp['query'][:80]}...")
            print(f"   Baseline DRM: {imp['baseline_drm']*100:.1f}%")
            print(f"   SAC DRM: {imp['sac_drm']*100:.1f}%")
            print(f"   Improvement: {imp['improvement']*100:.1f} percentage points")
        
        return improvements
    
    def _get_query_drm(self, query, results_by_query):
        """Helper to get DRM for a specific query from results"""
        # This would need to be adapted based on how results are stored
        # Placeholder implementation
        return 0.5


def save_results_to_json(results, filename='results.json'):
    """Save experimental results to JSON file"""
    # Convert numpy types to Python types
    json_results = {}
    for method, k_results in results.items():
        json_results[method] = {}
        for k, metrics in k_results.items():
            json_results[method][str(k)] = {
                metric: float(value) for metric, value in metrics.items()
            }
    
    with open(filename, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to {filename}")


def load_results_from_json(filename='results.json'):
    """Load experimental results from JSON file"""
    with open(filename, 'r') as f:
        json_results = json.load(f)
    
    # Convert string keys back to integers
    results = {}
    for method, k_results in json_results.items():
        results[method] = {}
        for k_str, metrics in k_results.items():
            results[method][int(k_str)] = metrics
    
    return results


# Example usage
if __name__ == "__main__":
    # Mock results for demonstration
    mock_results = {
        'Baseline': {
            1: {'drm': 85.0, 'precision': 0.08, 'recall': 0.12},
            2: {'drm': 80.0, 'precision': 0.10, 'recall': 0.18},
            4: {'drm': 75.0, 'precision': 0.12, 'recall': 0.25},
            8: {'drm': 70.0, 'precision': 0.11, 'recall': 0.35},
            16: {'drm': 65.0, 'precision': 0.09, 'recall': 0.48},
            32: {'drm': 60.0, 'precision': 0.07, 'recall': 0.58},
            64: {'drm': 58.0, 'precision': 0.05, 'recall': 0.68},
        },
        'SAC-Generic': {
            1: {'drm': 45.0, 'precision': 0.15, 'recall': 0.20},
            2: {'drm': 40.0, 'precision': 0.18, 'recall': 0.30},
            4: {'drm': 35.0, 'precision': 0.22, 'recall': 0.42},
            8: {'drm': 30.0, 'precision': 0.25, 'recall': 0.55},
            16: {'drm': 28.0, 'precision': 0.23, 'recall': 0.65},
            32: {'drm': 25.0, 'precision': 0.20, 'recall': 0.72},
            64: {'drm': 23.0, 'precision': 0.18, 'recall': 0.78},
        },
        'SAC-Expert': {
            1: {'drm': 48.0, 'precision': 0.14, 'recall': 0.18},
            2: {'drm': 42.0, 'precision': 0.17, 'recall': 0.28},
            4: {'drm': 37.0, 'precision': 0.20, 'recall': 0.40},
            8: {'drm': 32.0, 'precision': 0.23, 'recall': 0.52},
            16: {'drm': 30.0, 'precision': 0.21, 'recall': 0.62},
            32: {'drm': 27.0, 'precision': 0.19, 'recall': 0.70},
            64: {'drm': 25.0, 'precision': 0.17, 'recall': 0.76},
        }
    }
    
    # Create visualizations
    viz = ResultsVisualizer(mock_results)
    viz.generate_all_visualizations()
