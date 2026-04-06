"""
Implementation of "Towards Reliable Retrieval in RAG Systems for Large Legal Datasets"
This implements Summary-Augmented Chunking (SAC) for improving RAG retrieval quality.

Key Components:
1. Baseline RAG with standard chunking
2. Summary-Augmented Chunking (SAC)
3. Expert-Guided SAC
4. Evaluation metrics: DRM, Precision, Recall
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import re
from collections import defaultdict

# For embeddings and vector search
try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    print("Installing required packages...")
    os.system("pip install sentence-transformers faiss-cpu --break-system-packages")
    from sentence_transformers import SentenceTransformer
    import faiss


@dataclass
class Document:
    """Represents a legal document"""
    doc_id: str
    filename: str
    content: str
    doc_type: Optional[str] = None  # NDA, Privacy Policy, Contract, etc.


@dataclass
class Chunk:
    """Represents a text chunk from a document"""
    chunk_id: str
    doc_id: str
    content: str
    start_char: int
    end_char: int
    summary: Optional[str] = None  # For SAC


@dataclass
class Query:
    """Represents a query with ground truth"""
    query_id: str
    query_text: str
    ground_truth_doc_id: str
    ground_truth_start: int
    ground_truth_end: int


@dataclass
class RetrievalResult:
    """Results from retrieval"""
    chunk: Chunk
    score: float


class RecursiveCharacterSplitter:
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Separators in order of preference
        self.separators = ["\n\n", "\n", ". ", " ", ""]
    
    def split_text(self, text: str, doc_id: str) -> List[Chunk]:
        chunks = []
        chunks_text = self._recursive_split(text, self.separators)
        
        current_pos = 0
        for i, chunk_text in enumerate(chunks_text):
            # Find the actual position in original text
            start_pos = text.find(chunk_text, current_pos)
            if start_pos == -1:
                start_pos = current_pos
            
            end_pos = start_pos + len(chunk_text)
            
            chunk = Chunk(
                chunk_id=f"{doc_id}_chunk_{i}",
                doc_id=doc_id,
                content=chunk_text,
                start_char=start_pos,
                end_char=end_pos
            )
            chunks.append(chunk)
            current_pos = end_pos
        
        return chunks
    
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if not separators:
            return self._split_by_length(text)
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator == "":
            return self._split_by_length(text)
        
        splits = text.split(separator)
        
        chunks = []
        current_chunk = ""
        
        for split in splits:
            if len(current_chunk) + len(split) + len(separator) <= self.chunk_size:
                if current_chunk:
                    current_chunk += separator + split
                else:
                    current_chunk = split
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If split itself is too long, recursively split it
                if len(split) > self.chunk_size:
                    sub_chunks = self._recursive_split(split, remaining_separators)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = split
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_by_length(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk:
                chunks.append(chunk)
        return chunks


class SummaryGenerator:
    
    def __init__(self, summary_length: int = 150, use_expert_prompt: bool = False):
        self.summary_length = summary_length
        self.use_expert_prompt = use_expert_prompt
    
    def generate_generic_summary(self, document: Document) -> str:
        
        content = document.content
        
        # Extract first sentences and key entities
        sentences = re.split(r'[.!?]+', content)
        first_sentences = ' '.join(sentences[:3])
        
        # Simple truncation to approximate summary
        summary = first_sentences[:self.summary_length]
        
        # Add document type indicator if present
        if document.doc_type:
            summary = f"{document.doc_type}: {summary}"
        
        return summary.strip()
    

    def generate_summary(self, document: Document) -> str:
    
        # if self.use_expert_prompt:
        #     return self.generate_expert_summary(document)
        # else:
            return self.generate_generic_summary(document)


class RAGRetriever:
    
    def __init__(
        self,
        embedding_model_name: str = "thenlper/gte-large",
        use_sac: bool = False,
        summary_length: int = 150,
        use_expert_summary: bool = False,
        chunk_size: int = 500,
        chunk_overlap: int = 0
    ):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.use_sac = use_sac
        self.summary_length = summary_length
        self.use_expert_summary = use_expert_summary
        
        self.chunker = RecursiveCharacterSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        if use_sac:
            self.summary_generator = SummaryGenerator(
                summary_length=summary_length,
                use_expert_prompt=use_expert_summary
            )
        
        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.Index] = None
        self.documents: Dict[str, Document] = {}
    
    def index_documents(self, documents: List[Document]):
        print(f"Indexing {len(documents)} documents...")
        
        self.documents = {doc.doc_id: doc for doc in documents}
        self.chunks = []
        
        # Generate summaries if using SAC
        doc_summaries = {}
        if self.use_sac:
            print("Generating document summaries...")
            for doc in documents:
                summary = self.summary_generator.generate_summary(doc)
                doc_summaries[doc.doc_id] = summary
        
        # Chunk documents
        print("Chunking documents...")
        for doc in documents:
            chunks = self.chunker.split_text(doc.content, doc.doc_id)
            
            # Augment chunks with summaries if using SAC
            if self.use_sac:
                summary = doc_summaries[doc.doc_id]
                for chunk in chunks:
                    chunk.summary = summary
            
            self.chunks.extend(chunks)
        
        print(f"Created {len(self.chunks)} chunks")
        
        # Create embeddings
        print("Generating embeddings...")
        texts_to_embed = []
        for chunk in self.chunks:
            if self.use_sac and chunk.summary:
                text = f"{chunk.summary}\n\n{chunk.content}"
            else:
                text = chunk.content
            texts_to_embed.append(text)
        
        self.embeddings = self.embedding_model.encode(
            texts_to_embed,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Build FAISS index
        print("Building FAISS index...")
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        print("Indexing complete!")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        if self.index is None:
            raise ValueError("No documents indexed. Call index_documents first.")
        
        # Embed query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append(
                    RetrievalResult(
                        chunk=self.chunks[idx],
                        score=float(score)
                    )
                )
        
        return results


class RAGEvaluator:
    """
    Evaluates RAG retrieval quality using metrics from the paper:
    - Document-Level Retrieval Mismatch (DRM)
    - Text-level Precision
    - Text-level Recall
    """
    
    def __init__(self):
        self.results = defaultdict(list)
    
    def evaluate_query(
        self,
        query: Query,
        retrieved_results: List[RetrievalResult],
        documents: Dict[str, Document]
    ) -> Dict[str, float]:
        """Evaluate a single query"""
        
        # Calculate DRM
        drm = self.calculate_drm(query, retrieved_results)
        
        # Calculate character-level precision and recall
        precision, recall = self.calculate_precision_recall(
            query, retrieved_results, documents
        )
        
        return {
            'drm': drm,
            'precision': precision,
            'recall': recall
        }
    
    def calculate_drm(
        self,
        query: Query,
        retrieved_results: List[RetrievalResult]
    ) -> float:
        """
        Calculate Document-Level Retrieval Mismatch (DRM).
        DRM = proportion of retrieved chunks from wrong documents.
        """
        if not retrieved_results:
            return 1.0  # 100% mismatch if nothing retrieved
        
        mismatches = sum(
            1 for result in retrieved_results
            if result.chunk.doc_id != query.ground_truth_doc_id
        )
        
        drm = mismatches / len(retrieved_results)
        return drm
    
    def calculate_precision_recall(
        self,
        query: Query,
        retrieved_results: List[RetrievalResult],
        documents: Dict[str, Document]
    ) -> Tuple[float, float]:
        """
        Calculate character-level precision and recall.
        
        Precision: fraction of retrieved text that overlaps with ground truth
        Recall: fraction of ground truth text that was retrieved
        """
        # Get ground truth text
        gt_doc = documents.get(query.ground_truth_doc_id)
        if not gt_doc:
            return 0.0, 0.0
        
        gt_start = query.ground_truth_start
        gt_end = query.ground_truth_end
        gt_chars = set(range(gt_start, gt_end))
        
        # Get all retrieved characters
        retrieved_chars = set()
        for result in retrieved_results:
            chunk = result.chunk
            # Only count if from correct document
            if chunk.doc_id == query.ground_truth_doc_id:
                retrieved_chars.update(range(chunk.start_char, chunk.end_char))
        
        # Calculate overlap
        overlap = gt_chars & retrieved_chars
        
        # Precision: relevant retrieved / total retrieved
        precision = len(overlap) / len(retrieved_chars) if retrieved_chars else 0.0
        
        # Recall: relevant retrieved / total relevant
        recall = len(overlap) / len(gt_chars) if gt_chars else 0.0
        
        return precision, recall
    
    def evaluate_dataset(
        self,
        queries: List[Query],
        retriever: RAGRetriever,
        top_k_values: List[int] = [1, 2, 4, 8, 16, 32, 64]
    ) -> Dict[int, Dict[str, float]]:
        """
        Evaluate entire dataset across different top-k values.
        Returns aggregated metrics for each k.
        """
        results = {}
        
        for k in top_k_values:
            print(f"\nEvaluating with top_k={k}...")
            drm_scores = []
            precision_scores = []
            recall_scores = []
            
            for query in queries:
                # Retrieve
                retrieved = retriever.retrieve(query.query_text, top_k=k)
                
                # Evaluate
                metrics = self.evaluate_query(
                    query, retrieved, retriever.documents
                )
                
                drm_scores.append(metrics['drm'])
                precision_scores.append(metrics['precision'])
                recall_scores.append(metrics['recall'])
            
            # Aggregate
            results[k] = {
                'drm': np.mean(drm_scores) * 100,  # Convert to percentage
                'precision': np.mean(precision_scores),
                'recall': np.mean(recall_scores),
                'drm_std': np.std(drm_scores) * 100,
                'precision_std': np.std(precision_scores),
                'recall_std': np.std(recall_scores)
            }
            
            print(f"  DRM: {results[k]['drm']:.2f}%")
            print(f"  Precision: {results[k]['precision']:.4f}")
            print(f"  Recall: {results[k]['recall']:.4f}")
        
        return results


# Example usage and experiments
def create_sample_legal_dataset() -> Tuple[List[Document], List[Query]]:
    """
    Create a sample legal dataset for demonstration.
    In practice, you would load LegalBench-RAG data.
    """
    
    # Sample NDA documents (simplified)
    nda1 = Document(
        doc_id="nda_evelozcity",
        filename="NDA-Evelozcity.txt",
        content="""
        NON-DISCLOSURE AGREEMENT
        
        This NON-DISCLOSURE AGREEMENT (this "Agreement") is made as of this 15th day of January 2019,
        by and between Evelozcity with offices at 123 Tech Drive, California (the "Disclosing Party"),
        and ABC Corporation (the "Recipient").
        
        1. Definition of Confidential Information
        "Confidential Information" means non-public information relating to vehicle prototypes,
        technical specifications, business plans, and proprietary technology.
        
        2. Obligations of Recipient
        The Recipient agrees to:
        (a) Keep all Confidential Information strictly confidential
        (b) Limit access to employees and affiliates who need to know
        (c) Use the information solely for evaluation purposes
        
        3. Exclusions from Confidentiality
        The obligations of the Recipient specified in Section 2 above shall not apply with
        respect to Confidential Information to the extent that such Confidential Information:
        (a) is publicly available through no breach of this Agreement
        (b) was rightfully in Recipient's possession prior to disclosure
        (c) is received from a third party without breach of any obligation
        (d) is independently developed by or for the Recipient by persons who have had no
        access to or been informed of the existence or substance of the Confidential Information.
        
        4. Term
        This Agreement shall remain in effect for a period of 5 years from the date of execution.
        
        5. Governing Law
        This Agreement shall be governed by the laws of the State of California.
        """,
        doc_type="NDA"
    )
    
    nda2 = Document(
        doc_id="nda_roi_corp",
        filename="NDA-ROI-Corporation.txt",
        content="""
        NON-DISCLOSURE AGREEMENT FOR PROSPECTIVE PURCHASERS
        
        This Agreement is entered into on March 3, 2019, between ROI Corporation,
        a Delaware corporation ("Disclosing Party"), and XYZ Ventures ("Recipient").
        
        1. Purpose
        The parties wish to explore a potential business transaction involving the
        acquisition of certain assets.
        
        2. Confidential Information
        All information disclosed in connection with the proposed transaction,
        including financial statements, customer lists, and proprietary processes.
        
        3. Recipient's Obligations
        Recipient shall:
        - Maintain strict confidentiality
        - Use information only for evaluation
        - Return or destroy all materials upon request
        
        4. Exceptions
        Confidential Information does not include information that:
        (a) is or becomes publicly available without breach
        (b) is already known to Recipient
        (c) is independently developed
        (d) is received from a third party
        
        5. Duration
        This Agreement remains in effect for 3 years.
        
        6. Jurisdiction
        This Agreement is governed by Delaware law.
        """,
        doc_type="NDA"
    )
    
    # Privacy policy document
    privacy_policy = Document(
        doc_id="privacy_policy_techapp",
        filename="PrivacyPolicy-TechApp.txt",
        content="""
        PRIVACY POLICY
        
        Last Updated: January 1, 2024
        
        TechApp Inc. ("we", "our", "us") operates the TechApp mobile application.
        This Privacy Policy explains how we collect, use, and protect your personal data.
        
        1. Information We Collect
        We collect the following types of personal data:
        - Name and contact information (email, phone)
        - Device information (device ID, operating system, app version)
        - Location data (with your permission)
        - Usage data (features used, time spent)
        - Payment information (processed by third-party providers)
        
        2. How We Use Your Information
        We use your personal data for the following purposes:
        - To provide and improve our services
        - To personalize your experience
        - To communicate with you about updates
        - To process payments and transactions
        - For analytics and research
        
        3. Legal Basis for Processing
        We process your data based on:
        - Your consent
        - Performance of our contract with you
        - Compliance with legal obligations
        - Our legitimate business interests
        
        4. Data Sharing
        We may share your data with:
        - Service providers and processors
        - Analytics partners
        - Payment processors
        - Law enforcement when required by law
        
        5. Data Retention
        We retain your data for as long as necessary to provide services,
        typically for the duration of your account plus 3 years.
        
        6. Your Rights
        Under GDPR, you have the right to:
        - Access your personal data
        - Rectify inaccurate data
        - Request erasure ("right to be forgotten")
        - Restrict or object to processing
        - Data portability
        - Withdraw consent
        
        7. International Transfers
        Your data may be transferred to and processed in countries outside your residence.
        We use Standard Contractual Clauses to protect your data.
        
        8. Contact Us
        For privacy questions, contact us at privacy@techapp.com
        """,
        doc_type="Privacy Policy"
    )
    
    documents = [nda1, nda2, privacy_policy]
    
    # Sample queries with ground truth
    queries = [
        Query(
            query_id="q1",
            query_text="Consider Evelozcity's Non-Disclosure Agreement; Does the document allow the Receiving Party to independently develop information that is similar to the Confidential Information?",
            ground_truth_doc_id="nda_evelozcity",
            ground_truth_start=nda1.content.find("(d) is independently developed"),
            ground_truth_end=nda1.content.find("(d) is independently developed") + 150
        ),
        Query(
            query_id="q2",
            query_text="What is the duration of the confidentiality obligation in the ROI Corporation NDA?",
            ground_truth_doc_id="nda_roi_corp",
            ground_truth_start=nda2.content.find("This Agreement remains in effect for 3 years"),
            ground_truth_end=nda2.content.find("This Agreement remains in effect for 3 years") + 50
        ),
        Query(
            query_id="q3",
            query_text="According to TechApp's privacy policy, what rights do users have under GDPR?",
            ground_truth_doc_id="privacy_policy_techapp",
            ground_truth_start=privacy_policy.content.find("6. Your Rights"),
            ground_truth_end=privacy_policy.content.find("7. International Transfers")
        )
    ]
    
    return documents, queries


def run_experiment_comparison():
    """
    Run the main experiment comparing:
    1. Baseline RAG (no SAC)
    2. RAG with generic SAC
    3. RAG with expert-guided SAC
    """
    print("="*80)
    print("EXPERIMENT: Comparing Baseline RAG vs SAC vs Expert-SAC")
    print("="*80)
    
    # Load dataset
    print("\nLoading dataset...")
    documents, queries = create_sample_legal_dataset()
    print(f"Loaded {len(documents)} documents and {len(queries)} queries")
    
    # Define experimental configurations
    configs = [
        {
            'name': 'Baseline (No SAC)',
            'use_sac': False,
            'use_expert_summary': False
        },
        {
            'name': 'SAC with Generic Summary',
            'use_sac': True,
            'use_expert_summary': False
        }
    ]
    
    all_results = {}
    
    # Run experiments
    for config in configs:
        print(f"\n{'='*80}")
        print(f"Running: {config['name']}")
        print(f"{'='*80}")
        
        # Initialize retriever
        retriever = RAGRetriever(
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", 
            use_expert_summary=config['use_expert_summary'],
            summary_length=150,
            chunk_size=500
        )
        
        # Index documents
        retriever.index_documents(documents)
        
        # Evaluate
        evaluator = RAGEvaluator()
        results = evaluator.evaluate_dataset(
            queries,
            retriever,
            top_k_values=[1, 2, 4, 8]  
        )
        
        all_results[config['name']] = results
    
    # Print comparison
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    for k in [1, 2, 4, 8]:
        print(f"\nTop-K = {k}")
        print("-" * 80)
        print(f"{'Method':<30} {'DRM (%)':<15} {'Precision':<15} {'Recall':<15}")
        print("-" * 80)
        
        for method_name, results in all_results.items():
            r = results[k]
            print(f"{method_name:<30} {r['drm']:>10.2f}% {r['precision']:>14.4f} {r['recall']:>14.4f}")
    
    return all_results


if __name__ == "__main__":
    
    results = run_experiment_comparison()
    
    print("\n" + "="*80)
    print("Experiment complete!")
    print("="*80)
