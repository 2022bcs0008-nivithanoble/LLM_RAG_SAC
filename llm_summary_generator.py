"""
LLM-powered SummaryGenerator for SAC (Summary-Augmented Chunking)
Uses Groq API — free tier, Llama 3 model, very fast inference.

HOW TO GET A FREE GROQ API KEY:
  1. Go to https://console.groq.com
  2. Sign up (free, no credit card needed)
  3. Go to API Keys → Create API Key
  4. Copy the key (starts with "gsk_...")
  5. Set it as an environment variable:
       export GROQ_API_KEY="gsk_..."
     OR pass it directly when constructing LLMSummaryGenerator.

INSTALL:
  pip install groq --break-system-packages

USAGE (drop-in replacement for the original SummaryGenerator):
  from llm_summary_generator import LLMSummaryGenerator

  # Replace this in RAGRetriever.__init__:
  #   self.summary_generator = SummaryGenerator(...)
  # With:
  #   self.summary_generator = LLMSummaryGenerator(...)

  generator = LLMSummaryGenerator(
      summary_length=150,
      use_expert_prompt=False   # True = NDA/privacy-aware prompt
  )
  summary = generator.generate_summary(document)
"""

import os
import time
import re
from typing import Optional

try:
    from groq import Groq
except ImportError:
    print("Installing groq SDK...")
    os.system("pip install groq --break-system-packages")
    from groq import Groq

# ── reuse the same dataclass from the main file ──────────────────────────────
try:
    from rag_sac_implementation import Document
except ImportError:
    from dataclasses import dataclass
    @dataclass
    class Document:
        doc_id: str
        filename: str
        content: str
        doc_type: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

GENERIC_SYSTEM_PROMPT = """You are a legal document summarizer.
Your job is to produce a single SHORT summary of a legal document.
The summary must:
- Be under {max_chars} characters (hard limit)
- Identify the parties involved (names, roles)
- State the core purpose of the document
- Mention the most important legal topic or obligation
- Be a single line with no bullet points, no newlines
Output ONLY the summary text. Nothing else."""

GENERIC_USER_PROMPT = """Summarize this legal document in under {max_chars} characters.
Focus on: who the parties are, what the document is about, key obligations.

Document:
{content}

Remember: output only the summary, one line, under {max_chars} characters."""


EXPERT_SYSTEM_PROMPT = """You are an expert legal analyst specializing in contract review.
Your job is to produce a dense, structured summary of a legal document for use
as a retrieval hint in a vector search system.
The summary must:
- Be under {max_chars} characters (hard limit)
- Follow the pattern: [DocType]: [Party1] & [Party2] | [key topic 1]; [key topic 2]
- Prioritize: party names, document type, governing law, duration, key obligations
- Be a single line with no bullet points, no newlines
Output ONLY the summary. Nothing else."""

EXPERT_USER_TEMPLATE_NDA = """Summarize this NDA in under {max_chars} characters.
Extract and compress: disclosing party, receiving party, confidential info type,
key obligations, exclusions, term length, governing law.

Document:
{content}

Output only the summary, one line, under {max_chars} characters."""

EXPERT_USER_TEMPLATE_PRIVACY = """Summarize this privacy policy in under {max_chars} characters.
Extract and compress: company name, data types collected, key user rights,
legal basis for processing, data retention, jurisdiction.

Document:
{content}

Output only the summary, one line, under {max_chars} characters."""

EXPERT_USER_TEMPLATE_GENERIC = """Summarize this legal contract in under {max_chars} characters.
Extract and compress: parties involved, contract type, core obligations,
term, governing law, key restrictions.

Document:
{content}

Output only the summary, one line, under {max_chars} characters."""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────

class LLMSummaryGenerator:
    """
    Drop-in replacement for the mock SummaryGenerator in rag_sac_implementation.py.
    Uses Groq (free) with Llama 3.1 8B to generate real abstractive summaries.

    The generated summary is prepended to every chunk from that document
    before embedding — this is the core SAC technique.
    """

    def __init__(
        self,
        summary_length: int = 150,
        use_expert_prompt: bool = False,
        api_key: Optional[str] = None,
        model: str = "llama-3.1-8b-instant",   # free on Groq, very fast
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        """
        Parameters
        ----------
        summary_length   : max characters in the generated summary
        use_expert_prompt: if True, uses doc-type-aware prompts (NDA / privacy / generic)
        api_key          : Groq API key; falls back to GROQ_API_KEY env var
        model            : Groq model name. Free options:
                             "llama-3.1-8b-instant"   – fastest, good quality
                             "llama-3.1-70b-versatile" – slower, better quality
                             "mixtral-8x7b-32768"      – good for long docs
        max_retries      : retry count on rate-limit or transient errors
        retry_delay      : seconds to wait between retries
        """
        self.summary_length = summary_length
        self.use_expert_prompt = use_expert_prompt
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Resolve API key
        resolved_key = api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No Groq API key found.\n"
                "  Option 1: export GROQ_API_KEY='gsk_...'\n"
                "  Option 2: LLMSummaryGenerator(api_key='gsk_...')\n"
                "  Get a free key at https://console.groq.com"
            )

        self.client = Groq(api_key=resolved_key)
        print(f"LLMSummaryGenerator ready  |  model: {model}  |  max_chars: {summary_length}")

    # ── public entry point (same interface as original SummaryGenerator) ──────

    def generate_summary(self, document: Document) -> str:
        """
        Main entry point — called by RAGRetriever for every document.
        Routes to generic or expert prompt based on self.use_expert_prompt.
        Falls back to extractive summary if the API call fails.
        """
        if self.use_expert_prompt:
            return self._generate_expert_summary(document)
        else:
            return self._generate_generic_summary(document)

    # ── kept for compatibility with original interface ────────────────────────

    def generate_generic_summary(self, document: Document) -> str:
        return self._generate_generic_summary(document)

    def generate_expert_summary(self, document: Document) -> str:
        return self._generate_expert_summary(document)

    # ── internal generators ───────────────────────────────────────────────────

    def _generate_generic_summary(self, document: Document) -> str:
        """Generic prompt — works for any document type."""
        # Truncate content sent to the LLM to avoid token limits
        content_snippet = document.content[:3000]

        system = GENERIC_SYSTEM_PROMPT.format(max_chars=self.summary_length)
        user = GENERIC_USER_PROMPT.format(
            max_chars=self.summary_length,
            content=content_snippet
        )
        return self._call_llm(system, user, document)

    def _generate_expert_summary(self, document: Document) -> str:
        """Expert prompt — picks the right template based on document type."""
        content_lower = document.content.lower()
        content_snippet = document.content[:3000]

        # Detect document type
        if "non-disclosure" in content_lower or " nda" in content_lower or \
           "confidential information" in content_lower:
            user_template = EXPERT_USER_TEMPLATE_NDA
        elif "privacy policy" in content_lower or "personal data" in content_lower or \
             "gdpr" in content_lower:
            user_template = EXPERT_USER_TEMPLATE_PRIVACY
        else:
            user_template = EXPERT_USER_TEMPLATE_GENERIC

        system = EXPERT_SYSTEM_PROMPT.format(max_chars=self.summary_length)
        user = user_template.format(
            max_chars=self.summary_length,
            content=content_snippet
        )
        return self._call_llm(system, user, document)

    # ── Groq API call with retry logic ───────────────────────────────────────

    def _call_llm(self, system: str, user: str, document: Document) -> str:
        """
        Calls Groq API. Retries on rate-limit errors.
        Falls back to extractive summary if all retries fail.
        """
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    temperature=0.2,       # low temp = more consistent, factual
                    max_tokens=100,        # 150 chars ≈ 40-50 tokens; 100 is safe
                    stop=["\n"],           # stop at first newline — keep it one line
                )

                summary = response.choices[0].message.content.strip()

                # Clean up: remove quotes, strip to max length
                summary = summary.strip('"').strip("'")
                summary = summary[: self.summary_length]

                print(f"  [{document.doc_id}] summary: {summary}")
                return summary

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                if "rate_limit" in error_str or "429" in error_str:
                    wait = self.retry_delay * attempt   # back-off: 2s, 4s, 6s
                    print(f"  Rate limited — waiting {wait}s (attempt {attempt}/{self.max_retries})")
                    time.sleep(wait)
                else:
                    # Non-rate-limit error — don't retry
                    print(f"  API error for {document.doc_id}: {e}")
                    break

        # All retries failed — fall back to simple extractive summary
        print(f"  Falling back to extractive summary for {document.doc_id}")
        return self._extractive_fallback(document)

    # ── fallback when API is unavailable ─────────────────────────────────────

    def _extractive_fallback(self, document: Document) -> str:
        """
        Simple extractive fallback — same logic as the original mock generator.
        Used when the LLM API is unavailable or all retries fail.
        """
        sentences = re.split(r"[.!?]+", document.content.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        summary = " ".join(sentences[:2])[: self.summary_length]
        if document.doc_type:
            summary = f"{document.doc_type}: {summary}"
        return summary.strip()


# ─────────────────────────────────────────────────────────────────────────────
# PATCH FUNCTION  —  wires LLMSummaryGenerator into existing RAGRetriever
# ─────────────────────────────────────────────────────────────────────────────

def patch_retriever_with_llm_summaries(
    retriever,
    api_key: Optional[str] = None,
    model: str = "llama-3.1-8b-instant",
):
    """
    Patches an existing RAGRetriever instance to use LLMSummaryGenerator.

    Usage:
        retriever = RAGRetriever(use_sac=True, ...)
        patch_retriever_with_llm_summaries(retriever, api_key="gsk_...")
        retriever.index_documents(documents)
    """
    if not retriever.use_sac:
        print("Warning: retriever.use_sac=False — SAC is disabled, summaries won't be used.")
        return

    retriever.summary_generator = LLMSummaryGenerator(
        summary_length=retriever.summary_length,
        use_expert_prompt=retriever.use_expert_summary,
        api_key=api_key,
        model=model,
    )
    print(f"Retriever patched with LLMSummaryGenerator (model={model})")


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from rag_sac_implementation import (
        RAGRetriever,
        RAGEvaluator,
        create_sample_legal_dataset,
    )

    print("=" * 70)
    print("SAC with REAL LLM summaries (Groq / Llama 3)")
    print("=" * 70)

    # ── 1. Load sample dataset ──────────────────────────────────────────────
    documents, queries = create_sample_legal_dataset()
    print(f"\nLoaded {len(documents)} documents, {len(queries)} queries\n")

    # ── 2. Build retriever with SAC ─────────────────────────────────────────
    retriever = RAGRetriever(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        use_sac=True,
        use_expert_summary=False,
        summary_length=150,
        chunk_size=500,
    )

    # ── 3. Patch in the real LLM summary generator ─────────────────────────
    #    API key is read from GROQ_API_KEY env var automatically
    patch_retriever_with_llm_summaries(retriever)

    # ── 4. Index — this is where LLM summaries are generated ───────────────
    print("\nIndexing documents (LLM will be called once per document)...\n")
    retriever.index_documents(documents)

    # ── 5. Run a test query ─────────────────────────────────────────────────
    test_query = (
        "Consider Evelozcity's Non-Disclosure Agreement; "
        "can the recipient independently develop similar information?"
    )
    print(f"\nQuery: {test_query}\n")

    results = retriever.retrieve(test_query, top_k=3)

    print("Top 3 retrieved chunks:")
    print("-" * 70)
    for i, r in enumerate(results, 1):
        print(f"\n[{i}]  score={r.score:.4f}  doc={r.chunk.doc_id}")
        print(f"     summary used: {r.chunk.summary}")
        print(f"     chunk text:   {r.chunk.content[:120].strip()}...")

    # ── 6. Evaluate ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Evaluation across all queries")
    print("=" * 70)
    evaluator = RAGEvaluator()
    eval_results = evaluator.evaluate_dataset(queries, retriever, top_k_values=[1, 2, 4, 8])

    print(f"\n{'k':<6} {'DRM %':<12} {'Precision':<14} {'Recall'}")
    print("-" * 46)
    for k, metrics in eval_results.items():
        print(f"{k:<6} {metrics['drm']:>8.2f}%   {metrics['precision']:>10.4f}   {metrics['recall']:.4f}")
