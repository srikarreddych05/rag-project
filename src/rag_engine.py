"""
rag_engine.py -- Core RAG engine for Scholar Stream.

Provides RAGEngine, a single importable class that handles:
  1. Hybrid retrieval  (dense FAISS + sparse BM25, weighted by alpha)
  2. LLM generation   (Groq Llama, Anthropic Claude, or OpenAI GPT with cited answers)
  3. Grounding check  (token-overlap proxy for hallucination detection)

All methods are individually callable for testing and ablation.
API key priority: GROQ_API_KEY > ANTHROPIC_API_KEY > OPENAI_API_KEY
If no key is present, generation is skipped gracefully -- no crash.

Usage:
    engine = RAGEngine(index_path, chunks_path, model_name)
    result = engine.ask("What accuracy does CoT achieve on GSM8K?", top_k=5, alpha=0.5)
"""

import os
import re
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# ── Reproducibility ────────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL_NAME   = "all-mpnet-base-v2"
DEFAULT_TOP_K        = 5
DEFAULT_ALPHA        = 0.5
GENERATION_MODEL     = "llama-3.3-70b-versatile"
MAX_CONTEXT_CHARS    = 8000
GROUNDING_NGRAM_N    = 2

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler()],
)


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class ChunkResult:
    """A single retrieved chunk with its metadata and retrieval scores."""
    chunk_id:    str
    source:      str
    topic:       str
    text:        str
    score:       float
    dense_score: float
    bm25_score:  float
    rank:        int
    title:       str  = ""
    authors:     str  = ""
    published:   str  = ""
    arxiv_id:    str  = ""
    char_start:  int  = 0


@dataclass
class GenerationResult:
    """LLM-generated answer with metadata."""
    answer:          str
    model:           str
    prompt_tokens:   int   = 0
    answer_tokens:   int   = 0
    latency_ms:      float = 0.0
    fallback:        bool  = False


@dataclass
class FullResult:
    """Complete RAG result: retrieval + generation + grounding."""
    query:           str
    chunks:          list
    generation:      GenerationResult
    grounding_score: float
    retrieval_ms:    float
    total_ms:        float


# ── RAGEngine ──────────────────────────────────────────────────────────────────

class RAGEngine:
    """
    Production RAG engine combining hybrid retrieval and LLM generation.

    Parameters
    ----------
    index_path  : path to FAISS index file (.faiss)
    chunks_path : path to chunks JSON file (chunks.json)
    model_name  : SentenceTransformer model name
    use_bm25    : whether to enable BM25 sparse retrieval
    """

    def __init__(
        self,
        index_path:  str,
        chunks_path: str,
        model_name:  str  = DEFAULT_MODEL_NAME,
        use_bm25:    bool = True,
    ):
        self.model_name = model_name
        self.use_bm25   = use_bm25
        self._load_index(index_path)
        self._load_chunks(chunks_path)
        self._load_embed_model()
        if use_bm25:
            self._build_bm25()
        self._load_llm_client()
        log.info("RAGEngine ready -- %d chunks, bm25=%s", len(self.chunks), use_bm25)

    # ── Loading ────────────────────────────────────────────────────────────────

    def _load_index(self, index_path: str) -> None:
        """Load FAISS index from disk."""
        try:
            self.index = faiss.read_index(str(index_path))
            log.info("FAISS index loaded: %d vectors, dim=%d",
                     self.index.ntotal, self.index.d)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load FAISS index from {index_path}: {exc}"
            ) from exc

    def _load_chunks(self, chunks_path: str) -> None:
        """Load chunk metadata from chunks.json."""
        try:
            with open(chunks_path, encoding="utf-8") as fh:
                self.chunks: list[dict] = json.load(fh)
            log.info("Chunks loaded: %d entries", len(self.chunks))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load chunks from {chunks_path}: {exc}"
            ) from exc

    def _load_embed_model(self) -> None:
        """Load SentenceTransformer embedding model."""
        log.info("Loading embedding model: %s", self.model_name)
        self.embed_model = SentenceTransformer(self.model_name)

    def _build_bm25(self) -> None:
        """Tokenise corpus and build BM25Okapi index."""
        log.info("Building BM25 index over %d chunks ...", len(self.chunks))
        tokenised  = [c["text"].lower().split() for c in self.chunks]
        self.bm25  = BM25Okapi(tokenised)
        log.info("BM25 index ready")

    def _load_llm_client(self) -> None:
        """
        Initialise LLM client.
        Priority: GROQ_API_KEY > ANTHROPIC_API_KEY > OPENAI_API_KEY
        Falls back gracefully if no key is found.
        """
        self.llm_provider = None
        self.llm_client   = None
        self.llm_model    = GENERATION_MODEL

        # ── 1. Groq (free tier, highest priority) ─────────────────────────────
        groq_key = os.environ.get("GROQ_API_KEY", "").strip()
        if groq_key:
            try:
                from groq import Groq
                self.llm_client   = Groq(api_key=groq_key)
                self.llm_provider = "groq"
                self.llm_model    = os.environ.get(
                    "GROQ_MODEL", GENERATION_MODEL
                )
                log.info("Groq client initialised (model=%s)", self.llm_model)
                return
            except Exception as exc:
                log.warning("Groq SDK error: %s", exc)

        # ── 2. Anthropic ───────────────────────────────────────────────────────
        anthro_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if anthro_key and anthro_key != "your_key_here":
            try:
                import anthropic
                self.llm_client   = anthropic.Anthropic(api_key=anthro_key)
                self.llm_provider = "anthropic"
                self.llm_model    = "claude-sonnet-4-5"
                log.info("Anthropic client initialised (model=%s)", self.llm_model)
                return
            except Exception as exc:
                log.warning("Anthropic SDK error: %s", exc)

        # ── 3. OpenAI ──────────────────────────────────────────────────────────
        openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if openai_key:
            try:
                import openai
                self.llm_client   = openai.OpenAI(api_key=openai_key)
                self.llm_provider = "openai"
                self.llm_model    = "gpt-4o-mini"
                log.info("OpenAI client initialised (model=%s)", self.llm_model)
                return
            except Exception as exc:
                log.warning("OpenAI SDK error: %s", exc)

        log.warning(
            "No LLM API key found "
            "(checked GROQ_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY) "
            "-- generation disabled"
        )

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def _dense_scores(self, query: str, top_k: int) -> dict:
        """
        Run FAISS dense retrieval.

        Returns dict mapping chunk index -> normalised cosine score [0, 1].
        """
        q_emb = self.embed_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        k = min(top_k * 4, self.index.ntotal)
        scores, indices = self.index.search(q_emb, k)
        result = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                result[int(idx)] = float((score + 1.0) / 2.0)
        return result

    def _bm25_scores(self, query: str, top_k: int) -> dict:
        """
        Run BM25 sparse retrieval.

        Returns dict mapping chunk index -> normalised BM25 score [0, 1].
        """
        if not self.use_bm25:
            return {}
        tokens     = query.lower().split()
        raw_scores = self.bm25.get_scores(tokens)
        max_score  = float(np.max(raw_scores)) if np.max(raw_scores) > 0 else 1.0
        top_idx    = np.argsort(raw_scores)[::-1][: top_k * 4]
        return {
            int(i): float(raw_scores[i] / max_score)
            for i in top_idx
            if raw_scores[i] > 0
        }

    def retrieve(
        self,
        query: str,
        top_k: int   = DEFAULT_TOP_K,
        alpha: float = DEFAULT_ALPHA,
    ) -> list:
        """
        Hybrid retrieval combining dense FAISS and sparse BM25.

        Score formula:
            hybrid_score = alpha * dense_score + (1 - alpha) * bm25_score

        Parameters
        ----------
        query  : natural language question
        top_k  : number of results to return
        alpha  : weight for dense retrieval (0.0=BM25 only, 1.0=FAISS only)

        Returns
        -------
        List of ChunkResult sorted by hybrid score descending.
        """
        dense = self._dense_scores(query, top_k)
        bm25  = self._bm25_scores(query, top_k) if self.use_bm25 else {}

        all_indices = set(dense.keys()) | set(bm25.keys())

        scored = []
        for idx in all_indices:
            d_score = dense.get(idx, 0.0)
            b_score = bm25.get(idx, 0.0)
            hybrid  = alpha * d_score + (1.0 - alpha) * b_score
            scored.append((hybrid, idx))

        scored.sort(reverse=True)
        top = scored[:top_k]

        results = []
        for rank, (hybrid_score, idx) in enumerate(top, 1):
            c = self.chunks[idx]
            results.append(ChunkResult(
                chunk_id    = c.get("chunk_id",  f"chunk_{idx}"),
                source      = c.get("source",    ""),
                topic       = c.get("topic",     ""),
                text        = c.get("text",      ""),
                score       = round(hybrid_score,        4),
                dense_score = round(dense.get(idx, 0.0), 4),
                bm25_score  = round(bm25.get(idx,  0.0), 4),
                rank        = rank,
                title       = c.get("title",     c.get("source", "")),
                authors     = c.get("authors",   ""),
                published   = c.get("published", ""),
                arxiv_id    = c.get("arxiv_id",  c.get("source", "")),
                char_start  = c.get("char_start", 0),
            ))
        return results

    # ── Generation ─────────────────────────────────────────────────────────────

    def _build_prompt(self, query: str, chunks: list) -> str:
        """
        Build the generation prompt from query and retrieved chunks.

        Context is capped at MAX_CONTEXT_CHARS to stay within token limits.
        """
        context_parts = []
        total_chars   = 0
        for chunk in chunks:
            label   = f"[Source: {chunk.arxiv_id or chunk.source}, Rank {chunk.rank}]"
            excerpt = chunk.text[:1200]
            block   = f"{label}\n{excerpt}"
            if total_chars + len(block) > MAX_CONTEXT_CHARS:
                break
            context_parts.append(block)
            total_chars += len(block)

        context = "\n\n---\n\n".join(context_parts)
        return (
            "You are a precise academic research assistant. "
            "Answer the question using ONLY the provided source excerpts. "
            "Cite sources inline using [Source: ...] notation. "
            "If the excerpts do not contain enough information, say so explicitly.\n\n"
            f"QUESTION: {query}\n\n"
            f"SOURCE EXCERPTS:\n{context}\n\n"
            "ANSWER (cite sources inline):"
        )

    def generate(
        self,
        query:  str,
        chunks: list,
        model:  str = "",
    ) -> GenerationResult:
        """
        Generate a cited answer using the configured LLM provider.

        Falls back gracefully if no API key is configured or the call fails.

        Parameters
        ----------
        query  : original user question
        chunks : retrieved chunks to use as context
        model  : model override (uses self.llm_model by default)

        Returns
        -------
        GenerationResult with answer text and usage metadata.
        """
        if self.llm_client is None or not chunks:
            fallback_text = (
                "[Generation unavailable -- no API key configured]\n\n"
                f"Top retrieved excerpt (score={chunks[0].score:.3f}):\n"
                f"{chunks[0].text[:500]}..."
            ) if chunks else "[No results retrieved]"
            return GenerationResult(
                answer   = fallback_text,
                model    = "fallback",
                fallback = True,
            )

        prompt  = self._build_prompt(query, chunks)
        t_start = time.perf_counter()
        model   = model or self.llm_model

        try:
            # ── Groq ───────────────────────────────────────────────────────────
            if self.llm_provider == "groq":
                response = self.llm_client.chat.completions.create(
                    model    = model,
                    max_tokens = 1024,
                    messages = [{"role": "user", "content": prompt}],
                )
                latency_ms = (time.perf_counter() - t_start) * 1000
                answer     = response.choices[0].message.content
                prompt_tok = response.usage.prompt_tokens     if response.usage else 0
                answer_tok = response.usage.completion_tokens if response.usage else 0

            # ── Anthropic ──────────────────────────────────────────────────────
            elif self.llm_provider == "anthropic":
                response = self.llm_client.messages.create(
                    model      = model,
                    max_tokens = 1024,
                    messages   = [{"role": "user", "content": prompt}],
                )
                latency_ms = (time.perf_counter() - t_start) * 1000
                answer     = response.content[0].text
                prompt_tok = response.usage.input_tokens
                answer_tok = response.usage.output_tokens

            # ── OpenAI ─────────────────────────────────────────────────────────
            else:
                response = self.llm_client.chat.completions.create(
                    model      = model,
                    max_tokens = 1024,
                    messages   = [{"role": "user", "content": prompt}],
                )
                latency_ms = (time.perf_counter() - t_start) * 1000
                answer     = response.choices[0].message.content
                prompt_tok = response.usage.prompt_tokens     if response.usage else 0
                answer_tok = response.usage.completion_tokens if response.usage else 0

            return GenerationResult(
                answer        = answer,
                model         = model,
                prompt_tokens = prompt_tok,
                answer_tokens = answer_tok,
                latency_ms    = round(latency_ms, 1),
                fallback      = False,
            )

        except Exception as exc:
            log.error("Generation failed (%s): %s", self.llm_provider, exc)
            return GenerationResult(
                answer   = (
                    f"[Generation error: {exc}]\n\n"
                    f"Top retrieved excerpt:\n{chunks[0].text[:300]}..."
                ),
                model    = model,
                fallback = True,
            )

    # ── Grounding check ────────────────────────────────────────────────────────

    def grounding_check(self, answer: str, chunks: list) -> float:
        """
        Estimate how grounded the generated answer is in the retrieved chunks.

        Method: bigram overlap between answer tokens and combined chunk text.
        Score 1.0 = every bigram in the answer appears in the context.
        Score 0.0 = no overlap (possible hallucination).

        Parameters
        ----------
        answer : generated answer text
        chunks : retrieved chunks used as generation context

        Returns
        -------
        Grounding score in [0.0, 1.0].
        """
        if not answer or not chunks:
            return 0.0

        def ngrams(text: str, n: int) -> set:
            tokens = re.sub(r"[^a-z0-9 ]", "", text.lower()).split()
            return {
                " ".join(tokens[i: i + n])
                for i in range(len(tokens) - n + 1)
            }

        context_text   = " ".join(c.text for c in chunks)
        answer_ngrams  = ngrams(answer, GROUNDING_NGRAM_N)
        context_ngrams = ngrams(context_text, GROUNDING_NGRAM_N)

        if not answer_ngrams:
            return 0.0

        overlap = answer_ngrams & context_ngrams
        return round(len(overlap) / len(answer_ngrams), 3)

    # ── Full pipeline ──────────────────────────────────────────────────────────

    def ask(
        self,
        query: str,
        top_k: int   = DEFAULT_TOP_K,
        alpha: float = DEFAULT_ALPHA,
        model: str   = "",
    ) -> FullResult:
        """
        Full RAG pipeline: retrieve -> generate -> grounding check.

        Parameters
        ----------
        query  : user question
        top_k  : number of chunks to retrieve
        alpha  : hybrid retrieval weight (0=BM25 only, 1=FAISS only)
        model  : LLM model override

        Returns
        -------
        FullResult containing chunks, answer, grounding score, and timings.
        """
        t0 = time.perf_counter()

        # Step 1: Retrieve
        chunks       = self.retrieve(query, top_k=top_k, alpha=alpha)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        # Step 2: Generate
        generation = self.generate(query, chunks, model=model or self.llm_model)

        # Step 3: Grounding check
        grounding = self.grounding_check(generation.answer, chunks)

        total_ms = (time.perf_counter() - t0) * 1000

        return FullResult(
            query           = query,
            chunks          = chunks,
            generation      = generation,
            grounding_score = grounding,
            retrieval_ms    = round(retrieval_ms, 1),
            total_ms        = round(total_ms, 1),
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def n_chunks(self) -> int:
        """Total number of indexed chunks."""
        return len(self.chunks)

    @property
    def n_vectors(self) -> int:
        """Total vectors in the FAISS index."""
        return self.index.ntotal