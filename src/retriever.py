"""
retriever.py — Retriever class shared by query_rag.py and run_eval.py.

Encapsulates: loading the FAISS index + chunks, embedding a query,
and returning ranked results with scores. Keeping this in one class
means both the demo script and the eval script use identical retrieval
logic — no silent inconsistency between what you demo and what you measure.
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass

import faiss
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (INDEX_DIR, DEFAULT_EMBED_MODEL, DEFAULT_CHUNK,
                    EMBEDDING_MODELS, SEED)


@dataclass
class RetrievalResult:
    chunk_id   : str
    source     : str
    topic      : str
    text       : str
    score      : float     # cosine similarity (higher = better)
    char_start : int


class Retriever:
    """
    Loads a FAISS index built by build_index.py and answers queries.

    Parameters
    ----------
    model_key  : key into EMBEDDING_MODELS dict  (e.g. 'mpnet')
    chunk_key  : key into CHUNK_CONFIGS dict      (e.g. 'medium')
    """

    def __init__(self, model_key: str = DEFAULT_EMBED_MODEL,
                 chunk_key: str = DEFAULT_CHUNK):
        self.model_key = model_key
        self.chunk_key = chunk_key
        self.run_name  = f"{model_key}_{chunk_key}"
        self._load()

    def _load(self):
        run_dir = INDEX_DIR / self.run_name
        if not run_dir.exists():
            raise FileNotFoundError(
                f"Index not found at {run_dir}. "
                f"Run: python src/build_index.py --model {self.model_key} --chunk {self.chunk_key}"
            )

        self.index  = faiss.read_index(str(run_dir / "index.faiss"))
        with open(run_dir / "chunks.json") as f:
            self.chunks = json.load(f)
        with open(run_dir / "build_meta.json") as f:
            self.meta = json.load(f)

        model_name  = EMBEDDING_MODELS[self.model_key]
        self.model  = SentenceTransformer(model_name)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Embed query and return top-k chunks by cosine similarity."""
        q_emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(q_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            c = self.chunks[idx]
            results.append(RetrievalResult(
                chunk_id   = c["chunk_id"],
                source     = c["source"],
                topic      = c["topic"],
                text       = c["text"],
                score      = float(score),
                char_start = c.get("char_start", 0),
            ))
        return results

    @property
    def n_chunks(self) -> int:
        return self.index.ntotal

    @property
    def n_vectors(self) -> int:
        return self.index.ntotal

    def __repr__(self):
        return (f"Retriever(model={self.model_key}, chunk={self.chunk_key}, "
                f"n_chunks={self.n_chunks})")
