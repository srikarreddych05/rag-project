"""
Step 4 — Query the RAG system.

Usage:
    python src/query_rag.py --query "What accuracy does CoT achieve on GSM8K?"
    python src/query_rag.py --query "..." --top_k 5

All queries and results are logged to results/query_log.jsonl.
Each logged entry includes: query, top-k chunks, sources, timestamp.

The retrieval step is a genuine pipeline step — the LLM call
(generation step) will be added in Days 3–4.  Milestone 1 requires
the retrieval to be working, which this script demonstrates.
"""

import json
import random
import argparse
import datetime
import numpy as np
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

# ── Reproducibility ───────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K_DEFAULT = 3

ROOT       = Path(__file__).parent.parent
INDEX_PATH = ROOT / "data" / "index.faiss"
CHUNKS_PATH= ROOT / "data" / "chunks.json"
LOG_PATH   = ROOT / "results" / "query_log.jsonl"

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_artifacts():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            "FAISS index not found. Run src/build_index.py first."
        )
    index  = faiss.read_index(str(INDEX_PATH))
    chunks = json.loads(CHUNKS_PATH.read_text())
    return index, chunks


def retrieve(query: str, model, index, chunks, top_k: int) -> list[dict]:
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(q_emb, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx].copy()
        chunk["l2_distance"] = float(dist)
        results.append(chunk)
    return results


def pretty_print(query: str, results: list[dict]):
    print("\n" + "═" * 70)
    print(f"  QUERY: {query}")
    print("═" * 70)
    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] Source : {r['source']}")
        print(f"       ChunkID: {r['chunk_id']}")
        print(f"       L2 dist: {r['l2_distance']:.4f}")
        print(f"       Preview: {r['text'][:300].replace(chr(10), ' ')}...")
    print()


def main():
    parser = argparse.ArgumentParser(description="Query the academic RAG system")
    parser.add_argument("--query", type=str, required=True, help="Your question")
    parser.add_argument("--top_k", type=int, default=TOP_K_DEFAULT,
                        help=f"Number of chunks to retrieve (default: {TOP_K_DEFAULT})")
    args = parser.parse_args()

    print(f"[query_rag] Loading model '{EMBED_MODEL}' ...")
    model = SentenceTransformer(EMBED_MODEL)

    print("[query_rag] Loading index and chunks ...")
    index, chunks = load_artifacts()
    print(f"[query_rag] Index has {index.ntotal} vectors | {len(chunks)} chunks loaded")

    results = retrieve(args.query, model, index, chunks, args.top_k)
    pretty_print(args.query, results)

    # Log to JSONL
    entry = {
        "timestamp"   : datetime.datetime.utcnow().isoformat(),
        "query"       : args.query,
        "top_k"       : args.top_k,
        "results"     : [
            {"chunk_id": r["chunk_id"], "source": r["source"],
             "l2_distance": r["l2_distance"], "preview": r["text"][:200]}
            for r in results
        ],
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[query_rag] Result logged → {LOG_PATH}")


if __name__ == "__main__":
    main()