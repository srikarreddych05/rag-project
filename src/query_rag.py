"""
Step 5 — Interactive RAG query for demos and live evaluation.

Usage:
    python src/query_rag.py --query "What accuracy does CoT achieve on GSM8K?"
    python src/query_rag.py --query "..." --top_k 5 --model mpnet --chunk medium
    python src/query_rag.py --interactive   # REPL mode for live demo

All queries are appended to results/query_log.jsonl for audit trail.
This is the script you run during the Milestone Demo — the graders will
ask you to modify this code on the spot. Key modification points are
clearly marked with # MODIFY: comments.
"""

import argparse
import datetime
import json
import random
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (RESULTS_DIR, DEFAULT_EMBED_MODEL, DEFAULT_CHUNK,
                    DEFAULT_TOP_K, EMBEDDING_MODELS, CHUNK_CONFIGS, SEED)
from retriever import Retriever

random.seed(SEED)
np.random.seed(SEED)

LOG_PATH = RESULTS_DIR / "query_log.jsonl"


def pretty_print(query: str, results, model_key: str, chunk_key: str):
    print("\n" + "═" * 72)
    print(f"  Query  : {query}")
    print(f"  Config : model={model_key}  chunk={chunk_key}")
    print("═" * 72)

    for i, r in enumerate(results, 1):
        # MODIFY: change this threshold to demo score filtering
        relevance = "HIGH" if r.score > 0.6 else ("MED" if r.score > 0.4 else "LOW")
        print(f"\n  [{i}] {relevance}  score={r.score:.4f}")
        print(f"       Source : {r.source}")
        print(f"       Topic  : {r.topic}")
        print(f"       Preview: {r.text[:250].replace(chr(10), ' ')}...")

    print()


def log_query(query: str, results, model_key: str, chunk_key: str, top_k: int):
    entry = {
        "timestamp" : datetime.datetime.utcnow().isoformat(),
        "query"     : query,
        "model"     : model_key,
        "chunk"     : chunk_key,
        "top_k"     : top_k,
        "results"   : [
            {
                "rank"    : i + 1,
                "chunk_id": r.chunk_id,
                "source"  : r.source,
                "topic"   : r.topic,
                "score"   : round(r.score, 4),
                "preview" : r.text[:200],
            }
            for i, r in enumerate(results)
        ],
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def run_query(retriever: Retriever, query: str, top_k: int,
              model_key: str, chunk_key: str):
    results = retriever.retrieve(query, top_k=top_k)
    pretty_print(query, results, model_key, chunk_key)
    log_query(query, results, model_key, chunk_key, top_k)
    print(f"  [logged → {LOG_PATH}]")


def main():
    parser = argparse.ArgumentParser(description="Query the Academic RAG system")
    parser.add_argument("--query",       type=str,  help="Question to answer")
    parser.add_argument("--top_k",       type=int,  default=DEFAULT_TOP_K)
    parser.add_argument("--model",       default=DEFAULT_EMBED_MODEL,
                        choices=list(EMBEDDING_MODELS.keys()))
    parser.add_argument("--chunk",       default=DEFAULT_CHUNK,
                        choices=list(CHUNK_CONFIGS.keys()))
    parser.add_argument("--interactive", action="store_true",
                        help="Start interactive REPL (good for demos)")
    args = parser.parse_args()

    print(f"[query_rag] Loading Retriever(model={args.model}, chunk={args.chunk}) ...")
    retriever = Retriever(model_key=args.model, chunk_key=args.chunk)
    print(f"[query_rag] Ready. Index has {retriever.n_chunks} chunks.\n")

    if args.interactive:
        print("Interactive mode — type your question (Ctrl-C to exit)\n")
        while True:
            try:
                # MODIFY: change the prompt or add query pre-processing here
                query = input("Query > ").strip()
                if not query:
                    continue
                run_query(retriever, query, args.top_k, args.model, args.chunk)
            except KeyboardInterrupt:
                print("\nExiting.")
                break
    elif args.query:
        run_query(retriever, args.query, args.top_k, args.model, args.chunk)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
