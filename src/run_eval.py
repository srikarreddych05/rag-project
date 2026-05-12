"""
Step 4 — Quantitative evaluation of retrieval quality.

Usage:
    # Evaluate default system
    python src/run_eval.py

    # Evaluate a specific configuration
    python src/run_eval.py --model minilm --chunk small

    # Run full ablation grid (all chunk×model combos)
    python src/run_eval.py --ablation

Metrics computed:
    Precision@K  — fraction of top-K chunks that are relevant
    Recall@K     — fraction of relevant chunks found in top-K
    MRR          — Mean Reciprocal Rank (how high the first correct chunk ranks)
    Hit@K        — fraction of queries where any correct chunk is in top-K

A 'relevant' chunk is one whose source filename matches the QA pair's 'source' field.
This is document-level relevance: we check whether retrieval correctly identifies
which paper answers each question. This is a conservative, verifiable metric.

Results written to:
    results/eval_<model>_<chunk>.json   — per-query detail
    results/ablation_summary.json       — cross-config comparison table
"""

import argparse
import json
import logging
import random
import numpy as np
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (EVAL_DIR, RESULTS_DIR, DEFAULT_EMBED_MODEL,
                    DEFAULT_CHUNK, EVAL_K_VALUES, CHUNK_CONFIGS,
                    EMBEDDING_MODELS, SEED)
from retriever import Retriever

random.seed(SEED)
np.random.seed(SEED)

LOG_FILE = RESULTS_DIR / "eval.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# ── Metric functions ───────────────────────────────────────────────────────────

def precision_at_k(retrieved_sources: list[str], gold_source: str, k: int) -> float:
    """Fraction of top-K retrieved chunks from the correct source."""
    top_k = retrieved_sources[:k]
    hits  = sum(1 for s in top_k if gold_source in s)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved_sources: list[str], gold_source: str, k: int,
                total_relevant: int = 1) -> float:
    """
    Fraction of relevant chunks found in top-K.
    We treat total_relevant=1 (document-level: at least one chunk from the
    correct source). This can be extended with chunk-level labels on Day 6.
    """
    top_k = retrieved_sources[:k]
    hits  = min(sum(1 for s in top_k if gold_source in s), total_relevant)
    return hits / total_relevant if total_relevant > 0 else 0.0


def reciprocal_rank(retrieved_sources: list[str], gold_source: str) -> float:
    """1/rank of the first correct chunk. 0 if none found."""
    for i, src in enumerate(retrieved_sources, 1):
        if gold_source in src:
            return 1.0 / i
    return 0.0


def hit_at_k(retrieved_sources: list[str], gold_source: str, k: int) -> int:
    """1 if any of the top-K chunks comes from the correct source, else 0."""
    return int(any(gold_source in s for s in retrieved_sources[:k]))


# ── Evaluation runner ──────────────────────────────────────────────────────────

def evaluate(model_key: str, chunk_key: str,
             qa_path: Optional[Path] = None) -> dict:
    """
    Run full evaluation for one model+chunk configuration.
    Returns summary dict with all metrics.
    """
    qa_path = qa_path or EVAL_DIR / "test_qa.json"
    if not qa_path.exists():
        log.error(f"QA file not found: {qa_path}")
        return {}

    with open(qa_path) as f:
        qa_pairs = json.load(f)

    # Filter out template entries (those not yet filled in)
    qa_pairs = [q for q in qa_pairs if "FILL_IN" not in q.get("answer", "")]
    if not qa_pairs:
        log.error("No completed QA pairs found — fill in eval/test_qa.json first.")
        return {}

    log.info(f"\n[eval] model={model_key}  chunk={chunk_key}  "
             f"n_qa={len(qa_pairs)}")

    retriever = Retriever(model_key=model_key, chunk_key=chunk_key)
    log.info(f"[eval] Index: {retriever.n_chunks} chunks")

    # Collect per-query results
    per_query = []
    all_rr = []
    metrics_by_k = {k: {"precision": [], "recall": [], "hit": []}
                    for k in EVAL_K_VALUES}

    for qa in qa_pairs:
        question    = qa["question"]
        gold_source = Path(qa["source"]).stem   # match on filename stem

        results = retriever.retrieve(question, top_k=max(EVAL_K_VALUES))
        retrieved_sources = [r.source for r in results]

        rr = reciprocal_rank(retrieved_sources, gold_source)
        all_rr.append(rr)

        k_metrics = {}
        for k in EVAL_K_VALUES:
            p = precision_at_k(retrieved_sources, gold_source, k)
            r = recall_at_k(retrieved_sources, gold_source, k)
            h = hit_at_k(retrieved_sources, gold_source, k)
            metrics_by_k[k]["precision"].append(p)
            metrics_by_k[k]["recall"].append(r)
            metrics_by_k[k]["hit"].append(h)
            k_metrics[k] = {"precision": round(p, 4), "recall": round(r, 4), "hit": h}

        per_query.append({
            "question"         : question,
            "gold_source"      : gold_source,
            "retrieved_sources": retrieved_sources,
            "reciprocal_rank"  : round(rr, 4),
            "metrics_by_k"     : k_metrics,
            "difficulty"       : qa.get("difficulty", "unknown"),
        })

        log.info(f"  Q: {question[:60]}...  RR={rr:.3f}  Hit@5={k_metrics[5]['hit']}")

    # Aggregate
    summary = {
        "model_key"  : model_key,
        "chunk_key"  : chunk_key,
        "n_qa"       : len(qa_pairs),
        "n_chunks"   : retriever.n_chunks,
        "MRR"        : round(float(np.mean(all_rr)), 4),
    }
    for k in EVAL_K_VALUES:
        summary[f"P@{k}"]    = round(float(np.mean(metrics_by_k[k]["precision"])), 4)
        summary[f"R@{k}"]    = round(float(np.mean(metrics_by_k[k]["recall"])),    4)
        summary[f"Hit@{k}"]  = round(float(np.mean(metrics_by_k[k]["hit"])),       4)

    log.info(f"\n[eval] Results for {model_key}_{chunk_key}:")
    log.info(f"  MRR={summary['MRR']}  "
             f"P@1={summary['P@1']}  P@5={summary['P@5']}  "
             f"Hit@5={summary['Hit@5']}")

    # Save per-query details
    out = {"summary": summary, "per_query": per_query}
    out_path = RESULTS_DIR / f"eval_{model_key}_{chunk_key}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info(f"[eval] Detailed results → {out_path}")

    return summary


# ── Ablation grid ──────────────────────────────────────────────────────────────

def run_ablation():
    """Evaluate all chunk×model combinations and write a comparison table."""
    summaries = []
    for model_key in EMBEDDING_MODELS:
        for chunk_key in CHUNK_CONFIGS:
            try:
                s = evaluate(model_key, chunk_key)
                if s:
                    summaries.append(s)
            except FileNotFoundError as e:
                log.warning(f"  Skipping {model_key}_{chunk_key}: {e}")

    out_path = RESULTS_DIR / "ablation_summary.json"
    with open(out_path, "w") as f:
        json.dump(summaries, f, indent=2)
    log.info(f"\n[ablation] Summary → {out_path}")

    # Print ASCII table for quick inspection
    if summaries:
        log.info("\n  Config            MRR    P@1    P@5    Hit@5")
        log.info("  " + "-" * 50)
        for s in sorted(summaries, key=lambda x: -x["MRR"]):
            cfg = f"{s['model_key']}_{s['chunk_key']}"
            log.info(f"  {cfg:<18}  {s['MRR']:.3f}  {s['P@1']:.3f}  "
                     f"{s['P@5']:.3f}  {s['Hit@5']:.3f}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL,
                        choices=list(EMBEDDING_MODELS.keys()))
    parser.add_argument("--chunk", default=DEFAULT_CHUNK,
                        choices=list(CHUNK_CONFIGS.keys()))
    parser.add_argument("--ablation", action="store_true",
                        help="Evaluate all chunk×model combinations")
    args = parser.parse_args()

    if args.ablation:
        run_ablation()
    else:
        evaluate(args.model, args.chunk)


if __name__ == "__main__":
    main()
