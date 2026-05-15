"""
evaluate.py -- Full evaluation pipeline for Scholar Stream RAG system.

Measures both retrieval quality and generation quality:

Retrieval metrics:
  - Precision@K  : fraction of top-K chunks from correct source
  - MRR          : Mean Reciprocal Rank

Generation metrics:
  - ROUGE-L      : longest common subsequence overlap vs gold answer
  - BERTScore F1 : semantic similarity vs gold answer
  - Grounding    : token overlap between answer and retrieved context

CLI:
    python src/evaluate.py --qa eval/test_qa.json --retriever hybrid --top_k 5
    python src/evaluate.py --run_all --out_dir results/

Output:
    results/eval_<condition>.json   -- per-query detail
    results/eval_summary.json       -- chart-ready comparison table
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    EVAL_DIR, RESULTS_DIR, INDEX_DIR,
    DEFAULT_EMBED_MODEL, DEFAULT_CHUNK,
    EVAL_K_VALUES, SEED,
)
from rag_engine import RAGEngine

# ── Reproducibility ────────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_QA_PATH  = EVAL_DIR / "test_qa.json"
DEFAULT_TOP_K    = 5

# ── Logging ────────────────────────────────────────────────────────────────────
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = RESULTS_DIR / "eval.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ── Metric helpers ─────────────────────────────────────────────────────────────

def precision_at_k(retrieved: list[str], gold: str, k: int) -> float:
    """
    Compute Precision@K for one query.

    Parameters
    ----------
    retrieved : list of source filenames for top-K retrieved chunks
    gold      : correct source filename stem
    k         : cutoff rank

    Returns
    -------
    Float in [0, 1].
    """
    top_k = retrieved[:k]
    hits  = sum(1 for s in top_k if gold in s)
    return hits / k if k > 0 else 0.0


def reciprocal_rank(retrieved: list[str], gold: str) -> float:
    """
    Compute Reciprocal Rank for one query.

    Parameters
    ----------
    retrieved : ranked list of source filenames
    gold      : correct source filename stem

    Returns
    -------
    1/rank of first correct result, or 0.0 if not found.
    """
    for i, src in enumerate(retrieved, 1):
        if gold in src:
            return 1.0 / i
    return 0.0


def hit_at_k(retrieved: list[str], gold: str, k: int) -> int:
    """
    Binary hit metric: 1 if correct source in top-K, else 0.

    Parameters
    ----------
    retrieved : ranked list of source filenames
    gold      : correct source filename stem
    k         : cutoff rank

    Returns
    -------
    1 or 0.
    """
    return int(any(gold in s for s in retrieved[:k]))


def compute_rouge_l(hypothesis: str, reference: str) -> float:
    """
    Compute ROUGE-L score between hypothesis and reference strings.

    Uses the rouge_score library if available, falls back to a simple
    LCS-based implementation otherwise.

    Parameters
    ----------
    hypothesis : generated answer
    reference  : gold answer

    Returns
    -------
    ROUGE-L F1 score in [0, 1].
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        return round(scores["rougeL"].fmeasure, 4)
    except Exception:
        # Simple fallback LCS
        h_tokens = hypothesis.lower().split()
        r_tokens = reference.lower().split()
        if not h_tokens or not r_tokens:
            return 0.0
        m, n = len(r_tokens), len(h_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if r_tokens[i - 1] == h_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs = dp[m][n]
        precision = lcs / n if n > 0 else 0.0
        recall    = lcs / m if m > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return round(2 * precision * recall / (precision + recall), 4)


def compute_bertscore(hypotheses: list[str], references: list[str]) -> list[float]:
    """
    Compute BERTScore F1 for a batch of hypothesis-reference pairs.

    Falls back to returning 0.0 for each pair if bert_score is unavailable
    or if no API key is set (BERTScore downloads a model).

    Parameters
    ----------
    hypotheses : list of generated answers
    references : list of gold answers

    Returns
    -------
    List of F1 scores in [0, 1].
    """
    try:
        from bert_score import score as bert_score_fn
        P, R, F = bert_score_fn(
            hypotheses, references,
            lang="en", verbose=False,
            model_type="distilbert-base-uncased",
        )
        return [round(float(f), 4) for f in F.tolist()]
    except Exception as exc:
        log.warning("BERTScore unavailable: %s -- using 0.0", exc)
        return [0.0] * len(hypotheses)


# ── Engine loader ──────────────────────────────────────────────────────────────

def load_engine(
    model_key:  str = DEFAULT_EMBED_MODEL,
    chunk_key:  str = DEFAULT_CHUNK,
    use_bm25:   bool = True,
) -> RAGEngine:
    """
    Load RAGEngine for a given model+chunk configuration.

    Parameters
    ----------
    model_key : embedding model key
    chunk_key : chunk size key
    use_bm25  : whether to enable BM25 retrieval

    Returns
    -------
    Initialised RAGEngine.
    """
    run_dir     = INDEX_DIR / f"{model_key}_{chunk_key}"
    index_path  = run_dir / "index.faiss"
    chunks_path = run_dir / "chunks.json"

    if not index_path.exists():
        raise FileNotFoundError(
            f"Index not found at {index_path}. "
            f"Run: python src/build_index.py --model {model_key} --chunk {chunk_key}"
        )

    from config import EMBEDDING_MODELS
    return RAGEngine(
        index_path  = str(index_path),
        chunks_path = str(chunks_path),
        model_name  = EMBEDDING_MODELS[model_key],
        use_bm25    = use_bm25,
    )


# ── Evaluation runner ──────────────────────────────────────────────────────────

def evaluate(
    engine:     RAGEngine,
    qa_path:    Path,
    top_k:      int,
    alpha:      float,
    label:      str,
    out_path:   Path,
) -> dict:
    """
    Run full evaluation (retrieval + generation) for one configuration.

    Parameters
    ----------
    engine   : initialised RAGEngine
    qa_path  : path to test_qa.json
    top_k    : number of chunks to retrieve per query
    alpha    : hybrid retrieval weight
    label    : condition label for summary table
    out_path : where to save per-query JSON results

    Returns
    -------
    Summary dict with all aggregated metrics.
    """
    try:
        with open(qa_path, encoding="utf-8") as fh:
            qa_pairs = json.load(fh)
    except Exception as exc:
        log.error("Cannot load QA file %s: %s", qa_path, exc)
        return {}

    # Filter out unfilled template entries
    qa_pairs = [q for q in qa_pairs if "FILL_IN" not in q.get("answer", "")]
    if not qa_pairs:
        log.error("No completed QA pairs found in %s", qa_path)
        return {}

    log.info("\n[evaluate] label=%s  n_qa=%d  top_k=%d  alpha=%.2f",
             label, len(qa_pairs), top_k, alpha)

    per_query    = []
    all_rr       = []
    all_rouge    = []
    hypotheses   = []
    references   = []
    groundings   = []

    metrics_by_k = {k: {"precision": [], "hit": []} for k in EVAL_K_VALUES}

    for qa in qa_pairs:
        question    = qa["question"]
        gold_answer = qa["answer"]
        gold_source = Path(qa["source"]).stem

        # Retrieve
        chunks           = engine.retrieve(question, top_k=top_k, alpha=alpha)
        retrieved_sources = [c.source for c in chunks]

        # Retrieval metrics
        rr = reciprocal_rank(retrieved_sources, gold_source)
        all_rr.append(rr)
        for k in EVAL_K_VALUES:
            metrics_by_k[k]["precision"].append(
                precision_at_k(retrieved_sources, gold_source, k))
            metrics_by_k[k]["hit"].append(
                hit_at_k(retrieved_sources, gold_source, k))

        # Generation
        gen = engine.generate(question, chunks)
        grounding = engine.grounding_check(gen.answer, chunks)
        groundings.append(grounding)

        # ROUGE-L
        rouge = compute_rouge_l(gen.answer, gold_answer)
        all_rouge.append(rouge)
        hypotheses.append(gen.answer)
        references.append(gold_answer)

        log.info("  Q: %s...  RR=%.3f  ROUGE-L=%.3f  grounding=%.3f",
                 question[:55], rr, rouge, grounding)

        per_query.append({
            "question"         : question,
            "gold_source"      : gold_source,
            "retrieved_sources": retrieved_sources,
            "reciprocal_rank"  : round(rr, 4),
            "rouge_l"          : round(rouge, 4),
            "grounding_score"  : round(grounding, 4),
            "generated_answer" : gen.answer[:300],
            "gold_answer"      : gold_answer[:300],
        })

    # BERTScore (batch for efficiency)
    bert_scores = compute_bertscore(hypotheses, references)

    # Aggregate
    summary = {
        "label"            : label,
        "n_qa"             : len(qa_pairs),
        "top_k"            : top_k,
        "alpha"            : alpha,
        "MRR"              : round(float(np.mean(all_rr)), 4),
        "ROUGE_L"          : round(float(np.mean(all_rouge)), 4),
        "BERTScore_F1"     : round(float(np.mean(bert_scores)), 4),
        "grounding_mean"   : round(float(np.mean(groundings)), 4),
    }
    for k in EVAL_K_VALUES:
        summary[f"P@{k}"]   = round(float(np.mean(metrics_by_k[k]["precision"])), 4)
        summary[f"Hit@{k}"] = round(float(np.mean(metrics_by_k[k]["hit"])),       4)

    log.info("\n[evaluate] %s: MRR=%.4f  ROUGE-L=%.4f  BERTScore=%.4f  grounding=%.4f",
             label, summary["MRR"], summary["ROUGE_L"],
             summary["BERTScore_F1"], summary["grounding_mean"])

    # Save per-query detail
    out = {"summary": summary, "per_query": per_query}
    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        log.info("[evaluate] Saved to %s", out_path)
    except Exception as exc:
        log.error("Could not save results: %s", exc)

    return summary


# ── Ablation runner ────────────────────────────────────────────────────────────

def run_all_conditions(qa_path: Path, out_dir: Path) -> None:
    """
    Evaluate 4 experimental conditions and write eval_summary.json.

    Conditions:
      1. BM25 only         (alpha=0.0, chunk=medium)
      2. FAISS chunk=small  (alpha=1.0, chunk=small)
      3. FAISS chunk=medium (alpha=1.0, chunk=medium)  <- default
      4. FAISS top_k=10    (alpha=1.0, chunk=medium, top_k=10)

    Output format matches dashboard.html Chart.js schema.

    Parameters
    ----------
    qa_path : path to test_qa.json
    out_dir : directory for all output files
    """
    conditions = [
        {"label": "BM25",        "chunk": "medium", "alpha": 0.0, "top_k": 5,  "bm25": True},
        {"label": "FAISS-256",   "chunk": "small",  "alpha": 1.0, "top_k": 5,  "bm25": False},
        {"label": "FAISS-512",   "chunk": "medium", "alpha": 1.0, "top_k": 5,  "bm25": False},
        {"label": "FAISS-K10",   "chunk": "medium", "alpha": 1.0, "top_k": 10, "bm25": False},
    ]

    summaries = []
    for cond in conditions:
        try:
            engine = load_engine(
                model_key = DEFAULT_EMBED_MODEL,
                chunk_key = cond["chunk"],
                use_bm25  = cond["bm25"],
            )
            out_path = out_dir / f"eval_{cond['label'].lower().replace('-', '_')}.json"
            summary  = evaluate(
                engine   = engine,
                qa_path  = qa_path,
                top_k    = cond["top_k"],
                alpha    = cond["alpha"],
                label    = cond["label"],
                out_path = out_path,
            )
            if summary:
                summaries.append(summary)
        except FileNotFoundError as exc:
            log.warning("Skipping condition %s: %s", cond["label"], exc)

    if not summaries:
        log.error("No conditions evaluated -- check that indexes exist.")
        return

    # Write chart-ready summary
    chart_data = {
        "conditions"   : [s["label"]         for s in summaries],
        "precision_at_k": [s.get("P@5", 0.0) for s in summaries],
        "mrr"          : [s["MRR"]            for s in summaries],
        "rouge_l"      : [s["ROUGE_L"]        for s in summaries],
        "bertscore_f1" : [s["BERTScore_F1"]   for s in summaries],
        "grounding"    : [s["grounding_mean"] for s in summaries],
    }
    summary_path = out_dir / "eval_summary.json"
    try:
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(chart_data, fh, indent=2)
        log.info("\n[evaluate] Ablation summary saved to %s", summary_path)
    except Exception as exc:
        log.error("Could not save ablation summary: %s", exc)

    # Print ASCII table
    log.info("\n  Condition      MRR    P@5    ROUGE-L  BERTScore  Grounding")
    log.info("  " + "-" * 62)
    for s in summaries:
        log.info(
            "  %-14s %.4f %.4f  %.4f   %.4f     %.4f",
            s["label"], s["MRR"], s.get("P@5", 0), s["ROUGE_L"],
            s["BERTScore_F1"], s["grounding_mean"],
        )


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    """Parse CLI arguments and run evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Scholar Stream RAG system")
    parser.add_argument("--qa",        default=str(DEFAULT_QA_PATH),
                        help="Path to test_qa.json")
    parser.add_argument("--retriever", default="hybrid",
                        choices=["faiss", "bm25", "hybrid"])
    parser.add_argument("--top_k",     type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--out",       default=str(RESULTS_DIR / "eval_results.json"),
                        help="Output path for single-condition results")
    parser.add_argument("--run_all",   action="store_true",
                        help="Evaluate all 4 experimental conditions")
    parser.add_argument("--out_dir",   default=str(RESULTS_DIR),
                        help="Output directory for --run_all mode")
    args = parser.parse_args()

    qa_path = Path(args.qa)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.run_all:
        run_all_conditions(qa_path, out_dir)
        return

    # Single condition
    alpha    = {"faiss": 1.0, "bm25": 0.0, "hybrid": 0.5}[args.retriever]
    use_bm25 = args.retriever in ("bm25", "hybrid")

    try:
        engine = load_engine(use_bm25=use_bm25)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)

    evaluate(
        engine   = engine,
        qa_path  = qa_path,
        top_k    = args.top_k,
        alpha    = alpha,
        label    = args.retriever,
        out_path = Path(args.out),
    )


if __name__ == "__main__":
    main()