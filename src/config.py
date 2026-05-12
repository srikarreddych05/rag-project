"""
config.py — Central configuration for the Academic RAG system.

All hyperparameters, paths, and model choices live here.
Change values here; every other script imports from this file.
This design ensures reproducibility: one file to audit, one file to change.
"""

from pathlib import Path

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / "data"
PAPERS_DIR  = DATA_DIR / "papers"
PARSED_DIR  = DATA_DIR / "parsed"
INDEX_DIR   = DATA_DIR / "index"
EVAL_DIR    = ROOT / "eval"
RESULTS_DIR = ROOT / "results"

for d in [PAPERS_DIR, PARSED_DIR, INDEX_DIR, EVAL_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Data collection ────────────────────────────────────────────────────────────
ARXIV_TOPICS = {
    "cot_reasoning": {
        "query": "chain-of-thought reasoning large language models",
        "max_results": 100,
    },
    "rlhf_alignment": {
        "query": "reinforcement learning human feedback alignment large language models",
        "max_results": 100,
    },
    "efficient_inference": {
        "query": "efficient inference LLM quantization pruning distillation",
        "max_results": 50,
    },
}
DOWNLOAD_SLEEP_SEC = 2

# ── Parsing ────────────────────────────────────────────────────────────────────
SPACING_BUG_THRESHOLD = 0.40

# ── Chunking strategies (for ablation study) ───────────────────────────────────
CHUNK_CONFIGS = {
    "small":  {"chars": 1024, "overlap": 128},
    "medium": {"chars": 2048, "overlap": 256},
    "large":  {"chars": 4096, "overlap": 512},
}
DEFAULT_CHUNK = "medium"

# ── Embedding models (for ablation study) ─────────────────────────────────────
EMBEDDING_MODELS = {
    "minilm": "all-MiniLM-L6-v2",      # 384-dim — baseline
    "mpnet":  "all-mpnet-base-v2",      # 768-dim — primary
}
DEFAULT_EMBED_MODEL = "mpnet"

# ── Retrieval ─────────────────────────────────────────────────────────────────
DEFAULT_TOP_K  = 5
EVAL_K_VALUES  = [1, 3, 5, 10]
