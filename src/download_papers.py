"""
Step 1 — Download LLM Reasoning papers from arXiv.

Usage:
    python src/download_papers.py

Output:
    data/papers/   — downloaded PDFs
    data/metadata.json — arXiv metadata (title, authors, abstract, etc.)
    results/download.log
"""

import arxiv
import json
import time
import re
import random
import numpy as np
from pathlib import Path

# ── Reproducibility ───────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
QUERY       = "large language model reasoning chain-of-thought"
MAX_RESULTS = 18
SLEEP_SEC   = 2          # polite delay between downloads (arXiv ToS)

ROOT     = Path(__file__).parent.parent
OUT_DIR  = ROOT / "data" / "papers"
META_OUT = ROOT / "data" / "metadata.json"
LOG_FILE = ROOT / "results" / "download.log"

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def safe_filename(title: str, arxiv_id: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9 ]", "", title).strip()
    slug = re.sub(r"\s+", "_", slug)[:60]
    return f"{slug}__{arxiv_id}.pdf"


def log(msg: str):
    print(msg)
    # Make sure encoding="utf-8" is right here!
    with open(LOG_FILE, "a", encoding="utf-8") as f: 
        f.write(msg + "\n")


def main():
    log(f"[download_papers] query='{QUERY}'  max={MAX_RESULTS}")

    client = arxiv.Client()
    search = arxiv.Search(
        query=QUERY,
        max_results=MAX_RESULTS,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    metadata = []
    downloaded = 0

    for result in client.results(search):
        arxiv_id = result.entry_id.split("/")[-1]
        fname    = safe_filename(result.title, arxiv_id)
        out_path = OUT_DIR / fname

        if out_path.exists():
            log(f"  [SKIP] {fname}")
        else:
            try:
                result.download_pdf(dirpath=str(OUT_DIR), filename=fname)
                log(f"  [OK]   {fname}")
                downloaded += 1
                time.sleep(SLEEP_SEC)
            except Exception as e:
                log(f"  [ERR]  {fname}: {e}")
                continue

        metadata.append({
            "filename"  : fname,
            "title"     : result.title,
            "authors"   : [a.name for a in result.authors],
            "published" : str(result.published.date()),
            "arxiv_id"  : result.entry_id,
            "abstract"  : result.summary[:600],
            "categories": result.categories,
        })

    META_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(META_OUT, "w") as f:
        json.dump(metadata, f, indent=2)

    log(f"\n[download_papers] Done. {downloaded} new files. metadata → {META_OUT}")


if __name__ == "__main__":
    main()