"""
Step 1 — Download papers from arXiv across three sub-topics.

Usage:
    python src/download_papers.py [--topic cot_reasoning|rlhf_alignment|efficient_inference|all]

Downloads up to 250 papers total (100 + 100 + 50) into data/papers/<topic>/.
Saves unified metadata to data/metadata.json.
All downloads logged to results/download.log.
"""

import argparse
import json
import re
import time
import random
import logging
import numpy as np
from pathlib import Path

import arxiv

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (ARXIV_TOPICS, DOWNLOAD_SLEEP_SEC, PAPERS_DIR,
                    DATA_DIR, RESULTS_DIR, SEED)

random.seed(SEED)
np.random.seed(SEED)

LOG_FILE = RESULTS_DIR / "download.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def safe_filename(title: str, arxiv_id: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9 ]", "", title).strip()
    slug = re.sub(r"\s+", "_", slug)[:55]
    return f"{slug}__{arxiv_id}.pdf"


def download_topic(topic_name: str, topic_cfg: dict) -> list[dict]:
    """Download papers for one topic. Returns metadata list."""
    out_dir = PAPERS_DIR / topic_name
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"[{topic_name}] query='{topic_cfg['query']}'  max={topic_cfg['max_results']}")

    client = arxiv.Client()
    search = arxiv.Search(
        query=topic_cfg["query"],
        max_results=topic_cfg["max_results"],
        sort_by=arxiv.SortCriterion.Relevance,
    )

    metadata, downloaded = [], 0

    for result in client.results(search):
        arxiv_id = result.entry_id.split("/")[-1]
        fname    = safe_filename(result.title, arxiv_id)
        out_path = out_dir / fname

        if out_path.exists():
            log.info(f"  [SKIP] {fname}")
        else:
            try:
                result.download_pdf(dirpath=str(out_dir), filename=fname)
                log.info(f"  [OK]   {fname}")
                downloaded += 1
                time.sleep(DOWNLOAD_SLEEP_SEC)
            except Exception as e:
                log.warning(f"  [ERR]  {fname}: {e}")
                continue

        metadata.append({
            "filename"  : fname,
            "topic"     : topic_name,
            "title"     : result.title,
            "authors"   : [a.name for a in result.authors],
            "published" : str(result.published.date()),
            "arxiv_id"  : result.entry_id,
            "abstract"  : result.summary[:600],
            "categories": result.categories,
        })

    log.info(f"[{topic_name}] Done. {downloaded} new downloads.\n")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="all",
                        choices=list(ARXIV_TOPICS.keys()) + ["all"])
    args = parser.parse_args()

    topics = ARXIV_TOPICS if args.topic == "all" else {args.topic: ARXIV_TOPICS[args.topic]}

    all_meta = []
    for name, cfg in topics.items():
        all_meta.extend(download_topic(name, cfg))

    meta_path = DATA_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(all_meta, f, indent=2)

    log.info(f"Metadata saved → {meta_path}  ({len(all_meta)} entries)")
    log.info(f"Topic breakdown: " +
             ", ".join(f"{k}={sum(1 for m in all_meta if m['topic']==k)}"
                       for k in topics))


if __name__ == "__main__":
    main()
