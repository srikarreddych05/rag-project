"""
Step 3 — Chunk parsed text, embed, and build a FAISS index.

Usage:
    # Default (medium chunks, mpnet embeddings)
    python src/build_index.py

    # Ablation: vary chunk size
    python src/build_index.py --chunk small
    python src/build_index.py --chunk large

    # Ablation: vary embedding model
    python src/build_index.py --model minilm
    python src/build_index.py --model mpnet

    # Combine for full ablation grid
    python src/build_index.py --chunk medium --model minilm

Output per configuration:
    data/index/<model>_<chunk>/index.faiss
    data/index/<model>_<chunk>/chunks.json
    data/index/<model>_<chunk>/build_meta.json   ← config snapshot for audit

All runs appended to results/index_build.log.
"""

import argparse
import json
import random
import logging
import time
import numpy as np
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (PARSED_DIR, INDEX_DIR, RESULTS_DIR, ARXIV_TOPICS,
                    CHUNK_CONFIGS, DEFAULT_CHUNK,
                    EMBEDDING_MODELS, DEFAULT_EMBED_MODEL, SEED)

random.seed(SEED)
np.random.seed(SEED)

LOG_FILE = RESULTS_DIR / "index_build.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_text(text: str, source: str, topic: str,
               chunk_chars: int, overlap_chars: int) -> list[dict]:
    """
    Fixed-size sliding window chunker with overlap.
    Each chunk stores its source file, topic, and character offset
    so we can trace retrievals back to exact document positions.
    """
    chunks, idx, start = [], 0, 0
    while start < len(text):
        end        = min(start + chunk_chars, len(text))
        chunk_body = text[start:end].strip()
        if chunk_body:
            chunks.append({
                "chunk_id"  : f"{topic}__{source}__{idx:04d}",
                "source"    : source,
                "topic"     : topic,
                "text"      : chunk_body,
                "char_start": start,
                "char_end"  : end,
            })
            idx += 1
        start += chunk_chars - overlap_chars
    return chunks


# ── Index builder ──────────────────────────────────────────────────────────────

def build_index(chunk_key: str, model_key: str):
    chunk_cfg  = CHUNK_CONFIGS[chunk_key]
    model_name = EMBEDDING_MODELS[model_key]
    run_name   = f"{model_key}_{chunk_key}"
    out_dir    = INDEX_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"\n[build_index] run={run_name}  model={model_name}  "
             f"chunk={chunk_cfg['chars']}c  overlap={chunk_cfg['overlap']}c")

    # 1. Collect all chunks across topics
    all_chunks: list[dict] = []
    for topic in ARXIV_TOPICS:
        topic_dir = PARSED_DIR / topic
        if not topic_dir.exists():
            log.warning(f"  No parsed dir for topic '{topic}' — skipping")
            continue
        txts = sorted(topic_dir.glob("*.txt"))
        for txt in txts:
            text   = txt.read_text(encoding="utf-8")
            chunks = chunk_text(text, txt.stem, topic,
                                chunk_cfg["chars"], chunk_cfg["overlap"])
            all_chunks.extend(chunks)
        log.info(f"  [{topic}] {len(txts)} files → "
                 f"{sum(1 for c in all_chunks if c['topic']==topic)} chunks so far")

    log.info(f"  Total chunks: {len(all_chunks)}")

    if not all_chunks:
        log.error("  No chunks found — run parse_pdfs.py first.")
        return

    # 2. Embed
    log.info(f"  Loading embedding model: {model_name}")
    t0    = time.time()
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in all_chunks]

    embeddings = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # normalise → cosine = dot product
    ).astype("float32")

    embed_time = time.time() - t0
    log.info(f"  Embedding done in {embed_time:.1f}s  shape={embeddings.shape}")

    # 3. Build FAISS index
    # Using IndexFlatIP (inner product on normalised vectors = cosine similarity)
    # Exact search — justified for <100k chunks; no approximation error.
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    log.info(f"  FAISS IndexFlatIP: {index.ntotal} vectors  dim={dim}")

    # 4. Save
    faiss.write_index(index, str(out_dir / "index.faiss"))
    with open(out_dir / "chunks.json", "w") as f:
        json.dump(all_chunks, f, indent=2)

    build_meta = {
        "run_name"      : run_name,
        "model_key"     : model_key,
        "model_name"    : model_name,
        "chunk_key"     : chunk_key,
        "chunk_chars"   : chunk_cfg["chars"],
        "overlap_chars" : chunk_cfg["overlap"],
        "total_chunks"  : len(all_chunks),
        "embedding_dim" : dim,
        "embed_time_sec": round(embed_time, 2),
        "index_type"    : "IndexFlatIP",
        "seed"          : SEED,
    }
    with open(out_dir / "build_meta.json", "w") as f:
        json.dump(build_meta, f, indent=2)

    index_mb = (out_dir / "index.faiss").stat().st_size / 1e6
    log.info(f"  Saved → {out_dir}  (index: {index_mb:.1f} MB)")
    return build_meta


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index")
    parser.add_argument("--chunk", default=DEFAULT_CHUNK,
                        choices=list(CHUNK_CONFIGS.keys()))
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL,
                        choices=list(EMBEDDING_MODELS.keys()))
    parser.add_argument("--all_ablations", action="store_true",
                        help="Build all chunk×model combinations for ablation study")
    args = parser.parse_args()

    if args.all_ablations:
        for chunk_key in CHUNK_CONFIGS:
            for model_key in EMBEDDING_MODELS:
                build_index(chunk_key, model_key)
    else:
        build_index(args.chunk, args.model)


if __name__ == "__main__":
    main()
