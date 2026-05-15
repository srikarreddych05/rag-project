"""
build_index.py -- Build FAISS vector index from parsed paper text.

Upgrades over v1:
  - Windows-safe logging (ASCII only, no Unicode arrows)
  - Enriched chunk metadata: title, authors, published, chunk_id, char_start, char_end
  - IVFFlat index option via --index_type flag
  - Metadata pulled from data/metadata.json if available
  - Fully backward-compatible CLI

Usage:
    python src/build_index.py                          # default: mpnet + medium
    python src/build_index.py --chunk small --model minilm
    python src/build_index.py --all_ablations
    python src/build_index.py --index_type ivf         # IVFFlat index
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_DIR, INDEX_DIR, PARSED_DIR, RESULTS_DIR, ARXIV_TOPICS,
    CHUNK_CONFIGS, DEFAULT_CHUNK, EMBEDDING_MODELS, DEFAULT_EMBED_MODEL, SEED,
)

# ── Reproducibility ────────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)

# ── Constants ──────────────────────────────────────────────────────────────────
IVF_NLIST = 50      # number of Voronoi cells for IVFFlat
IVF_NPROBE = 10     # cells to search at query time

# ── Logging (ASCII only for Windows compatibility) ─────────────────────────────
LOG_FILE = RESULTS_DIR / "index_build.log"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ── Metadata loader ────────────────────────────────────────────────────────────

def load_paper_metadata() -> dict[str, dict]:
    """
    Load per-paper metadata from data/metadata.json.

    Returns a dict keyed by filename stem for fast lookup.
    If the file is missing, returns an empty dict (graceful degradation).
    """
    meta_path = DATA_DIR / "metadata.json"
    if not meta_path.exists():
        log.warning("metadata.json not found -- chunks will have minimal metadata")
        return {}
    try:
        with open(meta_path, encoding="utf-8") as fh:
            entries = json.load(fh)
        return {
            Path(e["filename"]).stem: e
            for e in entries
            if "filename" in e
        }
    except Exception as exc:
        log.warning("Could not load metadata.json: %s", exc)
        return {}


# ── Chunker ────────────────────────────────────────────────────────────────────

def chunk_text(
    text:          str,
    source:        str,
    topic:         str,
    chunk_chars:   int,
    overlap_chars: int,
    paper_meta:    dict,
) -> list[dict]:
    """
    Split text into overlapping fixed-size windows.

    Each chunk is enriched with paper-level metadata pulled from metadata.json.
    The 'section' field is a best-effort heuristic based on keyword scanning.

    Parameters
    ----------
    text          : full extracted paper text
    source        : filename stem (used as source key)
    topic         : arXiv topic label
    chunk_chars   : window size in characters
    overlap_chars : overlap between consecutive windows
    paper_meta    : metadata dict for this paper (may be empty)

    Returns
    -------
    List of chunk dicts with enriched metadata.
    """
    chunks = []
    idx    = 0
    start  = 0

    # Pull paper-level fields with safe defaults
    title     = paper_meta.get("title", source)
    authors   = ", ".join(paper_meta.get("authors", [])[:3])
    if len(paper_meta.get("authors", [])) > 3:
        authors += " et al."
    published = paper_meta.get("published", "")
    arxiv_id  = paper_meta.get("arxiv_id", "")

    while start < len(text):
        end        = min(start + chunk_chars, len(text))
        body       = text[start:end].strip()
        if not body:
            start += chunk_chars - overlap_chars
            continue

        # Heuristic section detection
        section = _detect_section(body)

        chunks.append({
            "chunk_id"  : f"{topic}__{source}__{idx:04d}",
            "source"    : source,
            "topic"     : topic,
            "text"      : body,
            "char_start": start,
            "char_end"  : end,
            "title"     : title,
            "authors"   : authors,
            "published" : published,
            "arxiv_id"  : arxiv_id,
            "section"   : section,
        })
        idx   += 1
        start += chunk_chars - overlap_chars

    return chunks


def _detect_section(text: str) -> str:
    """
    Heuristic: identify which section of a paper a chunk likely comes from.

    Checks for common section header keywords in the first 200 characters.
    Returns 'unknown' if no match found.
    """
    preview = text[:200].lower()
    section_map = {
        "abstract":     "abstract",
        "introduction": "introduction",
        "related work": "related_work",
        "method":       "method",
        "experiment":   "experiments",
        "result":       "results",
        "discussion":   "discussion",
        "conclusion":   "conclusion",
        "limitation":   "limitations",
        "appendix":     "appendix",
        "reference":    "references",
    }
    for keyword, label in section_map.items():
        if keyword in preview:
            return label
    return "body"


# ── Index builder ──────────────────────────────────────────────────────────────

def build_index(
    chunk_key:  str,
    model_key:  str,
    index_type: str = "flat",
) -> dict:
    """
    Build a FAISS index for one chunk-size + embedding-model configuration.

    Steps:
      1. Collect and chunk all parsed .txt files
      2. Embed chunks with SentenceTransformer
      3. Build FAISS index (FlatIP or IVFFlat)
      4. Save index.faiss, chunks.json, build_meta.json

    Parameters
    ----------
    chunk_key  : key into CHUNK_CONFIGS ('small', 'medium', 'large')
    model_key  : key into EMBEDDING_MODELS ('minilm', 'mpnet')
    index_type : 'flat' (exact) or 'ivf' (approximate, scalable)

    Returns
    -------
    build_meta dict (also saved to disk).
    """
    chunk_cfg  = CHUNK_CONFIGS[chunk_key]
    model_name = EMBEDDING_MODELS[model_key]
    run_name   = f"{model_key}_{chunk_key}"
    out_dir    = INDEX_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("[build_index] run=%s  model=%s  chunk=%dc  overlap=%dc  index=%s",
             run_name, model_name, chunk_cfg["chars"], chunk_cfg["overlap"], index_type)

    paper_meta_map = load_paper_metadata()

    # 1. Collect chunks
    all_chunks: list[dict] = []
    for topic in ARXIV_TOPICS:
        topic_dir = PARSED_DIR / topic
        if not topic_dir.exists():
            log.warning("  No parsed dir for topic '%s' -- skipping", topic)
            continue
        txts = sorted(topic_dir.glob("*.txt"))
        for txt in txts:
            try:
                text  = txt.read_text(encoding="utf-8")
                meta  = paper_meta_map.get(txt.stem, {})
                chnks = chunk_text(
                    text, txt.stem, topic,
                    chunk_cfg["chars"], chunk_cfg["overlap"], meta,
                )
                all_chunks.extend(chnks)
            except Exception as exc:
                log.warning("  Could not process %s: %s", txt.name, exc)
        log.info("  [%s] %d files -> %d chunks total",
                 topic, len(txts), len(all_chunks))

    if not all_chunks:
        log.error("  No chunks found -- run parse_pdfs.py first.")
        return {}

    log.info("  Total chunks: %d", len(all_chunks))

    # 2. Embed
    log.info("  Loading embedding model: %s", model_name)
    t0    = time.perf_counter()
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in all_chunks]

    embeddings = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    embed_time = time.perf_counter() - t0
    log.info("  Embedding done in %.1fs  shape=%s", embed_time, embeddings.shape)

    # 3. Build FAISS index
    dim = embeddings.shape[1]

    if index_type == "ivf" and len(all_chunks) >= IVF_NLIST * 4:
        quantiser = faiss.IndexFlatIP(dim)
        index     = faiss.IndexIVFFlat(quantiser, dim, IVF_NLIST, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = IVF_NPROBE
        index_label  = f"IndexIVFFlat(nlist={IVF_NLIST}, nprobe={IVF_NPROBE})"
    else:
        if index_type == "ivf":
            log.warning("  Too few chunks for IVF -- falling back to FlatIP")
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        index_label = "IndexFlatIP"

    log.info("  FAISS %s: %d vectors  dim=%d", index_label, index.ntotal, dim)

    # 4. Save
    faiss.write_index(index, str(out_dir / "index.faiss"))

    with open(out_dir / "chunks.json", "w", encoding="utf-8") as fh:
        json.dump(all_chunks, fh, indent=2, ensure_ascii=False)

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
        "index_type"    : index_label,
        "seed"          : SEED,
    }
    with open(out_dir / "build_meta.json", "w", encoding="utf-8") as fh:
        json.dump(build_meta, fh, indent=2)

    index_mb = (out_dir / "index.faiss").stat().st_size / 1e6
    log.info("  Saved to %s  (index: %.1f MB)", out_dir, index_mb)
    return build_meta


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    """Parse CLI arguments and run index build."""
    parser = argparse.ArgumentParser(description="Build FAISS index for Scholar Stream")
    parser.add_argument("--chunk",          default=DEFAULT_CHUNK,
                        choices=list(CHUNK_CONFIGS.keys()))
    parser.add_argument("--model",          default=DEFAULT_EMBED_MODEL,
                        choices=list(EMBEDDING_MODELS.keys()))
    parser.add_argument("--index_type",     default="flat", choices=["flat", "ivf"],
                        help="flat=IndexFlatIP (exact), ivf=IndexIVFFlat (scalable)")
    parser.add_argument("--all_ablations",  action="store_true",
                        help="Build all chunk x model combinations")
    args = parser.parse_args()

    if args.all_ablations:
        for ck in CHUNK_CONFIGS:
            for mk in EMBEDDING_MODELS:
                build_index(ck, mk, args.index_type)
    else:
        build_index(args.chunk, args.model, args.index_type)


if __name__ == "__main__":
    main()