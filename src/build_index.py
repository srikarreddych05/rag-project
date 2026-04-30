"""
Step 3 — Build FAISS vector index from parsed paper text.

Usage:
    python src/build_index.py

Output:
    data/index.faiss        — FAISS flat-L2 index
    data/chunks.json        — chunk text + metadata (source, chunk_id)
    results/index_build.log

Chunking strategy:
    Fixed-size with overlap.  Chunk size and overlap are logged so
    you can ablate them on Day 5.
    CHUNK_SIZE = 512 tokens (chars ÷ 4 approximation)
    OVERLAP    = 64  tokens
"""

import json
import random
import numpy as np
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

# ── Reproducibility ───────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"   # 384-dim, fast, good for retrieval
CHUNK_CHARS = 2048                  # ~512 tokens (4 chars ≈ 1 token)
OVERLAP_CHARS = 256                 # ~64 tokens overlap

ROOT       = Path(__file__).parent.parent
PARSED_DIR = ROOT / "data" / "parsed"
INDEX_OUT  = ROOT / "data" / "index.faiss"
CHUNKS_OUT = ROOT / "data" / "chunks.json"
LOG_FILE   = ROOT / "results" / "index_build.log"

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:  # <-- Add encoding="utf-8"
        f.write(msg + "\n")


def chunk_text(text: str, source: str) -> list[dict]:
    """
    Split text into overlapping fixed-size windows.
    Returns list of dicts: {chunk_id, source, text, char_start}
    """
    chunks = []
    start  = 0
    idx    = 0
    while start < len(text):
        end  = min(start + CHUNK_CHARS, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "chunk_id"  : f"{source}::{idx}",
                "source"    : source,
                "text"      : chunk_text,
                "char_start": start,
            })
            idx += 1
        start += CHUNK_CHARS - OVERLAP_CHARS
    return chunks


def main():
    txts = sorted(PARSED_DIR.glob("*.txt"))
    if not txts:
        log("No parsed .txt files found. Run src/parse_pdfs.py first.")
        return

    log(f"\n[build_index] model={EMBED_MODEL}  chunk={CHUNK_CHARS}c  overlap={OVERLAP_CHARS}c")
    log(f"[build_index] Loading {len(txts)} parsed files")

    # 1. Build chunks
    all_chunks: list[dict] = []
    for txt in txts:
        text = txt.read_text(encoding="utf-8")
        file_chunks = chunk_text(text, txt.stem)
        all_chunks.extend(file_chunks)
        log(f"  {txt.name}  → {len(file_chunks)} chunks")

    log(f"\n[build_index] Total chunks: {len(all_chunks)}")

    # 2. Embed
    log(f"[build_index] Embedding with {EMBED_MODEL} ...")
    model  = SentenceTransformer(EMBED_MODEL)
    texts  = [c["text"] for c in all_chunks]
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    log(f"[build_index] Embedding shape: {embeddings.shape}")

    # 3. Build FAISS flat index (exact L2 search — good for <100k chunks)
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_OUT))
    log(f"[build_index] FAISS index saved → {INDEX_OUT}  ({index.ntotal} vectors)")

    # 4. Save chunk metadata (without embeddings — kept in FAISS)
    with open(CHUNKS_OUT, "w") as f:
        json.dump(all_chunks, f, indent=2)
    log(f"[build_index] Chunks saved → {CHUNKS_OUT}")

    log("\n[build_index] Done.")


if __name__ == "__main__":
    main()