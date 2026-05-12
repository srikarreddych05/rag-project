"""
Step 2 — Parse all PDFs → clean .txt files.

Usage:
    python src/parse_pdfs.py

Walks data/papers/<topic>/ for each topic.
Outputs to data/parsed/<topic>/<stem>.txt.
Writes data/parse_report.json with per-file health metrics.

Encoding health:
    Detects 'H e l l o  W o r l d' spacing artifact automatically.
    Falls back pdfplumber → pypdf when detected.
    Files that fail both extractors are logged as 'failed' and excluded
    from indexing — negative results are reported honestly.
"""

import re
import sys
import json
import random
import subprocess
import logging
import numpy as np
from pathlib import Path

import pdfplumber
from pypdf import PdfReader

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (PAPERS_DIR, PARSED_DIR, RESULTS_DIR, DATA_DIR,
                    ARXIV_TOPICS, SPACING_BUG_THRESHOLD, SEED)

random.seed(SEED)
np.random.seed(SEED)

LOG_FILE = RESULTS_DIR / "parse.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# ── Text extraction helpers ────────────────────────────────────────────────────

def detect_spacing_bug(text: str) -> bool:
    """Detect 'H e l l o  W o r l d' encoding artifact (char-spaced text)."""
    sample  = text[:500].replace("\n", " ")
    tokens  = sample.split()
    if not tokens:
        return False
    single  = sum(1 for t in tokens if len(t) == 1)
    return (single / len(tokens)) > SPACING_BUG_THRESHOLD


def clean_text(raw: str) -> str:
    """Normalise whitespace and remove form-feed characters."""
    text = re.sub(r"\x0c", "\n\n", raw)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdfplumber(path: Path) -> tuple[str, int]:
    parts = []
    with pdfplumber.open(path) as pdf:
        n = len(pdf.pages)
        for page in pdf.pages:
            t = page.extract_text(x_tolerance=2, y_tolerance=3)
            if t:
                parts.append(t)
    return "\n\n".join(parts), n


def extract_pypdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            parts.append(t)
    return "\n\n".join(parts)


def font_diagnostics(path: Path) -> str:
    try:
        r = subprocess.run(["pdffonts", str(path)],
                           capture_output=True, text=True, timeout=10)
        return r.stdout[:300]
    except Exception:
        return "pdffonts unavailable"


# ── Per-file processor ─────────────────────────────────────────────────────────

def process_pdf(pdf_path: Path, topic: str) -> dict:
    """
    Extract text from one PDF. Returns a health report dict.
    Saves .txt to data/parsed/<topic>/<stem>.txt on success.
    """
    report = {
        "file": pdf_path.name, "topic": topic, "status": "ok",
        "extractor": "pdfplumber", "pages": 0,
        "char_count": 0, "spacing_bug": False, "warnings": [],
    }

    out_dir = PARSED_DIR / topic
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        text, n_pages = extract_pdfplumber(pdf_path)
        report["pages"] = n_pages

        # Detect H-e-l-l-o encoding problem and fall back
        if detect_spacing_bug(text):
            report["warnings"].append("Spacing artifact — falling back to pypdf")
            log.warning(f"  [WARN] spacing artifact: {pdf_path.name}")
            text = extract_pypdf(pdf_path)
            report["extractor"]   = "pypdf (fallback)"
            report["spacing_bug"] = True

            if detect_spacing_bug(text):
                report["warnings"].append("Artifact persists after fallback — likely scanned PDF")
                report["status"]         = "degraded"
                report["font_diagnostics"] = font_diagnostics(pdf_path)

        if not text.strip():
            report["warnings"].append("No text extracted — possible scanned PDF; needs OCR")
            report["status"] = "failed"
            log.warning(f"  [FAIL] no text: {pdf_path.name}")
            return report

        text = clean_text(text)
        report["char_count"] = len(text)

        out_path = out_dir / (pdf_path.stem + ".txt")
        out_path.write_text(text, encoding="utf-8")
        log.info(f"  [OK]  {pdf_path.name}  {n_pages}p  {len(text):,}c")

    except Exception as e:
        report["status"]   = "error"
        report["warnings"].append(str(e))
        log.error(f"  [ERR] {pdf_path.name}: {e}")

    return report


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    all_reports = []
    totals = {"ok": 0, "degraded": 0, "failed": 0, "error": 0}

    for topic in ARXIV_TOPICS:
        topic_dir = PAPERS_DIR / topic
        if not topic_dir.exists():
            log.warning(f"[parse_pdfs] No papers dir for topic '{topic}' — skipping")
            continue

        pdfs = sorted(topic_dir.glob("*.pdf"))
        log.info(f"\n[parse_pdfs] Topic '{topic}': {len(pdfs)} PDFs")

        for pdf in pdfs:
            r = process_pdf(pdf, topic)
            all_reports.append(r)
            totals[r["status"]] = totals.get(r["status"], 0) + 1

    log.info(f"\n[parse_pdfs] Summary: " +
             "  ".join(f"{k}={v}" for k, v in totals.items()))

    report_path = DATA_DIR / "parse_report.json"
    with open(report_path, "w") as f:
        json.dump(all_reports, f, indent=2)
    log.info(f"[parse_pdfs] Report → {report_path}")


if __name__ == "__main__":
    main()
