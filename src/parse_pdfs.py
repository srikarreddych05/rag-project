"""
parse_pdfs.py -- Parse PDFs to clean text files for Scholar Stream.

Upgrades over v1:
  - pdfplumber with layout=True for column-aware multi-column PDF handling
  - Post-processing: fix hyphenated line breaks, remove non-ASCII garbage,
    collapse whitespace
  - Parse quality score per file (% non-ASCII chars as noise proxy)
  - ASCII-only logging (Windows compatible)
  - Same parse_report.json output format (backward compatible)

Usage:
    python src/parse_pdfs.py
"""

import json
import logging
import random
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pdfplumber
from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PAPERS_DIR, PARSED_DIR, RESULTS_DIR, DATA_DIR,
    ARXIV_TOPICS, SPACING_BUG_THRESHOLD, SEED,
)

# ── Reproducibility ────────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)

# ── Logging (ASCII only) ───────────────────────────────────────────────────────
LOG_FILE = RESULTS_DIR / "parse.log"
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


# ── Text extraction helpers ────────────────────────────────────────────────────

def detect_spacing_bug(text: str) -> bool:
    """
    Detect the 'H e l l o  W o r l d' character-spaced encoding artifact.

    Heuristic: if more than SPACING_BUG_THRESHOLD fraction of tokens
    in the first 500 characters are single characters, the encoding is broken.

    Parameters
    ----------
    text : extracted text to inspect

    Returns
    -------
    True if spacing artifact is detected.
    """
    sample  = text[:500].replace("\n", " ")
    tokens  = sample.split()
    if not tokens:
        return False
    single  = sum(1 for t in tokens if len(t) == 1)
    return (single / len(tokens)) > SPACING_BUG_THRESHOLD


def extract_pdfplumber(path: Path) -> tuple[str, int]:
    """
    Primary extractor using pdfplumber with layout-aware settings.

    Uses extract_text(layout=True) which respects column ordering
    in multi-column academic papers. Falls back to layout=False
    if layout mode raises an error.

    Parameters
    ----------
    path : path to PDF file

    Returns
    -------
    Tuple of (extracted_text, page_count).
    """
    parts = []
    with pdfplumber.open(path) as pdf:
        n_pages = len(pdf.pages)
        for page in pdf.pages:
            try:
                # layout=True handles two-column paper layouts correctly
                t = page.extract_text(layout=True, x_tolerance=2, y_tolerance=3)
            except Exception:
                # Fallback if layout mode fails on this page
                t = page.extract_text(x_tolerance=2, y_tolerance=3)
            if t:
                parts.append(t)
    return "\n\n".join(parts), n_pages


def extract_pypdf(path: Path) -> str:
    """
    Fallback extractor using pypdf.

    Used when pdfplumber produces garbled or spacing-bugged text.

    Parameters
    ----------
    path : path to PDF file

    Returns
    -------
    Extracted text string.
    """
    reader = PdfReader(str(path))
    parts  = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            parts.append(t)
    return "\n\n".join(parts)


def clean_text(raw: str) -> str:
    """
    Post-process extracted text to remove common PDF artifacts.

    Steps applied in order:
      1. Replace form-feed characters with paragraph breaks
      2. Fix hyphenated line breaks (re-join split words)
      3. Remove non-ASCII characters (garbage from encoding issues)
      4. Collapse excessive whitespace
      5. Limit consecutive blank lines to 2

    Parameters
    ----------
    raw : raw extracted text

    Returns
    -------
    Cleaned text string.
    """
    # Form-feed -> paragraph break
    text = re.sub(r"\x0c", "\n\n", raw)

    # Fix hyphenated line breaks: "meth-\nod" -> "method"
    text = re.sub(r"-\s*\n\s*([a-z])", r"\1", text)

    # Remove non-ASCII garbage characters (keep standard printable + newlines)
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)

    # Collapse multiple spaces/tabs to single space
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Collapse lines that are just whitespace
    text = re.sub(r"\n[ \t]+\n", "\n\n", text)

    # Limit consecutive blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def quality_score(text: str) -> float:
    """
    Compute a parse quality score for a text file.

    Proxy metric: fraction of printable ASCII characters.
    High score (close to 1.0) = clean extraction.
    Low score (< 0.85) = likely noisy or garbled.

    Parameters
    ----------
    text : cleaned text to assess

    Returns
    -------
    Float in [0.0, 1.0].
    """
    if not text:
        return 0.0
    ascii_count = sum(1 for c in text if 0x20 <= ord(c) <= 0x7E or c == "\n")
    return round(ascii_count / len(text), 4)


def font_diagnostics(path: Path) -> str:
    """
    Run pdffonts and return a summary string for diagnostic logging.

    Parameters
    ----------
    path : path to PDF file

    Returns
    -------
    pdffonts output (first 300 chars) or a fallback message.
    """
    try:
        result = subprocess.run(
            ["pdffonts", str(path)],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout[:300]
    except Exception:
        return "pdffonts unavailable"


# ── Per-file processor ─────────────────────────────────────────────────────────

def process_pdf(pdf_path: Path, topic: str) -> dict:
    """
    Extract, clean, and save text from a single PDF file.

    Tries pdfplumber first. If spacing artifact is detected, switches to
    pypdf fallback. Saves clean .txt to data/parsed/<topic>/<stem>.txt.
    Returns a health report dict for parse_report.json.

    Parameters
    ----------
    pdf_path : path to the PDF file
    topic    : topic label (used for output directory)

    Returns
    -------
    Dict with status, extractor, pages, char_count, quality_score, warnings.
    """
    report: dict = {
        "file"         : pdf_path.name,
        "topic"        : topic,
        "status"       : "ok",
        "extractor"    : "pdfplumber",
        "pages"        : 0,
        "char_count"   : 0,
        "quality_score": 0.0,
        "spacing_bug"  : False,
        "warnings"     : [],
    }

    out_dir = PARSED_DIR / topic
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        text, n_pages = extract_pdfplumber(pdf_path)
        report["pages"] = n_pages

        # Detect spacing artifact and fall back if found
        if detect_spacing_bug(text):
            report["warnings"].append("Spacing artifact detected -- switching to pypdf")
            log.warning("  [WARN] spacing artifact: %s", pdf_path.name)
            text                 = extract_pypdf(pdf_path)
            report["extractor"]  = "pypdf (fallback)"
            report["spacing_bug"] = True

            if detect_spacing_bug(text):
                report["warnings"].append("Artifact persists after fallback -- scanned PDF")
                report["status"]            = "degraded"
                report["font_diagnostics"]  = font_diagnostics(pdf_path)

        if not text.strip():
            report["warnings"].append("No text extracted -- likely scanned PDF (needs OCR)")
            report["status"] = "failed"
            log.warning("  [FAIL] no text: %s", pdf_path.name)
            return report

        text                   = clean_text(text)
        report["char_count"]   = len(text)
        report["quality_score"] = quality_score(text)

        # Flag low-quality parses
        if report["quality_score"] < 0.85:
            report["warnings"].append(
                f"Low quality score {report['quality_score']:.2f} -- "
                "possible encoding noise"
            )

        out_path = out_dir / (pdf_path.stem + ".txt")
        out_path.write_text(text, encoding="utf-8")

        log.info(
            "  [OK]  %s  %dp  %d chars  quality=%.2f",
            pdf_path.name, n_pages, len(text), report["quality_score"],
        )

    except Exception as exc:
        report["status"]   = "error"
        report["warnings"].append(str(exc))
        log.error("  [ERR] %s: %s", pdf_path.name, exc)

    return report


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Walk all topic directories and parse every PDF found."""
    all_reports: list[dict] = []
    totals: dict[str, int] = {"ok": 0, "degraded": 0, "failed": 0, "error": 0}

    for topic in ARXIV_TOPICS:
        topic_dir = PAPERS_DIR / topic
        if not topic_dir.exists():
            log.warning("[parse_pdfs] No papers dir for topic '%s' -- skipping", topic)
            continue

        pdfs = sorted(topic_dir.glob("*.pdf"))
        log.info("\n[parse_pdfs] Topic '%s': %d PDFs", topic, len(pdfs))

        for pdf in pdfs:
            report = process_pdf(pdf, topic)
            all_reports.append(report)
            totals[report["status"]] = totals.get(report["status"], 0) + 1

    log.info(
        "\n[parse_pdfs] Summary: %s",
        "  ".join(f"{k}={v}" for k, v in totals.items()),
    )

    report_path = DATA_DIR / "parse_report.json"
    try:
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(all_reports, fh, indent=2)
        log.info("[parse_pdfs] Report saved to %s", report_path)
    except Exception as exc:
        log.error("[parse_pdfs] Could not save report: %s", exc)


if __name__ == "__main__":
    main()