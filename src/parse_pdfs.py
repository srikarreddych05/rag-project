"""
Step 2 — Parse PDFs → clean text files.

Usage:
    python src/parse_pdfs.py

Output:
    data/parsed/      — one .txt per PDF
    data/parse_report.json — per-file health report
    results/parse.log

Encoding health:
    Detects the 'H e l l o  W o r l d' spacing artifact automatically.
    Falls back from pdfplumber → pypdf if detected.
    Logs font diagnostics for any file that still fails.
"""

import re
import sys
import json
import random
import subprocess
import numpy as np
from pathlib import Path

import pdfplumber
from pypdf import PdfReader

# ── Reproducibility ───────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
IN_DIR     = ROOT / "data" / "papers"
OUT_DIR    = ROOT / "data" / "parsed"
LOG_FILE   = ROOT / "results" / "parse.log"
REPORT_OUT = ROOT / "data" / "parse_report.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:  # <-- Add encoding="utf-8"
        f.write(msg + "\n")


def detect_spacing_bug(text: str) -> bool:
    """Detect 'H e l l o  W o r l d' encoding artifact."""
    sample  = text[:500].replace("\n", " ")
    tokens  = sample.split()
    if not tokens:
        return False
    single  = sum(1 for t in tokens if len(t) == 1)
    return (single / len(tokens)) > 0.40


def clean_text(raw: str) -> str:
    text = re.sub(r"\x0c", "\n\n", raw)        # form-feed → paragraph break
    text = re.sub(r"[ \t]{2,}", " ", text)      # collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)      # max 2 blank lines
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
    parts  = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            parts.append(t)
    return "\n\n".join(parts)


def font_diagnostics(path: Path) -> str:
    try:
        r = subprocess.run(["pdffonts", str(path)],
                           capture_output=True, text=True, timeout=10)
        return r.stdout[:400]
    except Exception:
        return "pdffonts unavailable"


def process_pdf(path: Path) -> dict:
    report = {
        "file": path.name, "status": "ok",
        "extractor": "pdfplumber", "pages": 0,
        "char_count": 0, "spacing_bug": False, "warnings": [],
    }

    try:
        text, n_pages = extract_pdfplumber(path)
        report["pages"] = n_pages

        if detect_spacing_bug(text):
            report["warnings"].append("Spacing artifact — falling back to pypdf")
            log(f"  [WARN] spacing artifact in {path.name}, retrying with pypdf")
            text = extract_pypdf(path)
            report["extractor"]   = "pypdf (fallback)"
            report["spacing_bug"] = True

            if detect_spacing_bug(text):
                report["warnings"].append("Spacing artifact persists — check pdffonts")
                report["status"] = "degraded"
                report["font_diagnostics"] = font_diagnostics(path)

        if not text.strip():
            report["warnings"].append("No text extracted — possible scanned PDF (needs OCR)")
            report["status"] = "failed"
            log(f"  [FAIL] {path.name}: no text extracted")
            return report

        text = clean_text(text)
        report["char_count"] = len(text)
        (OUT_DIR / (path.stem + ".txt")).write_text(text, encoding="utf-8")
        log(f"  [OK]  {path.name}  {n_pages}p  {len(text):,} chars")

    except Exception as e:
        report["status"]   = "error"
        report["warnings"].append(str(e))
        log(f"  [ERR] {path.name}: {e}")

    return report


def main():
    pdfs = sorted(IN_DIR.glob("*.pdf"))
    if not pdfs:
        log("No PDFs found in data/papers/. Run src/download_papers.py first.")
        sys.exit(1)

    log(f"\n[parse_pdfs] Processing {len(pdfs)} PDFs")
    log("=" * 60)

    reports          = [process_pdf(p) for p in pdfs]
    ok, degraded, fail = 0, 0, 0
    for r in reports:
        if r["status"] == "ok":       ok += 1
        elif r["status"] == "degraded": degraded += 1
        else:                           fail += 1

    log("=" * 60)
    log(f"[parse_pdfs] {ok} OK  |  {degraded} degraded  |  {fail} failed")

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUT, "w") as f:
        json.dump(reports, f, indent=2)
    log(f"[parse_pdfs] report → {REPORT_OUT}")


if __name__ == "__main__":
    main()