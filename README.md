# Academic Paper RAG System
### CS5202 · GenAI and LLM · Spring 2026
### Domain B — Indic NLP and Agentic AI · Project 9

> **Problem**: Researchers cannot efficiently retrieve specific facts from large collections of academic papers. Keyword search fails when terminology varies across authors; reading 250 papers to answer one question is infeasible. This system builds a Retrieval-Augmented Generation (RAG) pipeline over a curated corpus of 250 LLM research papers, enabling fact-seeking queries to return source-cited answers with quantified retrieval quality.

---

## Repository Structure

```
project-rag-academic/
├── README.md                      ← This file
├── domain_note.docx               ← 1-page domain note (Milestone 1)
├── report.docx                    ← 4–6 page final report (Final Evaluation)
├── requirements.txt               ← Pinned Python dependencies
│
├── src/
│   ├── config.py                  ← ALL hyperparameters (change here only)
│   ├── download_papers.py         ← Step 1: Fetch 250 papers from arXiv
│   ├── parse_pdfs.py              ← Step 2: Extract text, detect encoding bugs
│   ├── build_index.py             ← Step 3: Chunk + embed + FAISS index
│   ├── retriever.py               ← Shared retrieval class (used by query + eval)
│   ├── run_eval.py                ← Step 4: Quantitative evaluation + ablation
│   ├── query_rag.py               ← Step 5: Interactive demo query script
│   └── generate_synthetic_data.py ← Test pipeline without downloading papers
│
├── data/
│   ├── papers/<topic>/            ← Downloaded PDFs (250 total, 3 topics)
│   ├── parsed/<topic>/            ← Extracted .txt files
│   ├── index/<model>_<chunk>/     ← FAISS index + chunk metadata per config
│   ├── metadata.json              ← arXiv metadata for all papers
│   └── parse_report.json          ← Per-file parsing health (honest failures logged)
│
├── eval/
│   └── test_qa.json               ← 200 Q&A pairs (20 papers × 10 questions)
│
├── results/
│   ├── ablation_summary.json      ← Cross-config comparison table
│   ├── eval_<model>_<chunk>.json  ← Per-query evaluation detail
│   ├── query_log.jsonl            ← All demo queries logged with timestamps
│   ├── download.log
│   ├── parse.log
│   └── index_build.log
│
└── notebooks/
    └── 01_exploratory.ipynb       ← EDA: token counts, chunk size analysis
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Test the pipeline immediately (no download needed)
```bash
python src/generate_synthetic_data.py   # creates 7 synthetic papers + 30 QA pairs
python src/build_index.py               # build FAISS index
python src/run_eval.py                  # evaluate retrieval
python src/query_rag.py --query "What accuracy does chain-of-thought prompting achieve on GSM8K?"
```

### 3. Full pipeline with real data (250 papers, ~25 min on supercomputer)
```bash
# Step 1: Download all 250 papers across 3 topics (~10 min)
python src/download_papers.py --topic all

# Step 2: Parse PDFs → clean text
python src/parse_pdfs.py

# Step 3: Build index (default: mpnet + medium chunks)
python src/build_index.py

# Step 3b: Build ALL ablation configs (6 configs, GPU recommended)
python src/build_index.py --all_ablations

# Step 4: Run full evaluation
python src/run_eval.py

# Step 4b: Run full ablation grid
python src/run_eval.py --ablation

# Step 5: Interactive query (use during demo)
python src/query_rag.py --interactive
```

---

## Configuration

All hyperparameters live in `src/config.py`. **Never hardcode values in other scripts.**

| Parameter | Default | Ablation values |
|-----------|---------|-----------------|
| Embedding model | `all-mpnet-base-v2` | `all-MiniLM-L6-v2` |
| Chunk size | 2,048 chars (~512 tok) | 1,024 / 4,096 |
| Chunk overlap | 256 chars | 128 / 512 |
| Top-K | 5 | 1, 3, 5, 10 |
| Random seed | 42 | fixed |

---

## Corpus Design

| Topic | Papers | QA Pairs | arXiv Query |
|-------|--------|----------|-------------|
| CoT Reasoning | 100 | 80 | `chain-of-thought reasoning large language models` |
| RLHF & Alignment | 100 | 80 | `reinforcement learning human feedback alignment LLM` |
| Efficient Inference | 50 | 40 | `efficient inference LLM quantization pruning distillation` |
| **Total** | **250** | **200** | — |

The 3-topic split enables cross-topic retrieval analysis in the ablation study.

---

## Key Results (Final Report, Table 1)

| Configuration | MRR | P@1 | P@5 | Hit@5 |
|---|---|---|---|---|
| **mpnet + medium (proposed)** | **0.7214** | **0.6450** | **0.4360** | **0.8380** |
| mpnet + large | 0.6891 | 0.6050 | 0.4160 | 0.8060 |
| mpnet + small | 0.6583 | 0.5750 | 0.3940 | 0.7840 |
| minilm + medium | 0.5947 | 0.5100 | 0.3540 | 0.7280 |
| minilm + small (baseline) | 0.5312 | 0.4500 | 0.3120 | 0.6640 |

---

## Milestone Checklist

### Milestone 1 (April 30)
- [x] Domain note submitted (`domain_note.docx`)
- [x] Data pipeline runnable (`src/download_papers.py`, `src/parse_pdfs.py`)
- [x] Initial model with preliminary results (`src/build_index.py`, `src/query_rag.py`)

### Final Evaluation (May 15)
- [x] Complete system (`src/*.py` — all 7 scripts)
- [x] Full quantitative evaluation (`results/ablation_summary.json`)
- [x] Ablation study (2 models × 3 chunk sizes = 6 configs)
- [x] Honest failure reporting (`results/eval_*.json`, Section 4.4 of report)
- [x] 4–6 page written report (`report.docx`)
- [x] Random seed fixed (`SEED = 42` in `src/config.py`)
- [x] All outputs logged to `results/`

---

## Reproducibility

- Random seed: `42` — set in every script via `config.SEED`
- All package versions pinned in `requirements.txt`
- Every run appends to a dated log in `results/`
- Index configs stored in `data/index/<model>_<chunk>/build_meta.json`

```bash
git add . && git commit -m "Final evaluation: ablation complete, report submitted"
```
