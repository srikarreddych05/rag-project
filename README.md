Academic Paper RAG System
CS5202 · Spring 2026 · Domain B — Indic NLP and Agentic AI (Project 9)

> **Problem**: Researchers and students struggle to find precise answers buried inside large collections of academic papers. Keyword search fails when terminology varies; reading 20 papers to answer one question is infeasible. This system builds a Retrieval-Augmented Generation (RAG) pipeline over a curated corpus of LLM Reasoning papers, enabling fact-seeking queries to be answered with grounded, source-cited responses.


**Domain**: Indic NLP and Agentic AI- arXiv papers

---

## Repository Structure

```
project-rag-academic/
├── README.md                  
├── domain_note.pdf            ← 1-page domain note 
├── requirements.txt           ← Python dependencies
├── data/
│   ├── papers/                ← Downloaded PDFs (18 papers)
│   ├── parsed/                ← Extracted .txt files
│   ├── metadata.json          ← arXiv metadata for all papers
│   └── parse_report.json      ← Per-file parsing health log
├── src/
│   ├── download_papers.py     ← Pipeline Step 1: Fetch from arXiv
│   ├── parse_pdfs.py          ← Pipeline Step 2: Extract text
│   ├── build_index.py         ← Pipeline Step 3: Embed + FAISS index
│   └── query_rag.py           ← Pipeline Step 4: Query the RAG system
|
├── results/
│   ├── query_log.jsonl        ← All queries and retrieved chunks
│   └── parse.log      ← Copy of data/parse_report.json
|   └── download.log   ← Copy of dowload logs
|   └── index_build.log ← Copy of index building logs
└── eval/
    └── test_qa.json           ← 20 Q&A pairs for Day 5 evaluation
```

---



### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download papers (run from repo root)
```bash
python src/download_papers.py
```
Downloads 18 LLM Reasoning papers into `data/papers/`.  (for prototype, further we will use 250-300 papers)
Check `data/metadata.json` for titles and arXiv IDs.

### 3. Parse PDFs → clean text
```bash
python src/parse_pdfs.py
```
Outputs `.txt` files to `data/parsed/`.  
Check `data/parse_report.json` for any encoding issues.

### 4. Build FAISS index
```bash
python src/build_index.py
```
Embeds all parsed text chunks and saves a FAISS index to `data/`.

### 5. Query the system
```bash
python src/query_rag.py --query "What accuracy does chain-of-thought prompting achieve on GSM8K?"
```

---

## Milestone 1 achievements so far
- [x] Domain research note written
- [x] Data pipeline  written and running in src folder
- [x] PDF parser verified with encoding health check- with some failures
- [x] FAISS index builder implemented
- [x] RAG query pipeline working end-to-end


---

## Reproducibility
- All scripts use `random.seed(42)` and `numpy.random.seed(42)`  
- All outputs logged to `results/`  
- Negative results are reported in `data/parse_report.json`