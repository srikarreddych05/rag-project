# Scholar Stream
### Production RAG System over Academic Literature
**CS5202 · GenAI and LLM · Spring 2026 · Domain B — Indic NLP and Agentic AI, Project 9**

> Ask any fact-seeking question about LLM research. Scholar Stream retrieves the most relevant paper excerpts, synthesises a source-cited answer using Claude, and tells you exactly how grounded the answer is.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                    RAGEngine                         │
│                                                     │
│  ┌──────────────┐    ┌──────────────┐               │
│  │  BM25 Sparse │    │ FAISS Dense  │               │
│  │  (rank_bm25) │    │ (mpnet-768d) │               │
│  └──────┬───────┘    └──────┬───────┘               │
│         │    alpha weight   │                       │
│         └────────┬──────────┘                       │
│              Hybrid Score                           │
│                  │                                  │
│          Top-K ChunkResults                         │
│                  │                                  │
│  ┌───────────────▼────────────────┐                 │
│  │  Anthropic claude-sonnet-4-5   │                 │
│  │  (cited answer generation)     │                 │
│  └───────────────┬────────────────┘                 │
│                  │                                  │
│  ┌───────────────▼────────────────┐                 │
│  │  Grounding Check               │                 │
│  │  (bigram overlap proxy)        │                 │
│  └────────────────────────────────┘                 │
└─────────────────────────────────────────────────────┘
    │
    ▼
FullResult (answer + chunks + grounding_score + timings)
    │
    ├── Gradio UI    (src/app.py)
    └── FastAPI/HTML (webapp/main.py)
```

---

## Project Structure

```
scholar_stream/
├── .env.example                   ← copy to .env, add your API key
├── requirements.txt               ← all dependencies, pinned
├── README.md
│
├── src/
│   ├── config.py                  ← all hyperparameters (single source of truth)
│   ├── download_papers.py         ← Step 1: fetch 250 papers from arXiv
│   ├── parse_pdfs.py              ← Step 2: layout-aware PDF extraction
│   ├── build_index.py             ← Step 3: chunk + embed + FAISS/IVF index
│   ├── rag_engine.py              ← Core: hybrid retrieval + generation + grounding
│   ├── evaluate.py                ← Full eval: P@K, MRR, ROUGE-L, BERTScore
│   ├── app.py                     ← Gradio web UI
│   └── generate_synthetic_data.py ← Quick test data (no download needed)
│
├── webapp/
│   ├── main.py                    ← FastAPI application
│   ├── templates/
│   │   ├── base.html
│   │   ├── index.html             ← Hero + search + live stats
│   │   ├── results.html           ← Answer card + source cards
│   │   └── dashboard.html         ← Evaluation charts (Chart.js)
│   └── static/
│       ├── style.css              ← Dark navy + white design system
│       └── app.js                 ← Async search, loading states
│
├── data/
│   ├── papers/<topic>/            ← Downloaded PDFs (250 total)
│   ├── parsed/<topic>/            ← Extracted .txt files
│   ├── index/<model>_<chunk>/     ← FAISS index + chunks.json + build_meta.json
│   ├── metadata.json              ← arXiv paper metadata
│   └── parse_report.json          ← Per-file parsing health log
│
├── eval/
│   └── test_qa.json               ← 200 Q&A pairs for evaluation
│
└── results/
    ├── ablation_summary.json      ← Cross-config comparison
    ├── eval_summary.json          ← Chart-ready data for dashboard
    └── query_log.jsonl            ← All queries with timestamps
```

---

## Setup

### 1. Create environment
```bash
conda create -n scholar python=3.11 -y
conda activate scholar
pip install -r requirements.txt
```

### 2. Configure API key
```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=your_key_here
```

On Windows PowerShell:
```powershell
$env:ANTHROPIC_API_KEY = "your_key_here"
```

On Linux/Mac:
```bash
export ANTHROPIC_API_KEY="your_key_here"
```

---

## Run Order

### Quick test (no downloads, 2 minutes)
```bash
python src/generate_synthetic_data.py
python src/build_index.py
python src/evaluate.py
python src/app.py
```

### Full pipeline with real data (~45 min on GPU)
```bash
# Step 1: Download 250 papers
python src/download_papers.py --topic all

# Step 2: Parse PDFs
python src/parse_pdfs.py

# Step 3: Build index (default: mpnet + medium chunks)
python src/build_index.py

# Step 3b: Build all ablation configs (GPU recommended)
python src/build_index.py --all_ablations

# Step 4: Run full evaluation
python src/evaluate.py --run_all --out_dir results/

# Step 5a: Launch Gradio app
python src/app.py

# Step 5b: Launch FastAPI webapp (recommended for demo)
cd scholar_stream
uvicorn webapp.main:app --reload --port 8000
# Open http://localhost:8000
```

---

## CLI Reference

### build_index.py
```bash
python src/build_index.py --chunk [small|medium|large] --model [minilm|mpnet]
python src/build_index.py --index_type ivf     # IVFFlat for large corpora
python src/build_index.py --all_ablations      # all 6 configurations
```

### evaluate.py
```bash
python src/evaluate.py --retriever [faiss|bm25|hybrid] --top_k 5
python src/evaluate.py --run_all --out_dir results/
```

### query_rag.py
```bash
python src/query_rag.py --query "What accuracy does CoT achieve on GSM8K?"
python src/query_rag.py --interactive
```

---



## Known Limitations

1. **PDF parsing failures** (~5% of corpus): Scanned PDFs require OCR (e.g. Tesseract) which is not included. Failed files are logged in `data/parse_report.json` and excluded from the index rather than silently included with bad data.

2. **arXiv download errors**: Rate limiting, publisher access restrictions, and transient network errors affect ~10-15% of download attempts. The downloader retries gracefully and logs all failures. These are server-side constraints, not code bugs.

3. **Grounding score is approximate**: The bigram-overlap grounding check is a lightweight proxy. A trained NLI model (e.g. TRUE, AlignScore) would give more accurate hallucination detection but requires significant additional compute.

4. **BM25 on academic text**: BM25 is sensitive to exact term matching. Papers that use different terminology for the same concept (e.g. "scratchpad" vs "chain-of-thought") will score lower on BM25 than on the dense retriever. This is why the hybrid default (alpha=0.5) outperforms either alone.

5. **Context window limit**: Generation context is capped at 8,000 characters across retrieved chunks. Very long answers that require integrating many sections of a paper may be incomplete.

---

## References

1. Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020. arXiv:2005.11401

2. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP 2019. arXiv:1908.10084

3. Johnson, J., Douze, M., & Jégou, H. (2019). *Billion-Scale Similarity Search with GPUs.* IEEE Transactions on Big Data. arXiv:1702.08734 (FAISS)

4. Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in IR.

5. Wei, J., et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS 2022. arXiv:2201.11903

6. Ouyang, L., et al. (2022). *Training Language Models to Follow Instructions with Human Feedback.* NeurIPS 2022. arXiv:2203.02155
