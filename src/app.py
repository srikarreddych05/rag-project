"""
app.py -- Gradio web interface for Scholar Stream RAG system.

Upgrades over v1:
  - Uses RAGEngine for hybrid retrieval + LLM generation
  - Displays grounding score as a colored confidence badge
  - Shows generated answer at top, source cards below
  - Sliders for top_k and alpha (hybrid retrieval weight)
  - Toggle for generation ON/OFF (compare retrieval-only vs full RAG)

Run:
    python src/app.py
    python src/app.py --share    # public Gradio link
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# ── Load .env FIRST before anything else ──────────────────────────────────────
load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    INDEX_DIR, DEFAULT_EMBED_MODEL, DEFAULT_CHUNK,
    EMBEDDING_MODELS, SEED,
)
from rag_engine import RAGEngine, FullResult

# ── Reproducibility ────────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

# ── Constants ──────────────────────────────────────────────────────────────────
APP_TITLE = "Scholar Stream -- Academic RAG"
APP_DESC  = (
    "Ask any question about LLM Reasoning, RLHF & Alignment, or Efficient Inference. "
    "The system retrieves relevant paper excerpts and synthesises a cited answer."
)
DEMO_QUESTIONS = [
    "What accuracy does chain-of-thought prompting achieve on GSM8K?",
    "How many pairwise comparisons were used to train the InstructGPT reward model?",
    "What is the inference speedup of GPTQ INT4 over FP16?",
    "What trigger phrase does zero-shot CoT use?",
    "How does QLoRA reduce memory compared to full fine-tuning?",
]


# ── Engine loader (cached) ─────────────────────────────────────────────────────

_engine: RAGEngine | None = None

def get_engine() -> RAGEngine | None:
    """
    Load and cache the RAGEngine singleton.

    Returns None if the index files are not found (shows helpful error in UI).
    """
    global _engine
    if _engine is not None:
        return _engine

    run_name    = f"{DEFAULT_EMBED_MODEL}_{DEFAULT_CHUNK}"
    run_dir     = INDEX_DIR / run_name
    index_path  = run_dir / "index.faiss"
    chunks_path = run_dir / "chunks.json"

    if not index_path.exists():
        log.error("Index not found at %s -- run build_index.py first", index_path)
        return None

    try:
        _engine = RAGEngine(
            index_path  = str(index_path),
            chunks_path = str(chunks_path),
            model_name  = EMBEDDING_MODELS[DEFAULT_EMBED_MODEL],
            use_bm25    = True,
        )
        log.info("RAGEngine loaded -- provider=%s  model=%s",
                 _engine.llm_provider, _engine.llm_model)
        return _engine
    except Exception as exc:
        log.error("Failed to load RAGEngine: %s", exc)
        return None


# ── Grounding badge HTML ───────────────────────────────────────────────────────

def grounding_badge(score: float) -> str:
    """
    Return an HTML badge coloured by grounding score.

    Green  (>0.7) : well grounded
    Yellow (0.4-0.7): partially grounded
    Red    (<0.4) : possible hallucination
    """
    if score >= 0.7:
        colour, label = "#2d7a2d", "HIGH"
    elif score >= 0.4:
        colour, label = "#b8860b", "MEDIUM"
    else:
        colour, label = "#a02020", "LOW"

    return (
        f'<span style="background:{colour}; color:white; padding:3px 10px; '
        f'border-radius:12px; font-size:13px; font-weight:600;">'
        f'Grounding: {label} ({score:.2f})</span>'
    )


# ── Provider badge HTML ────────────────────────────────────────────────────────

def provider_badge(engine: RAGEngine) -> str:
    """Show which LLM provider is active as a small badge."""
    provider = engine.llm_provider or "none"
    colours  = {
        "groq":      "#6c4fc4",
        "anthropic": "#c96a2a",
        "openai":    "#19a37f",
        "none":      "#888888",
    }
    colour = colours.get(provider, "#888888")
    return (
        f'<span style="background:{colour}; color:white; padding:2px 8px; '
        f'border-radius:10px; font-size:11px; font-weight:500;">'
        f'LLM: {provider} / {engine.llm_model}</span>'
    )


# ── Main query handler ─────────────────────────────────────────────────────────

def query_handler(
    question:      str,
    top_k:         int,
    alpha:         float,
    generation_on: bool,
) -> tuple[str, str, str]:
    """
    Handle a query from the Gradio UI.

    Parameters
    ----------
    question      : user question string
    top_k         : number of chunks to retrieve
    alpha         : hybrid weight (0=BM25 only, 1=FAISS only)
    generation_on : whether to call the LLM for answer synthesis

    Returns
    -------
    Tuple of (answer_html, grounding_html, sources_html)
    """
    if not question or not question.strip():
        return (
            "<p style='color:gray'>Please enter a question.</p>",
            "", "",
        )

    engine = get_engine()
    if engine is None:
        return (
            "<p style='color:red'><b>Error:</b> Index not loaded. "
            "Run <code>python src/build_index.py</code> first.</p>",
            "", "",
        )

    try:
        if generation_on:
            result: FullResult = engine.ask(question, top_k=top_k, alpha=alpha)
            answer_text        = result.generation.answer
            grounding          = result.grounding_score
            chunks             = result.chunks
            timing             = result.total_ms
            fallback           = result.generation.fallback
        else:
            chunks      = engine.retrieve(question, top_k=top_k, alpha=alpha)
            answer_text = (
                "[Generation OFF -- showing raw retrieved chunks]\n\n"
                + "\n\n---\n\n".join(
                    f"[Rank {c.rank} | Score {c.score:.3f}]\n{c.text[:400]}..."
                    for c in chunks
                )
            )
            grounding = 0.0
            timing    = 0.0
            fallback  = True

        # ── Answer card ────────────────────────────────────────────────────────
        border_colour = "#2E75B6" if not fallback else "#888"
        answer_html   = (
            f"<div style='background:#f8f9fa; border-left:4px solid {border_colour}; "
            f"padding:16px; border-radius:4px; font-size:14px; line-height:1.7;'>"
            f"<b>Answer</b>"
            f"&nbsp;&nbsp;{provider_badge(engine)}<br><br>"
            f"{answer_text.replace(chr(10), '<br>')}"
            f"</div>"
            f"<p style='color:gray; font-size:12px; margin-top:6px;'>"
            f"Query time: {timing:.0f} ms &nbsp;|&nbsp; "
            f"top_k={top_k} &nbsp;|&nbsp; alpha={alpha:.2f} &nbsp;|&nbsp; "
            f"chunks indexed: {engine.n_chunks}</p>"
        )

        # ── Grounding badge ────────────────────────────────────────────────────
        badge_html = grounding_badge(grounding) if generation_on and not fallback else ""

        # ── Source cards ───────────────────────────────────────────────────────
        cards = []
        for c in chunks:
            arxiv_url = f"https://arxiv.org/abs/{c.arxiv_id}" if c.arxiv_id else "#"
            card = (
                f"<div style='border:1px solid #ddd; border-radius:8px; "
                f"padding:12px; margin-bottom:10px; background:#fff;'>"
                f"<b>#{c.rank}</b>&nbsp;&nbsp;"
                f"<span style='color:#2E75B6; font-weight:600;'>"
                f"{c.title or c.source or c.arxiv_id}</span>"
                f"<span style='float:right; font-size:12px; color:gray;'>"
                f"hybrid={c.score:.3f} &nbsp; "
                f"dense={c.dense_score:.3f} &nbsp; "
                f"bm25={c.bm25_score:.3f}</span><br>"
                f"<span style='font-size:12px; color:#555;'>"
                f"{c.authors or 'Unknown authors'}"
                f"{' &bull; ' + c.published[:4] if c.published else ''}"
                f"</span><br><br>"
                f"<span style='font-size:13px; color:#333;'>"
                f"{c.text[:350]}...</span><br><br>"
                f"<a href='{arxiv_url}' target='_blank' "
                f"style='font-size:12px; color:#2E75B6; text-decoration:none;'>"
                f"View on arXiv &rarr;</a>"
                f"</div>"
            )
            cards.append(card)

        sources_html = "\n".join(cards) if cards else "<p style='color:gray'>No sources found.</p>"

        return answer_html, badge_html, sources_html

    except Exception as exc:
        log.error("Query handler error: %s", exc)
        return (
            f"<p style='color:red'><b>Error:</b> {exc}</p>",
            "", "",
        )


# ── Gradio UI ──────────────────────────────────────────────────────────────────

def build_ui():
    """Build and return the Gradio Blocks UI."""
    try:
        import gradio as gr
    except ImportError:
        log.error("Gradio not installed. Run: pip install gradio")
        sys.exit(1)

    # Show provider info on startup
    engine = get_engine()
    provider_info = ""
    if engine:
        provider_info = (
            f"**LLM Provider:** {engine.llm_provider or 'none (generation disabled)'}  "
            f"&nbsp;|&nbsp;  **Model:** {engine.llm_model}  "
            f"&nbsp;|&nbsp;  **Chunks indexed:** {engine.n_chunks}"
        )

    with gr.Blocks(
        title=APP_TITLE,
        theme=gr.themes.Soft(primary_hue="blue"),
    ) as demo:

        gr.Markdown(f"# {APP_TITLE}")
        gr.Markdown(APP_DESC)

        if provider_info:
            gr.Markdown(provider_info)

        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="Your Question",
                    placeholder="What accuracy does chain-of-thought prompting achieve on GSM8K?",
                    lines=2,
                )
                with gr.Row():
                    submit_btn = gr.Button("Search", variant="primary")
                    clear_btn  = gr.Button("Clear")

            with gr.Column(scale=1):
                top_k_slider   = gr.Slider(
                    1, 10, value=5, step=1,
                    label="Top-K Results"
                )
                alpha_slider   = gr.Slider(
                    0.0, 1.0, value=0.5, step=0.05,
                    label="Hybrid Weight (0=BM25 only, 1=FAISS only)"
                )
                generation_tog = gr.Checkbox(
                    value=True,
                    label="Generation ON (uses LLM)"
                )

        gr.Markdown("### Answer")
        answer_out    = gr.HTML()
        grounding_out = gr.HTML()

        gr.Markdown("### Source Documents")
        sources_out = gr.HTML()

        gr.Markdown("#### Try these example questions:")
        gr.Examples(
            examples=[[q] for q in DEMO_QUESTIONS],
            inputs=[query_input],
        )

        # Wire up events
        submit_btn.click(
            fn=query_handler,
            inputs=[query_input, top_k_slider, alpha_slider, generation_tog],
            outputs=[answer_out, grounding_out, sources_out],
        )
        query_input.submit(
            fn=query_handler,
            inputs=[query_input, top_k_slider, alpha_slider, generation_tog],
            outputs=[answer_out, grounding_out, sources_out],
        )
        clear_btn.click(
            fn=lambda: ("", "", "", ""),
            outputs=[query_input, answer_out, grounding_out, sources_out],
        )

    return demo


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    """Parse arguments and launch the Gradio app."""
    parser = argparse.ArgumentParser(description="Scholar Stream Gradio App")
    parser.add_argument("--port",  type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Create public Gradio link")
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()