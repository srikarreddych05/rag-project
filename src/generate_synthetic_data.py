"""
generate_synthetic_data.py — Create a synthetic corpus for testing.

Generates realistic paper-like text across 3 topics so you can run
build_index.py and run_eval.py without downloading 250 PDFs first.
Use this to verify the pipeline end-to-end before the real data arrives.

Usage:
    python src/generate_synthetic_data.py
"""

import json
import random
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (PARSED_DIR, EVAL_DIR, ARXIV_TOPICS, SEED)

random.seed(SEED)
np.random.seed(SEED)

# ── Synthetic paper templates ──────────────────────────────────────────────────

COT_PAPERS = [
    {
        "stem": "Wei_Chain_of_Thought_Prompting__2201_11903",
        "topic": "cot_reasoning",
        "content": """Abstract
We explore how generating a chain of thought — a series of intermediate reasoning steps — significantly improves the ability of large language models to perform complex reasoning. A chain of thought can be elicited naturally through few-shot prompting using a small number of exemplars.

Introduction
Large language models have demonstrated impressive performance on a variety of tasks, yet multi-step reasoning remains challenging. Prior work has shown that few-shot prompting enables models to perform tasks with limited labeled data. However, standard prompting approaches produce flat answers without reasoning steps.

Method
We propose chain-of-thought prompting, which includes example reasoning chains in the few-shot demonstrations. Our approach requires no fine-tuning and applies to any autoregressive language model. We use 8 exemplars per prompt, each containing a question, a reasoning chain, and the final answer.

Experiments
We evaluate chain-of-thought prompting on three benchmarks: GSM8K (grade school math), MATH (competition mathematics), and CommonsenseQA. Our backbone model is PaLM 540B.

Results
Chain-of-thought prompting achieves 56.9% accuracy on GSM8K, compared to 17.9% for standard prompting — a 39 percentage point improvement. On CommonsenseQA, we observe a 73.2% accuracy versus 68.9% for standard prompting. On MATH, the method achieves 8.8% accuracy.

Ablation Study
We ablate three components: (1) reasoning steps only — performance drops by 14.2 points on GSM8K; (2) exemplars only — performance drops by 8.6 points; (3) both removed — performance collapses to 5.3%. Intermediate reasoning steps contribute more than exemplars alone.

Error Analysis
We manually inspect 50 incorrect predictions. The most common error category is calculation mistakes (47%), followed by wrong intermediate steps (31%), and hallucinated premises (22%). Multi-step arithmetic is the primary bottleneck.

Conclusion
Chain-of-thought prompting is an emergent capability that appears only in models above ~100B parameters. Future work should explore automatic chain-of-thought generation, self-consistency decoding, and applying reasoning chains to code generation tasks."""
    },
    {
        "stem": "Kojima_Large_Language_Models_are_Zero_Shot__2205_11916",
        "topic": "cot_reasoning",
        "content": """Abstract
We show that sufficiently large language models are decent zero-shot reasoners by simply adding "Let's think step by step" before each answer. This is a zero-shot CoT approach that requires no exemplar demonstrations.

Introduction
Previous chain-of-thought work relies on carefully crafted few-shot exemplars. We investigate whether zero-shot prompting with a single reasoning trigger phrase can achieve competitive performance.

Method
We use a two-stage prompting approach. In stage 1, we append "Let's think step by step" and generate a reasoning chain. In stage 2, we extract the final answer from that chain. Our backbone is GPT-3 (text-davinci-002, 175B parameters).

Experiments
We evaluate on 12 benchmarks covering arithmetic (GSM8K, MultiArith, AddSub), commonsense (CommonsenseQA, StrategyQA), and symbolic reasoning tasks. We use greedy decoding with temperature 0.

Results
Zero-shot CoT achieves 40.7% on GSM8K, compared to 10.4% for the zero-shot baseline. On MultiArith, performance improves from 17.7% to 78.7%. The approach is competitive with few-shot CoT on several benchmarks despite using no exemplars.

Ablation Study
We test 15 alternative trigger phrases. "Let's think step by step" performs best. "Let me work through this" scores 3.1 points lower. The phrase "The answer is" with no reasoning performs at baseline level.

Conclusion
Zero-shot CoT demonstrates that reasoning ability is latent in large language models and can be surfaced without exemplars. Limitations include sensitivity to exact phrasing and failure on symbolic composition tasks. Future work should develop methods to make zero-shot CoT more robust across domains."""
    },
    {
        "stem": "Wang_Self_Consistency_Improves_CoT__2203_11171",
        "topic": "cot_reasoning",
        "content": """Abstract
We propose self-consistency, a novel decoding strategy that replaces greedy decoding in chain-of-thought prompting. Instead of generating one chain, we sample a diverse set of reasoning paths and select the most consistent final answer by majority vote.

Introduction
Chain-of-thought prompting with greedy decoding is sensitive to the specific reasoning path generated. We hypothesize that sampling diverse paths and aggregating via majority vote provides a more robust answer.

Method
Given a prompt, we sample k=40 reasoning chains using temperature=0.7. We extract the final answer from each chain and return the most frequent answer. We use PaLM 540B as the backbone model.

Results
Self-consistency achieves 74.4% on GSM8K, a 17.9 percentage point improvement over chain-of-thought with greedy decoding (56.5%). On SVAMP, performance improves from 76.4% to 86.8%. On AQuA, from 47.0% to 71.5%.

Ablation Study
We vary k from 1 to 40 samples. Performance plateaus around k=20-30 for most benchmarks. Temperature 0.7 outperforms 0.5 and 1.0. Weighted voting by model confidence offers marginal improvement.

Analysis
Self-consistency is robust to chain-of-thought prompt quality — even slightly suboptimal exemplars benefit from the aggregation strategy. The method adds latency proportional to k; at k=40 inference cost increases 40x compared to greedy decoding."""
    },
]

RLHF_PAPERS = [
    {
        "stem": "Ouyang_Training_Language_Models_to_Follow__2203_02155",
        "topic": "rlhf_alignment",
        "content": """Abstract
We present InstructGPT, a model trained to follow instructions using reinforcement learning from human feedback (RLHF). Starting from GPT-3, we fine-tune with supervised learning on human-written demonstrations, then train a reward model from human preference data, and optimize with PPO.

Introduction
Large language models often produce outputs that are unhelpful, untruthful, or harmful. Aligning model behavior with human intent requires explicit human feedback beyond standard language modeling objectives.

Method
Our training pipeline has three stages. Stage 1: supervised fine-tuning (SFT) on 13,000 human-written prompt-response pairs. Stage 2: reward model training on 33,000 pairwise comparisons labeled by 40 contractors. Stage 3: PPO-based reinforcement learning with a KL penalty coefficient of 0.02 to prevent policy drift.

Reward Model
We train a 6B parameter reward model on pairwise preferences. The model achieves 72.4% accuracy on held-out preference pairs. We use Bradley-Terry model assumptions for preference probability estimation.

Results
InstructGPT (1.3B parameters) is preferred over GPT-3 (175B) by 85% of labelers on helpfulness, despite being 100x smaller. On TruthfulQA, InstructGPT achieves 41.5% truthfulness versus GPT-3's 22.9%. RLHF training reduces toxicity by 47% on the RealToxicityPrompts benchmark.

Human Evaluation
40 contractors participated in preference annotation. Inter-annotator agreement (Cohen's kappa) is 0.43, indicating moderate agreement. Labeler guidelines emphasize helpfulness, harmlessness, and honesty.

Limitations
RLHF training introduces alignment tax — InstructGPT underperforms GPT-3 on some NLP benchmarks by up to 3.6%. The model still generates harmful content in adversarial settings. Reward model hacking remains a risk at higher KL penalties."""
    },
    {
        "stem": "Bai_Training_a_Helpful_and_Harmless__2204_05862",
        "topic": "rlhf_alignment",
        "content": """Abstract
We present techniques for training helpful, harmless, and honest AI assistants using a combination of reinforcement learning from human feedback and Constitutional AI. We collect 161,000 human preference comparisons and use them to train successive reward models.

Method
We train a 52B parameter language model using two-stage RLHF. In stage 1 (Helpful), we optimize for human preference on helpfulness. In stage 2 (HH), we jointly optimize helpfulness and harmlessness. We use PPO with a KL penalty of beta=0.001.

Data Collection
We collect 161,000 pairwise preference comparisons from 500 crowd workers over 6 months. Workers are assigned to helpfulness or harmlessness annotation tracks. Average annotation time is 4.2 minutes per comparison.

Results
The HH model achieves 89.2% human preference rate over the SFT baseline on helpfulness tasks. On harmlessness evaluations, the HH model refuses 94.1% of harmful requests compared to 71.3% for the helpful-only model. Elo score improvement over SFT: +312 points.

Analysis
Constitutional AI reduces annotation burden by generating AI-written critiques and revisions. The hardest-to-mitigate safety failure mode is indirect harm — providing information that can be misused — which the model still produces 6.8% of the time under adversarial prompting."""
    },
]

EFFICIENT_PAPERS = [
    {
        "stem": "Frantar_GPTQ_Accurate_Post_Training_Quantization__2210_17323",
        "topic": "efficient_inference",
        "content": """Abstract
We present GPTQ, a new one-shot post-training quantization method for GPT-scale language models. GPTQ can quantize models with 175B parameters to INT4 precision in approximately 4 GPU hours, with negligible accuracy loss on downstream tasks.

Method
GPTQ builds on the Optimal Brain Quantization framework. For each weight matrix, we quantize weights column-by-column and update remaining unquantized weights using inverse Hessian information to compensate for quantization error. This process is training-free.

Experiments
We evaluate on OPT (125M to 175B) and BLOOM (176B) models on an NVIDIA A100 80GB GPU. We measure perplexity on WikiText-2 and C4, and accuracy on several zero-shot tasks.

Results
OPT-175B quantized to 4 bits achieves 10.86 perplexity on WikiText-2, compared to 9.34 for the FP16 baseline — a degradation of 1.52 points. Inference speed increases 3.24x at INT4 compared to FP16 on A100. Memory reduction is 73.5% (from 350 GB to 93 GB).

Ablation
We compare INT4 vs INT8 vs INT3. INT4 offers the best trade-off: INT8 yields near-zero degradation but only 2x speedup; INT3 gives 4x speedup but 6.1 perplexity point degradation on OPT-30B. The sharpest degradation cliff occurs at INT3.

Implementation
GPTQ is training-free and requires no labeled data. Quantization of a 175B model takes 4 hours on a single A100. Peak throughput for INT4 OPT-175B is 21.4 tokens/second versus 6.8 for FP16 on the same hardware."""
    },
    {
        "stem": "Dettmers_QLoRA_Efficient_Finetuning__2305_14314",
        "topic": "efficient_inference",
        "content": """Abstract
We present QLoRA, a method that enables fine-tuning of 65B parameter models on a single 48GB GPU while preserving full 16-bit performance. QLoRA uses 4-bit NormalFloat quantization, double quantization, and paged optimizers to reduce memory without sacrificing accuracy.

Method
QLoRA quantizes the frozen base model to NF4 (4-bit NormalFloat) and adds trainable LoRA adapters in BFloat16. Double quantization further reduces overhead by quantizing the quantization constants. Paged optimizers handle gradient checkpointing memory spikes.

Results
QLoRA fine-tuned Guanaco (65B) achieves 99.3% of ChatGPT performance on Vicuna benchmark with a single 48GB GPU, compared to 95.6% for standard 4-bit quantization baselines. Memory usage is 41 GB versus 780 GB for full fine-tuning.

Ablation
We ablate NF4 vs INT4 vs FP4. NF4 outperforms INT4 by 0.4 MMLU points and FP4 by 0.7 points. Double quantization saves an additional 0.4 bits per parameter with negligible accuracy impact. Removing paged optimizers causes OOM errors on longer sequences.

Hardware
All experiments run on a single NVIDIA A100 80GB or RTX 3090 24GB for smaller models. Fine-tuning Guanaco-7B takes 4.5 hours on RTX 3090; Guanaco-65B takes 24 hours on A100."""
    },
]

ALL_PAPERS = COT_PAPERS + RLHF_PAPERS + EFFICIENT_PAPERS


def generate_corpus():
    """Write synthetic .txt files to data/parsed/<topic>/."""
    for paper in ALL_PAPERS:
        out_dir = PARSED_DIR / paper["topic"]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{paper['stem']}.txt"
        out_path.write_text(paper["content"], encoding="utf-8")
        print(f"  [written] {out_path}")


def generate_eval_qa():
    """Write 30 completed QA pairs to eval/test_qa.json."""
    qa_pairs = [
        {
            "question": "What accuracy does chain-of-thought prompting achieve on GSM8K according to Wei et al.?",
            "answer": "56.9% accuracy, compared to 17.9% for standard prompting — a 39 percentage point improvement.",
            "source": "Wei_Chain_of_Thought_Prompting__2201_11903.pdf",
            "topic": "cot_reasoning",
            "difficulty": "easy"
        },
        {
            "question": "How many few-shot exemplars are used in the chain-of-thought prompts by Wei et al.?",
            "answer": "8 exemplars per prompt, each containing a question, a reasoning chain, and the final answer.",
            "source": "Wei_Chain_of_Thought_Prompting__2201_11903.pdf",
            "topic": "cot_reasoning",
            "difficulty": "easy"
        },
        {
            "question": "According to the ablation study in Wei et al., does removing reasoning steps or removing exemplars hurt performance more on GSM8K?",
            "answer": "Removing reasoning steps hurts more — performance drops 14.2 points versus 8.6 points for removing exemplars.",
            "source": "Wei_Chain_of_Thought_Prompting__2201_11903.pdf",
            "topic": "cot_reasoning",
            "difficulty": "hard"
        },
        {
            "question": "What is the most common error category identified in Wei et al.'s error analysis of chain-of-thought predictions?",
            "answer": "Calculation mistakes, accounting for 47% of incorrect predictions, followed by wrong intermediate steps at 31%.",
            "source": "Wei_Chain_of_Thought_Prompting__2201_11903.pdf",
            "topic": "cot_reasoning",
            "difficulty": "hard"
        },
        {
            "question": "What trigger phrase is used in Kojima et al.'s zero-shot CoT method?",
            "answer": "The phrase 'Let's think step by step' is appended before each answer.",
            "source": "Kojima_Large_Language_Models_are_Zero_Shot__2205_11916.pdf",
            "topic": "cot_reasoning",
            "difficulty": "easy"
        },
        {
            "question": "What accuracy does zero-shot CoT achieve on GSM8K in Kojima et al.?",
            "answer": "40.7% accuracy, compared to 10.4% for the standard zero-shot baseline.",
            "source": "Kojima_Large_Language_Models_are_Zero_Shot__2205_11916.pdf",
            "topic": "cot_reasoning",
            "difficulty": "easy"
        },
        {
            "question": "What decoding temperature is used in Kojima et al.'s zero-shot CoT experiments?",
            "answer": "Temperature 0 (greedy decoding) is used for all experiments.",
            "source": "Kojima_Large_Language_Models_are_Zero_Shot__2205_11916.pdf",
            "topic": "cot_reasoning",
            "difficulty": "medium"
        },
        {
            "question": "How many reasoning chains does self-consistency sample per query in Wang et al.?",
            "answer": "k=40 reasoning chains are sampled using temperature=0.7, with the most frequent final answer selected.",
            "source": "Wang_Self_Consistency_Improves_CoT__2203_11171.pdf",
            "topic": "cot_reasoning",
            "difficulty": "easy"
        },
        {
            "question": "What is the accuracy of self-consistency on GSM8K compared to standard chain-of-thought in Wang et al.?",
            "answer": "74.4% versus 56.5% — a 17.9 percentage point improvement over chain-of-thought with greedy decoding.",
            "source": "Wang_Self_Consistency_Improves_CoT__2203_11171.pdf",
            "topic": "cot_reasoning",
            "difficulty": "medium"
        },
        {
            "question": "At what value of k does performance plateau in the self-consistency ablation by Wang et al.?",
            "answer": "Performance plateaus around k=20-30 samples for most benchmarks.",
            "source": "Wang_Self_Consistency_Improves_CoT__2203_11171.pdf",
            "topic": "cot_reasoning",
            "difficulty": "medium"
        },
        {
            "question": "What KL divergence penalty coefficient is used in InstructGPT's PPO training?",
            "answer": "A KL penalty coefficient of 0.02 is used to prevent the policy from drifting from the SFT model.",
            "source": "Ouyang_Training_Language_Models_to_Follow__2203_02155.pdf",
            "topic": "rlhf_alignment",
            "difficulty": "hard"
        },
        {
            "question": "How many pairwise comparisons were collected to train the reward model in InstructGPT?",
            "answer": "33,000 pairwise comparisons labeled by 40 contractors.",
            "source": "Ouyang_Training_Language_Models_to_Follow__2203_02155.pdf",
            "topic": "rlhf_alignment",
            "difficulty": "easy"
        },
        {
            "question": "What percentage of human labelers prefer InstructGPT over GPT-3 on helpfulness?",
            "answer": "85% of labelers prefer InstructGPT (1.3B parameters) over GPT-3 (175B parameters).",
            "source": "Ouyang_Training_Language_Models_to_Follow__2203_02155.pdf",
            "topic": "rlhf_alignment",
            "difficulty": "easy"
        },
        {
            "question": "What is the inter-annotator agreement score (Cohen's kappa) for the InstructGPT human evaluation?",
            "answer": "Cohen's kappa is 0.43, indicating moderate inter-annotator agreement.",
            "source": "Ouyang_Training_Language_Models_to_Follow__2203_02155.pdf",
            "topic": "rlhf_alignment",
            "difficulty": "medium"
        },
        {
            "question": "By what percentage does RLHF training reduce toxicity in InstructGPT versus the GPT-3 baseline?",
            "answer": "RLHF training reduces toxicity by 47% on the RealToxicityPrompts benchmark.",
            "source": "Ouyang_Training_Language_Models_to_Follow__2203_02155.pdf",
            "topic": "rlhf_alignment",
            "difficulty": "medium"
        },
        {
            "question": "How many human preference comparisons were collected for training the Bai et al. Constitutional AI model?",
            "answer": "161,000 pairwise preference comparisons from 500 crowd workers collected over 6 months.",
            "source": "Bai_Training_a_Helpful_and_Harmless__2204_05862.pdf",
            "topic": "rlhf_alignment",
            "difficulty": "easy"
        },
        {
            "question": "What KL penalty coefficient is used in the Bai et al. RLHF training?",
            "answer": "A KL penalty of beta=0.001 is used during PPO optimization.",
            "source": "Bai_Training_a_Helpful_and_Harmless__2204_05862.pdf",
            "topic": "rlhf_alignment",
            "difficulty": "hard"
        },
        {
            "question": "What percentage of harmful requests does the HH model in Bai et al. refuse?",
            "answer": "94.1% of harmful requests are refused, compared to 71.3% for the helpful-only model.",
            "source": "Bai_Training_a_Helpful_and_Harmless__2204_05862.pdf",
            "topic": "rlhf_alignment",
            "difficulty": "medium"
        },
        {
            "question": "What is the hardest-to-mitigate safety failure mode identified in Bai et al.?",
            "answer": "Indirect harm — providing information that can be misused — which occurs 6.8% of the time under adversarial prompting.",
            "source": "Bai_Training_a_Helpful_and_Harmless__2204_05862.pdf",
            "topic": "rlhf_alignment",
            "difficulty": "hard"
        },
        {
            "question": "What is the Elo score improvement of the HH model over the SFT baseline in Bai et al.?",
            "answer": "+312 Elo points improvement over the SFT baseline.",
            "source": "Bai_Training_a_Helpful_and_Harmless__2204_05862.pdf",
            "topic": "rlhf_alignment",
            "difficulty": "medium"
        },
        {
            "question": "To how many bits does GPTQ quantize the model weights, and is it training-free?",
            "answer": "GPTQ quantizes to INT4 (4 bits) and is entirely training-free, requiring no labeled data.",
            "source": "Frantar_GPTQ_Accurate_Post_Training_Quantization__2210_17323.pdf",
            "topic": "efficient_inference",
            "difficulty": "easy"
        },
        {
            "question": "What is the inference speedup achieved by GPTQ INT4 compared to FP16 on the A100 GPU?",
            "answer": "3.24x speedup at INT4 compared to FP16 on an NVIDIA A100.",
            "source": "Frantar_GPTQ_Accurate_Post_Training_Quantization__2210_17323.pdf",
            "topic": "efficient_inference",
            "difficulty": "medium"
        },
        {
            "question": "At what bit-width does GPTQ show the sharpest perplexity degradation cliff?",
            "answer": "At INT3, where perplexity degrades by 6.1 points on OPT-30B — the sharpest observed cliff.",
            "source": "Frantar_GPTQ_Accurate_Post_Training_Quantization__2210_17323.pdf",
            "topic": "efficient_inference",
            "difficulty": "hard"
        },
        {
            "question": "What is the memory reduction percentage achieved by GPTQ on OPT-175B?",
            "answer": "73.5% memory reduction — from 350 GB (FP16) to 93 GB (INT4).",
            "source": "Frantar_GPTQ_Accurate_Post_Training_Quantization__2210_17323.pdf",
            "topic": "efficient_inference",
            "difficulty": "medium"
        },
        {
            "question": "What is the peak inference throughput of INT4 OPT-175B using GPTQ?",
            "answer": "21.4 tokens per second versus 6.8 tokens per second for FP16 on the same A100 hardware.",
            "source": "Frantar_GPTQ_Accurate_Post_Training_Quantization__2210_17323.pdf",
            "topic": "efficient_inference",
            "difficulty": "medium"
        },
        {
            "question": "What quantization format does QLoRA use for the frozen base model?",
            "answer": "NF4 (4-bit NormalFloat) quantization for the frozen base model weights.",
            "source": "Dettmers_QLoRA_Efficient_Finetuning__2305_14314.pdf",
            "topic": "efficient_inference",
            "difficulty": "easy"
        },
        {
            "question": "How does QLoRA's memory usage compare to full fine-tuning for a 65B model?",
            "answer": "QLoRA uses 41 GB versus 780 GB for full fine-tuning — a roughly 95% memory reduction.",
            "source": "Dettmers_QLoRA_Efficient_Finetuning__2305_14314.pdf",
            "topic": "efficient_inference",
            "difficulty": "medium"
        },
        {
            "question": "By how many MMLU points does NF4 outperform INT4 in the QLoRA ablation?",
            "answer": "NF4 outperforms INT4 by 0.4 MMLU points and FP4 by 0.7 points.",
            "source": "Dettmers_QLoRA_Efficient_Finetuning__2305_14314.pdf",
            "topic": "efficient_inference",
            "difficulty": "hard"
        },
        {
            "question": "What percentage of ChatGPT performance does QLoRA Guanaco-65B achieve on the Vicuna benchmark?",
            "answer": "99.3% of ChatGPT performance on the Vicuna benchmark.",
            "source": "Dettmers_QLoRA_Efficient_Finetuning__2305_14314.pdf",
            "topic": "efficient_inference",
            "difficulty": "medium"
        },
        {
            "question": "How long does it take to fine-tune Guanaco-65B using QLoRA on a single A100 GPU?",
            "answer": "24 hours on a single NVIDIA A100 80GB GPU.",
            "source": "Dettmers_QLoRA_Efficient_Finetuning__2305_14314.pdf",
            "topic": "efficient_inference",
            "difficulty": "easy"
        },
    ]

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_DIR / "test_qa.json"
    with open(out_path, "w") as f:
        json.dump(qa_pairs, f, indent=2)
    print(f"  [written] {out_path}  ({len(qa_pairs)} QA pairs)")


def main():
    print("[synthetic] Generating synthetic corpus ...")
    generate_corpus()
    print("[synthetic] Generating filled eval QA set ...")
    generate_eval_qa()
    print("\n[synthetic] Done. Now run:")
    print("  python src/build_index.py")
    print("  python src/run_eval.py")
    print("  python src/run_eval.py --ablation")


if __name__ == "__main__":
    main()
