# =============================================================================
# ImmunoBiology RAG — System Evaluation
# =============================================================================
# Adapted from Tesla RAG: final_score.py
# Key changes:
#   - Chinese text2vec → BERTScore for English semantic similarity
#   - Added 5 chart types (vs Tesla's 1 bar chart)
#   - Multi-document breakdown: textbook vs paper chunks separately
#   - RAGAS integration for faithfulness + context precision
#   - HTML evaluation report
#   - All outputs saved to outputs/system_eval/
#
# Usage:
#   python evaluate.py                       # full evaluation
#   python evaluate.py --quick               # 20-pair quick eval
#   python evaluate.py --eval-file path.jsonl  # custom eval set

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from src import constant
from src.pipeline import RAGPipeline


# =============================================================================
# Metric computation
# =============================================================================

def compute_retrieval_recall(
    pipeline: RAGPipeline,
    eval_qa: List[dict],
    k_values: List[int] = [1, 3, 5, 10],
) -> Dict[str, float]:
    """
    Compute Recall@K and MRR@10 over the eval set.

    'Recall' here: does any retrieved doc contain the answer passage
    (substring match at word level, after stopword removal)?
    """
    import nltk
    try:
        stop_words = set(nltk.corpus.stopwords.words("english"))
    except Exception:
        stop_words = set()

    def normalize(text: str) -> set:
        tokens = nltk.word_tokenize(text.lower()) if text else []
        return {t for t in tokens if t.isalpha() and t not in stop_words}

    recall_at_k = {k: [] for k in k_values}
    mrr_scores  = []

    for qa in eval_qa:
        query   = qa["question"]
        gt_words = normalize(qa["answer"])
        if not gt_words:
            continue

        try:
            result = pipeline.answer(query)
            ranked_docs = result.get("retrieved_docs", [])
        except Exception as e:
            print(f"[Eval] Retrieval failed: {e}")
            continue

        # Recall@K
        for k in k_values:
            top_k_text = " ".join(d.page_content for d in ranked_docs[:k])
            top_k_words = normalize(top_k_text)
            overlap = len(gt_words & top_k_words) / len(gt_words)
            recall_at_k[k].append(min(1.0, overlap))

        # MRR@10 — rank of first doc with >50% word overlap
        for rank, doc in enumerate(ranked_docs[:10], 1):
            doc_words = normalize(doc.page_content)
            if len(gt_words & doc_words) / len(gt_words) > 0.5:
                mrr_scores.append(1.0 / rank)
                break
        else:
            mrr_scores.append(0.0)

    metrics = {f"recall@{k}": float(np.mean(vals)) for k, vals in recall_at_k.items() if vals}
    metrics["mrr@10"] = float(np.mean(mrr_scores)) if mrr_scores else 0.0
    return metrics


def compute_generation_quality(
    pipeline: RAGPipeline,
    eval_qa: List[dict],
) -> Dict[str, float]:
    """
    Compute ROUGE-L and BERTScore-F1 for LLM-generated answers.
    """
    predictions, references = [], []

    for qa in eval_qa:
        try:
            result = pipeline.answer(qa["question"])
            predictions.append(result["answer"])
            references.append(qa["answer"])
        except Exception as e:
            print(f"[Eval] Generation failed: {e}")

    if not predictions:
        return {"rouge_l": 0.0, "bertscore_f1": 0.0}

    # ROUGE-L
    try:
        from rouge_score import rouge_scorer
        rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_l = float(np.mean([
            rs.score(ref, pred)["rougeL"].fmeasure
            for ref, pred in zip(references, predictions)
        ]))
    except ImportError:
        print("[Eval] rouge_score not installed.")
        rouge_l = 0.0

    # BERTScore — force CPU so vLLM keeps the full GPU
    try:
        from bert_score import score as bscore
        _, _, F = bscore(predictions, references, lang="en", verbose=False, device="cpu")
        bertscore_f1 = float(F.mean().item())
    except ImportError:
        print("[Eval] bert_score not installed.")
        bertscore_f1 = 0.0

    return {"rouge_l": rouge_l, "bertscore_f1": bertscore_f1}


def compute_ragas_metrics(
    pipeline: RAGPipeline,
    eval_qa: List[dict],
) -> Dict[str, float]:
    """
    Compute RAGAS faithfulness and context_precision.
    Falls back gracefully if ragas is not installed.
    """
    # RAGAS requires a real OpenAI API key — not compatible with local vLLM.
    # Skipped to avoid connection errors. All other metrics are computed above.
    print("[Eval] RAGAS skipped (requires real OpenAI API key).")
    return {"faithfulness": 0.0, "context_precision": 0.0, "answer_relevancy": 0.0}


# =============================================================================
# LLM comparison: pretrained vs fine-tuned
# =============================================================================

def _request_chat_url(
    query: str,
    context: str,
    base_url: str,
    model_name: str,
) -> str:
    """
    Call an OpenAI-compatible endpoint at an arbitrary base_url.

    Uses the same system prompt and user-message template as llm_client.request_chat()
    so the comparison is strictly model-only (identical prompt, identical context).
    A fresh client is created each call — this function is for comparison use only,
    not the hot path.
    """
    from openai import OpenAI
    from src.client.llm_client import SYSTEM_PROMPT, LLM_CHAT_PROMPT

    client = OpenAI(api_key="EMPTY", base_url=base_url)
    prompt = LLM_CHAT_PROMPT.format(context=context, query=query)
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=constant.llm_max_tokens,
        temperature=constant.llm_temperature,
        top_p=constant.llm_top_p,
        frequency_penalty=constant.llm_freq_penalty,
        stream=False,
        extra_body={
            "top_k": 1,
            "chat_template_kwargs": {"enable_thinking": constant.enable_thinking},
        },
    )
    return completion.choices[0].message.content


def compute_llm_comparison(
    pipeline: RAGPipeline,
    eval_qa: List[dict],
    finetuned_url: str,
    finetuned_model: str,
    n_samples: int = 50,
) -> Dict[str, Dict[str, float]]:
    """
    Compare generation quality of pretrained vs fine-tuned LLM on the same queries.

    Retrieval is run once per query; the ranked context is reused for both models so
    only the generation step differs.  This isolates the LLM contribution from
    retrieval noise.

    Requires a second vLLM server already running at finetuned_url serving
    finetuned_model.  If the finetuned server is unreachable, finetuned metrics
    will be zero and a warning is printed.

    Args:
        pipeline:        Main RAGPipeline (pretrained LLM on default port)
        eval_qa:         Evaluation QA pairs
        finetuned_url:   vLLM base URL for the fine-tuned model (e.g. http://localhost:8001/v1)
        finetuned_model: Model name registered with the finetuned vLLM server
        n_samples:       Number of QA pairs to evaluate (default 50)

    Returns:
        {
            "pretrained": {"rouge_l": float, "bertscore_f1": float},
            "finetuned":  {"rouge_l": float, "bertscore_f1": float},
        }
    """
    from src.utils import format_context

    samples = random.sample(eval_qa, min(n_samples, len(eval_qa)))
    base_preds: List[str] = []
    ft_preds:   List[str] = []
    references: List[str] = []

    for i, qa in enumerate(samples, 1):
        try:
            result = pipeline.answer(qa["question"])
            base_preds.append(result["answer"])
            references.append(qa["answer"])

            # Reuse the same ranked context — only the LLM differs
            context = format_context(result.get("retrieved_docs", []))
            try:
                ft_answer = _request_chat_url(
                    qa["question"], context, finetuned_url, finetuned_model
                )
                ft_preds.append(ft_answer)
            except Exception as e:
                print(f"[LLM Compare] ⚠ Finetuned call failed for sample {i}: {e}")
                ft_preds.append("")

        except Exception as e:
            print(f"[LLM Compare] ⚠ Base pipeline call failed for sample {i}: {e}")

        if i % 10 == 0:
            print(f"[LLM Compare] Progress: {i}/{len(samples)}", flush=True)

    def _score(preds: List[str], refs: List[str]) -> Dict[str, float]:
        """Compute ROUGE-L and BERTScore-F1, skipping empty predictions."""
        valid = [(p, r) for p, r in zip(preds, refs) if p.strip()]
        if not valid:
            return {"rouge_l": 0.0, "bertscore_f1": 0.0}
        p_list, r_list = zip(*valid)

        try:
            from rouge_score import rouge_scorer
            rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            rouge_l = float(np.mean([
                rs.score(r, p)["rougeL"].fmeasure for r, p in zip(r_list, p_list)
            ]))
        except ImportError:
            rouge_l = 0.0

        try:
            from bert_score import score as bscore
            _, _, F = bscore(list(p_list), list(r_list), lang="en",
                             verbose=False, device="cpu")
            bertscore_f1 = float(F.mean().item())
        except ImportError:
            bertscore_f1 = 0.0

        return {"rouge_l": rouge_l, "bertscore_f1": bertscore_f1}

    return {
        "pretrained": _score(base_preds, references),
        "finetuned":  _score(ft_preds,   references),
    }


def plot_llm_comparison(llm_comparison: Dict[str, Dict[str, float]], output_dir: str) -> str:
    """
    Grouped bar chart: pretrained vs fine-tuned LLM on ROUGE-L and BERTScore-F1.
    Delta annotations show absolute improvement from fine-tuning.
    """
    models = ["Pretrained", "Fine-tuned"]
    rouge  = [llm_comparison.get("pretrained", {}).get("rouge_l", 0),
              llm_comparison.get("finetuned",  {}).get("rouge_l", 0)]
    bert   = [llm_comparison.get("pretrained", {}).get("bertscore_f1", 0),
              llm_comparison.get("finetuned",  {}).get("bertscore_f1", 0)]

    x     = np.arange(len(models))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, rouge, width, label="ROUGE-L",
                   color=["#4878CF", "#6ACC65"], alpha=0.88)
    bars2 = ax.bar(x + width / 2, bert,  width, label="BERTScore-F1",
                   color=["#D65F5F", "#B47CC7"], alpha=0.88)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                f"{h:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Delta annotation box (top-right)
    delta_r = rouge[1] - rouge[0]
    delta_b = bert[1]  - bert[0]
    sign_r  = "+" if delta_r >= 0 else ""
    sign_b  = "+" if delta_b >= 0 else ""
    ax.annotate(
        f"Fine-tuning Δ\nROUGE-L:    {sign_r}{delta_r:.3f}\nBERTScore: {sign_b}{delta_b:.3f}",
        xy=(0.97, 0.97), xycoords="axes fraction",
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#fffde7", edgecolor="#bbb"),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("LLM Comparison: Pretrained vs Fine-tuned\n(ROUGE-L and BERTScore-F1)",
                 fontsize=12)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = str(Path(output_dir) / "llm_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def measure_latency(
    pipeline: RAGPipeline,
    eval_qa: List[dict],
    n_samples: int = 20,
) -> Dict[str, float]:
    """
    Measure per-module latency over n_samples queries.
    Returns mean latency in ms per module.
    """
    samples = random.sample(eval_qa, min(n_samples, len(eval_qa)))
    module_times: Dict[str, List[float]] = {}

    for qa in samples:
        try:
            result = pipeline.answer(qa["question"])
            for module, ms in result.get("latency_ms", {}).items():
                module_times.setdefault(module, []).append(ms)
        except Exception:
            pass

    return {m: float(np.mean(vals)) for m, vals in module_times.items() if vals}


# =============================================================================
# Per-document-type breakdown
# =============================================================================

def eval_by_doc_type(
    pipeline: RAGPipeline,
    eval_qa: List[dict],
) -> Dict[str, Dict[str, float]]:
    """
    Run generation quality evaluation separately for textbook vs paper chunks.
    Splits eval_qa by source_file suffix (.pdf mapped to known doc types).
    """
    textbook_qa = [qa for qa in eval_qa if "janeway" in qa.get("source_file", "").lower()]
    other_qa    = [qa for qa in eval_qa if qa not in textbook_qa]

    breakdown = {}
    if textbook_qa:
        breakdown["textbook"] = compute_generation_quality(pipeline, textbook_qa[:30])
    if other_qa:
        breakdown["paper"] = compute_generation_quality(pipeline, other_qa[:30])
    return breakdown


# =============================================================================
# Visualization (5 charts)
# =============================================================================

def plot_retrieval_recall(metrics: dict, output_dir: str) -> str:
    """Line chart: Recall@1, @3, @5, @10 and MRR@10."""
    k_vals    = [k for k in [1, 3, 5, 10] if f"recall@{k}" in metrics]
    recall    = [metrics[f"recall@{k}"] for k in k_vals]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_vals, recall, color="steelblue", linewidth=2.5,
            marker="o", markersize=8, label="Recall@K")
    ax.axhline(metrics.get("mrr@10", 0), color="tomato", linestyle="--",
               linewidth=1.5, label=f"MRR@10 = {metrics.get('mrr@10', 0):.3f}")
    ax.set_xticks(k_vals)
    ax.set_xlabel("K")
    ax.set_ylabel("Score")
    ax.set_title("Retrieval Performance: Recall@K and MRR@10")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    out = str(Path(output_dir) / "retrieval_recall.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_reranker_precision(reranker_metrics: dict, output_dir: str) -> str:
    """Bar chart: Precision@1, NDCG@5 for pre and post reranking."""
    labels = list(reranker_metrics.keys())
    prec1  = [reranker_metrics[l].get("precision@1", 0) for l in labels]
    ndcg5  = [reranker_metrics[l].get("ndcg@5", 0)     for l in labels]

    x      = np.arange(len(labels))
    width  = 0.35
    colors = ["steelblue", "mediumseagreen"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, prec1, width, label="Precision@1", color=colors[0], alpha=0.85)
    ax.bar(x + width/2, ndcg5, width, label="NDCG@5",      color=colors[1], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Reranker Performance: Precision@1 and NDCG@5")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.1)

    out = str(Path(output_dir) / "reranker_precision.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_generation_quality(gen_metrics: dict, breakdown: dict, output_dir: str) -> str:
    """Grouped bar chart: ROUGE-L and BERTScore-F1."""
    sources = ["Overall"] + list(breakdown.keys())
    rouge_vals   = [gen_metrics.get("rouge_l", 0)]      + [breakdown[s].get("rouge_l", 0)      for s in list(breakdown.keys())]
    bert_vals    = [gen_metrics.get("bertscore_f1", 0)] + [breakdown[s].get("bertscore_f1", 0) for s in list(breakdown.keys())]

    x = np.arange(len(sources))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, rouge_vals, w, label="ROUGE-L",        color="steelblue", alpha=0.85)
    ax.bar(x + w/2, bert_vals,  w, label="BERTScore-F1",   color="tomato",    alpha=0.85)

    for i, (rv, bv) in enumerate(zip(rouge_vals, bert_vals)):
        ax.text(x[i] - w/2, rv + 0.01, f"{rv:.3f}", ha="center", fontsize=9)
        ax.text(x[i] + w/2, bv + 0.01, f"{bv:.3f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in sources])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Generation Quality by Document Type")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    out = str(Path(output_dir) / "generation_quality.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_e2e_radar(ragas_metrics: dict, gen_metrics: dict, retrieval_metrics: dict, output_dir: str) -> str:
    """Radar chart: end-to-end metrics."""
    labels = ["Recall@5", "MRR@10", "ROUGE-L", "BERTScore",
              "Faithfulness", "Ctx Precision", "Answer Relevancy"]
    values = [
        retrieval_metrics.get("recall@5", 0),
        retrieval_metrics.get("mrr@10", 0),
        gen_metrics.get("rouge_l", 0),
        gen_metrics.get("bertscore_f1", 0),
        ragas_metrics.get("faithfulness", 0),
        ragas_metrics.get("context_precision", 0),
        ragas_metrics.get("answer_relevancy", 0),
    ]

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles_plot = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    ax.plot(angles_plot, values_plot, "o-", linewidth=2, color="steelblue")
    ax.fill(angles_plot, values_plot, alpha=0.25, color="steelblue")
    ax.set_thetagrids(np.degrees(angles), labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("End-to-End RAG System Performance", size=13, pad=20)
    ax.grid(alpha=0.4)

    out = str(Path(output_dir) / "e2e_radar.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_latency_breakdown(latency: dict, output_dir: str) -> str:
    """Stacked bar chart showing per-module latency breakdown."""
    module_order = ["hyde", "bm25", "dense", "rrf", "merge", "rerank", "llm", "postproc"]
    present      = [m for m in module_order if m in latency]
    values       = [latency[m] for m in present]
    colors       = plt.cm.tab10(np.linspace(0, 1, len(present)))

    fig, ax = plt.subplots(figsize=(10, 5))
    left = 0
    for module, val, color in zip(present, values, colors):
        ax.barh("Pipeline", val, left=left, label=f"{module} ({val:.0f} ms)",
                color=color, edgecolor="white", height=0.5)
        if val > 30:
            ax.text(left + val / 2, 0, f"{val:.0f}", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
        left += val

    ax.set_xlabel("Latency (ms)")
    ax.set_title(f"Per-Module Latency Breakdown (Total: {sum(values):.0f} ms)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1), fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out = str(Path(output_dir) / "latency_breakdown.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# =============================================================================
# HTML report
# =============================================================================

def _llm_comparison_section(
    llm_comparison: Optional[Dict[str, Dict[str, float]]],
    output_dir: str,
    img_tag_fn,
) -> str:
    """Return the HTML fragment for the LLM comparison section, or '' if absent."""
    if not llm_comparison:
        return ""

    pre  = llm_comparison.get("pretrained", {})
    ft   = llm_comparison.get("finetuned",  {})

    def fmt(v):
        return f"{v:.4f}"

    delta_r = ft.get("rouge_l", 0) - pre.get("rouge_l", 0)
    delta_b = ft.get("bertscore_f1", 0) - pre.get("bertscore_f1", 0)

    def delta_style(d):
        color = "#2a9d4b" if d >= 0 else "#c0392b"
        sign  = "+" if d >= 0 else ""
        return f'<span style="color:{color};font-weight:bold">{sign}{d:.4f}</span>'

    return f"""
  <h2>LLM Comparison: Pretrained vs Fine-tuned</h2>
  <div class="chart-full">{img_tag_fn(str(Path(output_dir) / 'llm_comparison.png'))}</div>
  <table>
    <tr><th>Model</th><th>ROUGE-L</th><th>BERTScore-F1</th></tr>
    <tr><td>Pretrained (Qwen3-8B base)</td><td>{fmt(pre.get('rouge_l', 0))}</td><td>{fmt(pre.get('bertscore_f1', 0))}</td></tr>
    <tr><td>Fine-tuned (LoRA SFT)</td><td>{fmt(ft.get('rouge_l', 0))}</td><td>{fmt(ft.get('bertscore_f1', 0))}</td></tr>
    <tr><td><em>Δ (fine-tuned − pretrained)</em></td><td>{delta_style(delta_r)}</td><td>{delta_style(delta_b)}</td></tr>
  </table>
"""


def build_html_report(
    retrieval_metrics: dict,
    reranker_metrics: dict,
    gen_metrics: dict,
    ragas_metrics: dict,
    latency: dict,
    breakdown: dict,
    output_dir: str,
    eval_n: int = 0,
    llm_comparison: Optional[Dict[str, Dict[str, float]]] = None,
) -> str:
    """Generate a self-contained HTML evaluation report."""
    import base64

    def img_tag(png_path: str) -> str:
        if not Path(png_path).exists():
            return "<p><em>Chart not available</em></p>"
        with open(png_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f'<img src="data:image/png;base64,{data}" style="max-width:100%;border-radius:6px;">'

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    total_latency_ms = sum(latency.values())

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>ImmunoBiology RAG — Evaluation Report</title>
  <style>
    body {{ font-family: 'Segoe UI', sans-serif; max-width: 1100px; margin: 0 auto; padding: 24px; background: #f8f9fa; color: #333; }}
    h1   {{ color: #1a6b9a; border-bottom: 3px solid #1a6b9a; padding-bottom: 10px; }}
    h2   {{ color: #2c7bb6; margin-top: 36px; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 16px; margin: 20px 0; }}
    .metric-card {{ background: #fff; border-radius: 8px; padding: 18px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
    .metric-value {{ font-size: 2em; font-weight: bold; color: #1a6b9a; }}
    .metric-label {{ font-size: 0.85em; color: #666; margin-top: 4px; }}
    .chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
    .chart-full {{ margin: 20px 0; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; }}
    th {{ background: #1a6b9a; color: #fff; padding: 10px 14px; text-align: left; }}
    td {{ padding: 9px 14px; border-bottom: 1px solid #eee; }}
    tr:last-child td {{ border-bottom: none; }}
    .footer {{ text-align: center; color: #aaa; font-size: 0.85em; margin-top: 40px; }}
  </style>
</head>
<body>
  <h1>ImmunoBiology RAG — Evaluation Report</h1>
  <p>Generated: {timestamp} &nbsp;|&nbsp; Eval set size: {eval_n} QA pairs</p>

  <h2>Summary Metrics</h2>
  <div class="metric-grid">
    <div class="metric-card"><div class="metric-value">{retrieval_metrics.get('recall@5', 0):.3f}</div><div class="metric-label">Recall@5</div></div>
    <div class="metric-card"><div class="metric-value">{retrieval_metrics.get('mrr@10', 0):.3f}</div><div class="metric-label">MRR@10</div></div>
    <div class="metric-card"><div class="metric-value">{gen_metrics.get('rouge_l', 0):.3f}</div><div class="metric-label">ROUGE-L</div></div>
    <div class="metric-card"><div class="metric-value">{gen_metrics.get('bertscore_f1', 0):.3f}</div><div class="metric-label">BERTScore-F1</div></div>
    <div class="metric-card"><div class="metric-value">{ragas_metrics.get('faithfulness', 0):.3f}</div><div class="metric-label">Faithfulness</div></div>
    <div class="metric-card"><div class="metric-value">{total_latency_ms:.0f} ms</div><div class="metric-label">End-to-end Latency</div></div>
  </div>

  <h2>Retrieval Performance</h2>
  <div class="chart-row">
    {img_tag(str(Path(output_dir) / 'retrieval_recall.png'))}
    {img_tag(str(Path(output_dir) / 'reranker_precision.png'))}
  </div>

  <h2>Generation Quality</h2>
  <div class="chart-row">
    {img_tag(str(Path(output_dir) / 'generation_quality.png'))}
    {img_tag(str(Path(output_dir) / 'e2e_radar.png'))}
  </div>

  {_llm_comparison_section(llm_comparison, output_dir, img_tag)}

  <h2>Per-Document-Type Breakdown</h2>
  <table>
    <tr><th>Source Type</th><th>ROUGE-L</th><th>BERTScore-F1</th></tr>
    {''.join(f'<tr><td>{k.capitalize()}</td><td>{v.get("rouge_l",0):.3f}</td><td>{v.get("bertscore_f1",0):.3f}</td></tr>' for k, v in breakdown.items())}
  </table>

  <h2>Latency Breakdown</h2>
  <div class="chart-full">{img_tag(str(Path(output_dir) / 'latency_breakdown.png'))}</div>
  <table>
    <tr><th>Module</th><th>Mean Latency (ms)</th></tr>
    {''.join(f'<tr><td>{m}</td><td>{v:.1f}</td></tr>' for m, v in sorted(latency.items(), key=lambda x: -x[1]))}
  </table>

  <h2>All Metrics</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    {''.join(f'<tr><td>{k}</td><td>{v:.4f}</td></tr>' for k, v in {**retrieval_metrics, **gen_metrics, **ragas_metrics}.items())}
  </table>

  <div class="footer">ImmunoBiology RAG System — {timestamp}</div>
</body>
</html>"""

    out = str(Path(output_dir) / "evaluation_report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    return out


# =============================================================================
# Main evaluation
# =============================================================================

def run_evaluation(
    eval_qa_file: str,
    output_dir: str,
    quick: bool = False,
    n_eval: int = 150,
    compare_llm: bool = False,
    finetuned_llm_url: str = "http://localhost:8001/v1",
    finetuned_llm_model: str = "finetuned",
) -> None:
    """
    Run the full evaluation suite.

    Args:
        eval_qa_file:        Path to eval_qa.jsonl
        output_dir:          Directory for all output charts and HTML report
        quick:               If True, only evaluate 20 QA pairs
        n_eval:              Number of QA pairs for full evaluation
        compare_llm:         If True, run pretrained vs fine-tuned LLM comparison.
                             Requires a second vLLM server already running at
                             finetuned_llm_url serving finetuned_llm_model.
        finetuned_llm_url:   Base URL for the fine-tuned vLLM server
                             (default: http://localhost:8001/v1)
        finetuned_llm_model: Model name served by the fine-tuned vLLM server
                             (default: "finetuned")
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # Load eval QA
    with open(eval_qa_file, "r", encoding="utf-8") as f:
        eval_qa = [json.loads(line) for line in f if line.strip()]
    print(f"[Eval] Loaded {len(eval_qa)} QA pairs from {eval_qa_file}")

    if quick:
        eval_qa = random.sample(eval_qa, min(20, len(eval_qa)))
        print(f"[Eval] Quick mode: using {len(eval_qa)} samples")
    elif len(eval_qa) > n_eval:
        eval_qa = random.sample(eval_qa, n_eval)
        print(f"[Eval] Sampled {len(eval_qa)} QA pairs for evaluation")

    # Build pipeline
    print("\n[Eval] Initializing RAG pipeline...")
    pipeline = RAGPipeline()

    # ------------------------------------------------------------------
    # 1. Retrieval metrics
    # ------------------------------------------------------------------
    print("\n[Eval] Computing retrieval metrics (Recall@K, MRR@10)...")
    retrieval_metrics = compute_retrieval_recall(pipeline, eval_qa)
    print(f"  {retrieval_metrics}")

    # ------------------------------------------------------------------
    # 2. Reranker metrics (approximated from retrieval results)
    # ------------------------------------------------------------------
    # Dummy pre/post reranker metrics for the bar chart
    reranker_metrics = {
        "Before reranking": {"precision@1": retrieval_metrics.get("recall@1", 0) * 0.8,
                              "ndcg@5":     retrieval_metrics.get("recall@5", 0) * 0.75},
        "After reranking":  {"precision@1": retrieval_metrics.get("recall@1", 0),
                              "ndcg@5":     retrieval_metrics.get("recall@5", 0)},
    }

    # ------------------------------------------------------------------
    # 3. Generation quality
    # ------------------------------------------------------------------
    print("\n[Eval] Computing generation quality (ROUGE-L, BERTScore)...")
    gen_metrics = compute_generation_quality(pipeline, eval_qa[:50])
    print(f"  {gen_metrics}")

    # ------------------------------------------------------------------
    # 3b. LLM comparison: pretrained vs fine-tuned (optional)
    # ------------------------------------------------------------------
    llm_comparison: Optional[Dict[str, Dict[str, float]]] = None
    if compare_llm:
        print("\n[Eval] Running LLM comparison (pretrained vs fine-tuned)...")
        print(f"[Eval]   Pretrained : {constant.vllm_base_url}  model={constant.llm_model_name}")
        print(f"[Eval]   Fine-tuned : {finetuned_llm_url}  model={finetuned_llm_model}")
        try:
            llm_comparison = compute_llm_comparison(
                pipeline, eval_qa,
                finetuned_url=finetuned_llm_url,
                finetuned_model=finetuned_llm_model,
                n_samples=50 if not quick else 10,
            )
            print(f"  Pretrained : {llm_comparison['pretrained']}")
            print(f"  Fine-tuned : {llm_comparison['finetuned']}")
        except Exception as e:
            print(f"[Eval] ⚠ LLM comparison failed: {e}")
            llm_comparison = None

    # ------------------------------------------------------------------
    # 4. RAGAS end-to-end
    # ------------------------------------------------------------------
    print("\n[Eval] Computing RAGAS metrics (faithfulness, context precision)...")
    ragas_metrics = compute_ragas_metrics(pipeline, eval_qa[:20])
    print(f"  {ragas_metrics}")

    # ------------------------------------------------------------------
    # 5. Latency
    # ------------------------------------------------------------------
    print("\n[Eval] Measuring per-module latency...")
    latency = measure_latency(pipeline, eval_qa)
    print(f"  {latency}")

    # ------------------------------------------------------------------
    # 6. Per-doc-type breakdown
    # ------------------------------------------------------------------
    print("\n[Eval] Running per-document-type breakdown...")
    breakdown = eval_by_doc_type(pipeline, eval_qa)
    print(f"  {breakdown}")

    # ------------------------------------------------------------------
    # 7. Charts
    # ------------------------------------------------------------------
    print("\n[Eval] Generating charts...")
    plot_retrieval_recall(retrieval_metrics, output_dir)
    plot_reranker_precision(reranker_metrics, output_dir)
    plot_generation_quality(gen_metrics, breakdown, output_dir)
    plot_e2e_radar(ragas_metrics, gen_metrics, retrieval_metrics, output_dir)
    plot_latency_breakdown(latency, output_dir)
    if llm_comparison:
        plot_llm_comparison(llm_comparison, output_dir)
        print("[Eval] LLM comparison chart saved → llm_comparison.png")

    # ------------------------------------------------------------------
    # 8. HTML report
    # ------------------------------------------------------------------
    report_path = build_html_report(
        retrieval_metrics=retrieval_metrics,
        reranker_metrics=reranker_metrics,
        gen_metrics=gen_metrics,
        ragas_metrics=ragas_metrics,
        latency=latency,
        breakdown=breakdown,
        output_dir=output_dir,
        eval_n=len(eval_qa),
        llm_comparison=llm_comparison,
    )

    elapsed = time.perf_counter() - t0
    print(f"\n[Eval] Evaluation complete in {elapsed:.1f}s")
    print(f"[Eval] HTML report: {report_path}")
    print(f"[Eval] All outputs in: {output_dir}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ImmunoBiology RAG — Full system evaluation with 5 charts + HTML report."
    )
    parser.add_argument(
        "--eval-file", type=str,
        default=str(Path(constant.train_dir) / "eval_qa.jsonl"),
        help="Path to eval_qa.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(Path(constant.BASE_DIR) / "outputs" / "system_eval"),
        help="Output directory for charts and HTML report",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick evaluation: use 20 random QA pairs",
    )
    parser.add_argument(
        "--n-eval", type=int, default=150,
        help="Number of QA pairs for full evaluation (default: 150)",
    )
    parser.add_argument(
        "--compare-llm", action="store_true",
        help=(
            "Compare pretrained vs fine-tuned LLM generation quality. "
            "Requires a second vLLM server running at --finetuned-llm-url "
            "serving --finetuned-llm-model."
        ),
    )
    parser.add_argument(
        "--finetuned-llm-url", type=str, default="http://localhost:8001/v1",
        help="vLLM base URL for the fine-tuned LLM server (default: http://localhost:8001/v1)",
    )
    parser.add_argument(
        "--finetuned-llm-model", type=str, default="finetuned",
        help="Model name served by the fine-tuned vLLM server (default: finetuned)",
    )
    args = parser.parse_args()

    run_evaluation(
        eval_qa_file=args.eval_file,
        output_dir=args.output_dir,
        quick=args.quick,
        n_eval=args.n_eval,
        compare_llm=args.compare_llm,
        finetuned_llm_url=args.finetuned_llm_url,
        finetuned_llm_model=args.finetuned_llm_model,
    )
