# =============================================================================
# ImmunoBiology RAG — Reranker Fine-Tuning  (v3: LoRA)
# =============================================================================
# Fine-tunes BAAI/bge-reranker-v2-m3 on immunology QA triplets.
#
# v1/v2 used full fine-tuning → catastrophic forgetting (all 560M params updated
# on 14,741 triplets erased broad ranking knowledge, Recall@1 dropped from 0.559→0.426).
# v3 uses LoRA (PEFT): base weights frozen, only ~7M adapter params trained (~1.3%).
# Controlled by config.yaml: training.reranker.use_lora (default: true)
# or --lora / --no-lora CLI flags.
#
# Features:
#   - LoRA mode (default): PEFT adapter, ~30-50 MB output, no catastrophic forgetting
#   - Full fine-tune mode (legacy): pass --no-lora to reproduce v1/v2 behaviour
#   - Checkpoint/resume: writes training_complete.json when done; skips on re-run
#   - Auto-resume from best/ checkpoint if interrupted
#   - ETA in progress logs
#   - Real-time train/eval loss curves via TensorBoard + matplotlib
#   - MRR@10 and NDCG@10 evaluation every N steps
#   - Pre/post fine-tuning comparison with grouped bar chart
#   - All outputs saved to outputs/reranker_eval/
#
# Usage:
#   python -m train.train_reranker                  # LoRA training (reads config.yaml)
#   python -m train.train_reranker --lora           # force LoRA mode
#   python -m train.train_reranker --no-lora        # force full fine-tuning (legacy)
#   python -m train.train_reranker --resume         # resume from best/ checkpoint
#   python -m train.train_reranker --force          # retrain from scratch
#   python -m train.train_reranker --eval-only      # evaluate base model only
#   python -m train.train_reranker --compare        # pre/post comparison chart only
#
# Sentinel file: outputs/models/reranker_finetuned/training_complete.json
#   Written after training finishes. Delete it (or use --force) to retrain.

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from src import constant

# PEFT / LoRA — optional dependency (pip install peft)
# Graceful import so the script still runs without peft for full fine-tuning mode.
try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False


# =============================================================================
# Dataset
# =============================================================================

class RerankerDataset(Dataset):
    """
    Loads reranker triplets from JSONL.

    Each record: {"query": str, "pos": str, "neg": str}
    Returns two training examples per record:
        (query, pos, label=1)  — positive pair
        (query, neg, label=0)  — hard negative pair
    """

    def __init__(self, jsonl_path: str):
        self.pairs = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                q, pos, neg = rec["query"], rec["pos"], rec["neg"]
                self.pairs.append((q, pos, 1.0))
                self.pairs.append((q, neg, 0.0))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        query, doc, label = self.pairs[idx]
        return query, doc, torch.tensor(label, dtype=torch.float)


def collate_fn(batch, tokenizer, max_length=512):
    queries = [item[0] for item in batch]
    docs    = [item[1] for item in batch]
    labels  = torch.stack([item[2] for item in batch])

    encoded = tokenizer(
        queries, docs,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return encoded, labels


# =============================================================================
# Evaluation metrics
# =============================================================================

def compute_ndcg_at_k(scores_by_query: List[List[Tuple[float, int]]], k: int = 10) -> float:
    """Compute NDCG@k across all queries."""
    def dcg_at_k(ranked_labels, k):
        dcg = 0.0
        for i, rel in enumerate(ranked_labels[:k], 1):
            dcg += (2**rel - 1) / math.log2(i + 1)
        return dcg

    ndcg_scores = []
    for pairs in scores_by_query:
        ranked = sorted(pairs, key=lambda x: x[0], reverse=True)
        ranked_labels = [rel for _score, rel in ranked]
        ideal_labels  = sorted(ranked_labels, reverse=True)
        idcg = dcg_at_k(ideal_labels, k)
        if idcg == 0:
            continue
        ndcg_scores.append(dcg_at_k(ranked_labels, k) / idcg)

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


def compute_mrr_at_k(scores_by_query: List[List[Tuple[float, int]]], k: int = 10) -> float:
    """Compute MRR@k across all queries."""
    mrr_scores = []
    for pairs in scores_by_query:
        ranked = sorted(pairs, key=lambda x: x[0], reverse=True)
        for rank, (_score, rel) in enumerate(ranked[:k], 1):
            if rel == 1:
                mrr_scores.append(1.0 / rank)
                break
        else:
            mrr_scores.append(0.0)
    return float(np.mean(mrr_scores)) if mrr_scores else 0.0


def evaluate_reranker(
    model,
    tokenizer,
    eval_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Run evaluation: compute loss, NDCG@10, MRR@10."""
    model.eval()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n_batches  = 0

    all_scores, all_labels = [], []

    with torch.no_grad():
        for encoded, labels in eval_loader:
            encoded = {k: v.to(device) for k, v in encoded.items()}
            labels  = labels.to(device)
            logits  = model(**encoded).logits.squeeze(-1)
            loss    = loss_fn(logits.float(), labels.float())
            total_loss += loss.item()
            n_batches  += 1
            all_scores.extend(logits.cpu().float().tolist())
            all_labels.extend(labels.cpu().float().tolist())

    avg_loss = total_loss / max(n_batches, 1)

    # Group pairs back into per-query lists (every 2 items = 1 query: pos + neg)
    scores_by_query = []
    for i in range(0, len(all_scores) - 1, 2):
        scores_by_query.append([
            (all_scores[i],   int(all_labels[i])),
            (all_scores[i+1], int(all_labels[i+1])),
        ])

    ndcg = compute_ndcg_at_k(scores_by_query, k=10)
    mrr  = compute_mrr_at_k(scores_by_query,  k=10)

    model.train()
    return {"eval_loss": avg_loss, "ndcg@10": ndcg, "mrr@10": mrr}


# =============================================================================
# Training
# =============================================================================

def train_reranker(
    model_path: str,
    train_file: str,
    eval_file:  str,
    output_dir: str,
    eval_dir:   str,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 1e-4,
    eval_steps: int = 200,
    warmup_ratio: float = 0.2,
    max_length: int = 512,
    fp16: bool = False,
    resume_from: Optional[str] = None,
    # ── LoRA v3 params ──────────────────────────────────────────────────────
    use_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[List[str]] = None,
) -> Dict[str, list]:
    """
    Fine-tune the BGE reranker on immunology triplets.

    Args:
        use_lora:    If True, wrap model with PEFT LoRA (v3 default).
                     Only ~1.3% of params are trained; base weights frozen.
                     Prevents catastrophic forgetting seen in full fine-tuning.
        resume_from: path to a checkpoint directory to resume from.
                     Auto-detects LoRA adapter (adapter_config.json present)
                     vs full model checkpoint.

    Returns:
        Training history dict with loss/metric curves for plotting.
    """
    if use_lora and not _PEFT_AVAILABLE:
        raise ImportError(
            "[Reranker] LoRA mode requires the 'peft' package. "
            "Install it with: pip install peft"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sentinel_path = Path(output_dir) / "training_complete.json"
    _lora_modules = lora_target_modules or ["query", "key", "value"]

    mode_tag = "LoRA" if use_lora else "full fine-tuning"
    print(f"[Reranker] Mode: {mode_tag}")
    print(f"[Reranker] Training on: {device}")
    print(f"[Reranker] Train file:  {train_file}")

    # ----- Build LoRA config (needed before resume detection) -----
    lora_cfg = None
    if use_lora:
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=_lora_modules,
            bias="none",
        )

    # ----- Load model + tokenizer -----
    # Tokenizer always comes from the base model (adapter doesn't change vocab).
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if resume_from and Path(resume_from).exists():
        _is_adapter = (Path(resume_from) / "adapter_config.json").exists()
        if use_lora and _is_adapter:
            # Resume: load base → apply saved adapter → re-wrap for training
            print(f"[Reranker] Resuming LoRA adapter from: {resume_from}")
            _base = AutoModelForSequenceClassification.from_pretrained(model_path)
            _peft_model = PeftModel.from_pretrained(_base, resume_from)
            # Unwrap to base model then re-apply lora_cfg so all adapter weights
            # are in the optimizer from the start of this training run.
            model = get_peft_model(_peft_model.base_model.model, lora_cfg)
        else:
            print(f"[Reranker] Resuming from full checkpoint: {resume_from}")
            model = AutoModelForSequenceClassification.from_pretrained(resume_from)
    else:
        print(f"[Reranker] Starting from base model: {model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

    model.to(device)
    if fp16 and device.type == "cuda":
        model.half()

    # ----- Apply LoRA wrapper (new training run, not resume) -----
    if use_lora and not (resume_from and Path(resume_from).exists()):
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    # ----- Datasets -----
    train_dataset = RerankerDataset(train_file)
    eval_dataset  = RerankerDataset(eval_file)

    _collate = lambda b: collate_fn(b, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  collate_fn=_collate)
    eval_loader  = DataLoader(eval_dataset,  batch_size=batch_size, shuffle=False, collate_fn=_collate)

    print(f"[Reranker] Train samples: {len(train_dataset)} | Eval samples: {len(eval_dataset)}")

    # ----- Optimizer + scheduler -----
    total_steps  = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    # Only optimize trainable parameters — for LoRA this is the adapter only (~7M).
    # For full fine-tuning all params are trainable so the filter is a no-op.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    print(f"[Reranker] Total steps: {total_steps} | Warmup steps: {warmup_steps}")

    loss_fn = torch.nn.BCEWithLogitsLoss()

    # ----- TensorBoard -----
    tb_dir = str(Path(output_dir) / "runs")
    writer = SummaryWriter(log_dir=tb_dir)

    # ----- History for matplotlib -----
    history = {
        "train_steps": [], "train_loss": [],
        "eval_steps":  [], "eval_loss": [], "ndcg10": [], "mrr10": [],
        "lr_steps":    [], "lr": [],
    }

    global_step   = 0
    best_ndcg     = 0.0
    t0            = time.perf_counter()
    step_times: List[float] = []   # rolling window for ETA

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t_epoch = time.perf_counter()

        for step, (encoded, labels) in enumerate(train_loader, 1):
            t_step = time.perf_counter()

            encoded = {k: v.to(device) for k, v in encoded.items()}
            labels  = labels.to(device)

            logits = model(**encoded).logits.squeeze(-1)
            loss   = loss_fn(logits.float(), labels.float())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss  += loss.item()
            global_step += 1

            # Track step time for ETA (rolling last 20 steps)
            step_times.append(time.perf_counter() - t_step)
            if len(step_times) > 20:
                step_times.pop(0)

            # Log train loss
            writer.add_scalar("Loss/train", loss.item(), global_step)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], global_step)

            history["train_steps"].append(global_step)
            history["train_loss"].append(loss.item())
            history["lr_steps"].append(global_step)
            history["lr"].append(scheduler.get_last_lr()[0])

            if global_step % eval_steps == 0:
                metrics = evaluate_reranker(model, tokenizer, eval_loader, device)
                writer.add_scalar("Loss/eval",    metrics["eval_loss"], global_step)
                writer.add_scalar("NDCG@10/eval", metrics["ndcg@10"],   global_step)
                writer.add_scalar("MRR@10/eval",  metrics["mrr@10"],    global_step)

                history["eval_steps"].append(global_step)
                history["eval_loss"].append(metrics["eval_loss"])
                history["ndcg10"].append(metrics["ndcg@10"])
                history["mrr10"].append(metrics["mrr@10"])

                # ETA estimate
                avg_step_s  = sum(step_times) / len(step_times)
                steps_left  = total_steps - global_step
                eta_min     = (steps_left * avg_step_s) / 60

                print(f"[Reranker] Epoch {epoch}/{epochs} Step {global_step}/{total_steps} | "
                      f"Train loss: {loss.item():.4f} | "
                      f"Eval loss: {metrics['eval_loss']:.4f} | "
                      f"NDCG@10: {metrics['ndcg@10']:.4f} | "
                      f"MRR@10: {metrics['mrr@10']:.4f} | "
                      f"ETA: ~{eta_min:.0f} min", flush=True)

                # Save best model
                if metrics["ndcg@10"] > best_ndcg:
                    best_ndcg = metrics["ndcg@10"]
                    save_path = str(Path(output_dir) / "best")
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    print(f"[Reranker] ✓ New best NDCG@10={best_ndcg:.4f} → saved to {save_path}", flush=True)

        epoch_elapsed = time.perf_counter() - t_epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"\n[Reranker] Epoch {epoch}/{epochs} complete | "
              f"Avg loss: {avg_epoch_loss:.4f} | "
              f"Elapsed: {epoch_elapsed/60:.1f} min\n", flush=True)

    writer.close()

    # Save final checkpoint
    final_path = str(Path(output_dir) / "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    total_elapsed = time.perf_counter() - t0
    print(f"[Reranker] Training complete. Best NDCG@10: {best_ndcg:.4f}")
    print(f"[Reranker] Total training time: {total_elapsed/60:.1f} min")
    print(f"[Reranker] Final model saved: {final_path}")

    # Write sentinel so re-runs skip training automatically
    sentinel_data = {
        "completed_at":    time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_ndcg_at_10": best_ndcg,
        "total_steps":     global_step,
        "epochs":          epochs,
        "elapsed_min":     round(total_elapsed / 60, 1),
        "mode":            "lora" if use_lora else "full",
        "lora_r":          lora_r if use_lora else None,
    }
    with open(sentinel_path, "w", encoding="utf-8") as f:
        json.dump(sentinel_data, f, indent=2)
    print(f"[Reranker] ✓ Sentinel written: {sentinel_path}", flush=True)

    return history


# =============================================================================
# Visualization
# =============================================================================

def plot_training_curves(history: dict, output_dir: str) -> None:
    """Save training curves: loss, metrics, LR schedule."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_steps"], history["train_loss"],
                 alpha=0.6, color="steelblue", linewidth=0.8, label="Train loss")
    if history["eval_steps"]:
        axes[0].plot(history["eval_steps"], history["eval_loss"],
                     color="tomato", linewidth=2, marker="o", markersize=4, label="Eval loss")
    axes[0].set_xlabel("Global step")
    axes[0].set_ylabel("BCE Loss")
    axes[0].set_title("Reranker Training & Eval Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if history["ndcg10"]:
        axes[1].plot(history["eval_steps"], history["ndcg10"],
                     color="mediumseagreen", linewidth=2, marker="s", markersize=4, label="NDCG@10")
        axes[1].plot(history["eval_steps"], history["mrr10"],
                     color="darkorange", linewidth=2, marker="^", markersize=4, label="MRR@10")
    axes[1].set_xlabel("Global step")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Reranker Eval Metrics")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = str(Path(output_dir) / "reranker_training_curves.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[Reranker] Training curves saved: {out}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["lr_steps"], history["lr"], color="mediumpurple", linewidth=1.5)
    ax.set_xlabel("Global step")
    ax.set_ylabel("Learning rate")
    ax.set_title("Learning Rate Schedule (Linear with Warmup)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(Path(output_dir) / "reranker_lr_schedule.png"), dpi=150)
    plt.close(fig)


def _load_model_for_eval(
    path: str,
    base_model_path: str,
    device: torch.device,
):
    """
    Load a reranker model for evaluation — handles both full models and LoRA adapters.

    LoRA adapters are identified by the presence of adapter_config.json in the directory.
    Full models (base or legacy full fine-tuned) are loaded directly.
    """
    if (Path(path) / "adapter_config.json").exists():
        if not _PEFT_AVAILABLE:
            raise ImportError("LoRA adapter found but 'peft' is not installed. pip install peft")
        print(f"[Reranker]   → detected LoRA adapter, loading base + adapter")
        base = AutoModelForSequenceClassification.from_pretrained(base_model_path)
        model = PeftModel.from_pretrained(base, path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(path)
    return model.to(device).eval()


def compare_pre_post(
    model_path: str,
    finetuned_path: str,
    eval_file: str,
    output_dir: str,
) -> None:
    """Compare pre-trained vs fine-tuned reranker. Saves bar chart + CSV.
    Supports both full fine-tuned models and LoRA adapters transparently."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}
    for name, path in [("Pre-trained", model_path), ("Fine-tuned", finetuned_path)]:
        if not Path(path).exists():
            print(f"[Reranker] {name} model not found at {path}, skipping.")
            continue
        print(f"[Reranker] Evaluating {name}: {path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)  # always from base
        model = _load_model_for_eval(path, model_path, device)

        dataset  = RerankerDataset(eval_file)
        _collate = lambda b: collate_fn(b, tokenizer, 512)
        loader   = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=_collate)

        metrics = evaluate_reranker(model, tokenizer, loader, device)
        results[name] = metrics
        print(f"  {name}: {metrics}")

        del model
        torch.cuda.empty_cache()

    if len(results) < 2:
        print("[Reranker] Not enough models for comparison.")
        return

    metrics_to_plot = ["eval_loss", "ndcg@10", "mrr@10"]
    labels = list(results.keys())
    x      = np.arange(len(metrics_to_plot))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, label in enumerate(labels):
        vals = [results[label].get(m, 0) for m in metrics_to_plot]
        ax.bar(x + i * width, vals, width, label=label,
               color=["steelblue", "tomato"][i], alpha=0.85)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(["Eval Loss", "NDCG@10", "MRR@10"])
    ax.set_ylabel("Score")
    ax.set_title("Reranker: Pre-trained vs Fine-tuned")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for i, label in enumerate(labels):
        vals = [results[label].get(m, 0) for m in metrics_to_plot]
        for j, v in enumerate(vals):
            ax.text(x[j] + i * width, v + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(str(Path(output_dir) / "comparison.png"), dpi=150)
    plt.close(fig)
    print(f"[Reranker] Comparison chart: {Path(output_dir) / 'comparison.png'}")

    import csv
    csv_path = str(Path(output_dir) / "comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "eval_loss", "ndcg@10", "mrr@10"])
        for name, m in results.items():
            writer.writerow([name, m.get("eval_loss", ""), m.get("ndcg@10", ""), m.get("mrr@10", "")])
    print(f"[Reranker] Comparison CSV: {csv_path}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ImmunoBiology RAG — Fine-tune BGE reranker (v3 LoRA default).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
LoRA vs full fine-tuning
------------------------
  --lora      Use LoRA (PEFT) — base weights frozen, ~7M adapter params trained.
              Default if config.yaml training.reranker.use_lora=true.
              Prevents catastrophic forgetting. Requires: pip install peft
  --no-lora   Force full fine-tuning (legacy v1/v2 behaviour).

Checkpoint/resume behaviour
---------------------------
  training_complete.json is written when training finishes.
  On re-run, if this file exists, training is skipped automatically.

  --resume   Load weights from best/ checkpoint before training
             (auto-detects LoRA adapter vs full model checkpoint)
  --force    Delete sentinel and retrain from scratch
        """,
    )
    parser.add_argument("--model",       type=str,   default=constant.bge_reranker_model_path)
    parser.add_argument("--train-file",  type=str,   default=str(Path(constant.train_dir) / "reranker_train.jsonl"))
    parser.add_argument("--eval-file",   type=str,   default=str(Path(constant.train_dir) / "reranker_test.jsonl"))
    parser.add_argument("--output-dir",  type=str,   default=constant.reranker_finetuned_path)
    parser.add_argument("--eval-dir",    type=str,   default=str(Path(constant.BASE_DIR) / "outputs" / "reranker_eval"))
    parser.add_argument("--epochs",      type=int,   default=constant.reranker_epochs)
    parser.add_argument("--batch-size",  type=int,   default=constant.reranker_batch_size)
    parser.add_argument("--lr",          type=float, default=constant.reranker_lr)
    parser.add_argument("--eval-steps",  type=int,   default=constant.reranker_eval_steps)
    parser.add_argument("--eval-only",   action="store_true", help="Evaluate base model only (no training)")
    parser.add_argument("--compare",     action="store_true", help="Pre/post comparison chart only")
    parser.add_argument("--resume",      action="store_true", help="Resume from best/ checkpoint")
    parser.add_argument("--force",       action="store_true", help="Retrain from scratch even if already complete")
    # ── LoRA flags ────────────────────────────────────────────────────────────
    _lora_group = parser.add_mutually_exclusive_group()
    _lora_group.add_argument("--lora",    dest="use_lora", action="store_true",  default=None,
                              help="Use LoRA fine-tuning (overrides config.yaml)")
    _lora_group.add_argument("--no-lora", dest="use_lora", action="store_false",
                              help="Use full fine-tuning (legacy — overrides config.yaml)")
    parser.add_argument("--lora-r",       type=int,   default=constant.reranker_lora_r)
    parser.add_argument("--lora-alpha",   type=int,   default=constant.reranker_lora_alpha)
    parser.add_argument("--lora-dropout", type=float, default=constant.reranker_lora_dropout)

    args = parser.parse_args()

    # Resolve LoRA flag: CLI takes precedence; fall back to config.yaml
    use_lora = constant.reranker_use_lora if args.use_lora is None else args.use_lora

    output_dir    = args.output_dir
    sentinel_path = Path(output_dir) / "training_complete.json"
    best_ckpt     = str(Path(output_dir) / "best")

    if args.compare:
        compare_pre_post(
            model_path=args.model,
            finetuned_path=best_ckpt,
            eval_file=args.eval_file,
            output_dir=args.eval_dir,
        )

    elif args.eval_only:
        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model     = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
        if use_lora:
            if not _PEFT_AVAILABLE:
                raise ImportError("pip install peft  (required for --lora --eval-only)")
            lora_cfg = LoraConfig(
                task_type=TaskType.SEQ_CLS, r=args.lora_r, lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=constant.reranker_lora_target_modules, bias="none",
            )
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()
        dataset  = RerankerDataset(args.eval_file)
        _collate = lambda b: collate_fn(b, tokenizer, 512)
        loader   = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=_collate)
        metrics  = evaluate_reranker(model, tokenizer, loader, device)
        print(f"[Reranker] Eval metrics (base model{' + LoRA scaffold' if use_lora else ''}): {metrics}")

    else:
        # --force: remove sentinel so training proceeds
        if args.force and sentinel_path.exists():
            sentinel_path.unlink()
            print(f"[Reranker] --force: removed sentinel, retraining from scratch.")

        # Check if already complete
        if sentinel_path.exists():
            with open(sentinel_path) as f:
                info = json.load(f)
            print(f"[Reranker] ✓ Training already complete (finished {info.get('completed_at', '?')}, "
                  f"best NDCG@10={info.get('best_ndcg_at_10', '?'):.4f}, "
                  f"mode={info.get('mode', 'unknown')}).")
            print(f"[Reranker]   Skipping training. Use --force to retrain, --compare for charts.")
        else:
            # Auto-detect resume point
            resume_from = best_ckpt if (args.resume and Path(best_ckpt).exists()) else None
            if resume_from:
                print(f"[Reranker] Found checkpoint at {resume_from} — resuming.")

            history = train_reranker(
                model_path=args.model,
                train_file=args.train_file,
                eval_file=args.eval_file,
                output_dir=output_dir,
                eval_dir=args.eval_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                eval_steps=args.eval_steps,
                resume_from=resume_from,
                use_lora=use_lora,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                lora_target_modules=constant.reranker_lora_target_modules,
            )
            plot_training_curves(history, args.eval_dir)

        # Always run comparison (plots are cheap and informative)
        if Path(best_ckpt).exists():
            compare_pre_post(
                model_path=args.model,
                finetuned_path=best_ckpt,
                eval_file=args.eval_file,
                output_dir=args.eval_dir,
            )
