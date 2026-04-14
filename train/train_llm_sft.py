# =============================================================================
# ImmunoBiology RAG — LLM Supervised Fine-Tuning (SFT)
# =============================================================================
# Fine-tunes Qwen/Qwen3-8B on immunology QA pairs using LoRA.
# Uses LLaMA-Factory as the training backend (called via subprocess).
#
# Features:
#   - Checkpoint/resume: writes training_complete.json when done; skips
#     training on re-run if already complete
#   - Auto-resume: detects existing checkpoint-N/ folders and passes
#     resume_from_checkpoint to LLaMA-Factory automatically
#   - LoRA: r=16, alpha=32, target modules q_proj + v_proj
#   - bf16; Flash Attention 2 optional
#   - Real-time loss + LR curves via TensorBoard + matplotlib
#   - Pre/post comparison: ROUGE-L, BERTScore-F1
#
# Usage:
#   python -m train.train_llm_sft                    # full training (or skip if done)
#   python -m train.train_llm_sft --force            # retrain from scratch
#   python -m train.train_llm_sft --compare          # comparison only
#   python -m train.train_llm_sft --parse-logs       # parse existing TensorBoard logs
#
# Sentinel file: outputs/models/llm_finetuned/training_complete.json
#   Written after training finishes. Delete it (or use --force) to retrain.

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import constant


# =============================================================================
# LLaMA-Factory config builder
# =============================================================================

LLAMAFACTORY_TRAIN_YAML = """### LLaMA-Factory SFT Config — ImmunoBiology RAG
model_name_or_path: {model_path}

do_train: true

finetuning_type: lora
lora_target: q_proj,v_proj
lora_rank: {lora_r}
lora_alpha: {lora_alpha}
lora_dropout: 0.05

dataset: {dataset_name}
dataset_dir: {dataset_dir}
template: qwen
cutoff_len: 2048
overwrite_cache: true
preprocessing_num_workers: 1

output_dir: {output_dir}
logging_dir: {logging_dir}
save_strategy: epoch
logging_steps: 5
save_total_limit: 3

num_train_epochs: {epochs}
per_device_train_batch_size: {batch_size}
gradient_accumulation_steps: 2
learning_rate: {lr}
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: {bf16}
flash_attn: {flash_attn}

eval_strategy: "no"

report_to: tensorboard
{resume_line}
"""


def build_llamafactory_config(
    output_dir: str,
    logging_dir: str,
    dataset_dir: str,
    dataset_name: str = "immuno_sft",
    resume_from_checkpoint: Optional[str] = None,
) -> str:
    """
    Build a LLaMA-Factory YAML config and return the config file path.

    If resume_from_checkpoint is provided, adds the resume directive so
    LLaMA-Factory continues training from that checkpoint.
    """
    resume_line = (
        f"resume_from_checkpoint: {resume_from_checkpoint}"
        if resume_from_checkpoint
        else ""
    )

    config_content = LLAMAFACTORY_TRAIN_YAML.format(
        model_path=constant.llm_local_path,
        lora_r=constant.lora_r,
        lora_alpha=constant.lora_alpha,
        dataset_name=dataset_name,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        logging_dir=logging_dir,
        epochs=constant.llm_sft_epochs,
        batch_size=constant.llm_sft_batch_size,
        lr=constant.llm_sft_lr,
        bf16="true" if constant.llm_sft_bf16 else "false",
        flash_attn="fa2" if constant.llm_sft_flash_attn else "disabled",
        resume_line=resume_line,
    )
    config_path = str(Path(output_dir) / "sft_config.yaml")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_content)
    return config_path


def build_dataset_info(
    sft_file: str,
    dataset_dir: str,
    dataset_name: str = "immuno_sft",
) -> None:
    """Write dataset_info.json for LLaMA-Factory to discover the SFT dataset."""
    info = {
        dataset_name: {
            "file_name":  Path(sft_file).name,
            "formatting": "sharegpt",
            "columns":    {"messages": "messages"},
        }
    }
    info_path = Path(dataset_dir) / "dataset_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"[SFT] dataset_info.json written: {info_path}")


# =============================================================================
# Run LLaMA-Factory training via subprocess
# =============================================================================

def run_llamafactory_training(
    config_path: str,
    llamafactory_dir: Optional[str] = None,
) -> int:
    """
    Launch LLaMA-Factory training as a subprocess.
    Returns the process exit code (0 = success).
    """
    if llamafactory_dir is None:
        candidates = [
            constant.llamafactory_dir,
            str(Path(constant.BASE_DIR) / "LLaMA-Factory"),
            str(Path(constant.BASE_DIR) / "LLaMA-Factory-main"),
            str(Path(constant.BASE_DIR).parent / "LLaMA-Factory"),
        ]
        llamafactory_dir = next((d for d in candidates if Path(d).exists()), None)

    if llamafactory_dir is None or not Path(llamafactory_dir).exists():
        print("[SFT] ERROR: LLaMA-Factory not found.")
        print("     Install it inside the project directory:")
        print(f"       cd {constant.BASE_DIR}")
        print("       git clone https://github.com/hiyouga/LLaMA-Factory.git LLaMA-Factory")
        print('       cd LLaMA-Factory && pip install -e ".[torch,metrics]"')
        return 1

    train_script = str(Path(llamafactory_dir) / "src" / "train.py")
    if not Path(train_script).exists():
        cmd = [sys.executable, "-m", "llamafactory.train", config_path]
    else:
        cmd = [sys.executable, train_script, config_path]

    print(f"[SFT] Running LLaMA-Factory: {' '.join(cmd)}")
    print(f"[SFT] Working directory: {llamafactory_dir}")

    result = subprocess.run(
        cmd,
        cwd=llamafactory_dir,
        env={**os.environ, "PYTHONPATH": llamafactory_dir},
    )
    return result.returncode


# =============================================================================
# Parse TensorBoard logs → training history
# =============================================================================

def parse_tensorboard_logs(logging_dir: str) -> Dict[str, list]:
    """Parse TensorBoard event files to extract loss and LR curves."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("[SFT] tensorboard not available for log parsing.")
        return {}

    history = {
        "train_steps": [], "train_loss": [],
        "eval_steps":  [], "eval_loss": [],
        "lr_steps":    [], "lr": [],
    }

    try:
        ea = EventAccumulator(logging_dir)
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
    except Exception as e:
        print(f"[SFT] TensorBoard log parsing failed ({e}), skipping curves.")
        return history

    for tag in tags:
        events = ea.Scalars(tag)
        steps  = [e.step  for e in events]
        values = [e.value for e in events]

        t = tag.lower()
        if "train/loss" in t or "train_loss" in t:
            history["train_steps"] = steps
            history["train_loss"]  = values
        elif "eval/loss" in t or "eval_loss" in t:
            history["eval_steps"] = steps
            history["eval_loss"]  = values
        elif "learning_rate" in t or "lr" in t:
            history["lr_steps"] = steps
            history["lr"]       = values

    return history


# =============================================================================
# Evaluation: ROUGE-L, BERTScore
# =============================================================================

def evaluate_llm(
    model_path: str,
    eval_qa_file: str,
    pipeline=None,
    sample_n: int = 50,
) -> Dict[str, float]:
    """Evaluate LLM on the eval QA set using ROUGE-L + BERTScore."""
    import random

    with open(eval_qa_file, "r", encoding="utf-8") as f:
        eval_qa = [json.loads(line) for line in f if line.strip()]

    if sample_n and len(eval_qa) > sample_n:
        eval_qa = random.sample(eval_qa, sample_n)

    print(f"[SFT Eval] Evaluating {len(eval_qa)} QA pairs from: {eval_qa_file}")

    if pipeline is None:
        from src.pipeline import RAGPipeline
        pipeline = RAGPipeline()

    predictions, references = [], []
    for qa in eval_qa:
        try:
            result = pipeline.answer(qa["question"])
            predictions.append(result["answer"])
            references.append(qa["answer"])
        except Exception as e:
            print(f"[SFT Eval] Warning: pipeline failed: {e}")

    if not predictions:
        return {"rouge_l": 0.0, "bertscore_f1": 0.0}

    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_l = float(np.mean([
            scorer.score(ref, pred)["rougeL"].fmeasure
            for ref, pred in zip(references, predictions)
        ]))
    except ImportError:
        print("[SFT Eval] rouge_score not installed. Skipping ROUGE-L.")
        rouge_l = 0.0

    try:
        from bert_score import score as bert_score
        P, R, F = bert_score(predictions, references, lang="en", verbose=False)
        bertscore_f1 = float(F.mean().item())
    except ImportError:
        print("[SFT Eval] bert_score not installed. Skipping BERTScore.")
        bertscore_f1 = 0.0

    return {"rouge_l": rouge_l, "bertscore_f1": bertscore_f1}


# =============================================================================
# Visualization
# =============================================================================

def plot_sft_curves(history: dict, output_dir: str) -> None:
    """Save SFT training loss + LR schedule curves."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if history.get("train_steps"):
        axes[0].plot(history["train_steps"], history["train_loss"],
                     alpha=0.6, color="steelblue", linewidth=0.8, label="Train loss")
    if history.get("eval_steps"):
        axes[0].plot(history["eval_steps"], history["eval_loss"],
                     color="tomato", linewidth=2, marker="o", markersize=4, label="Eval loss")
    axes[0].set_xlabel("Global step")
    axes[0].set_ylabel("Cross-entropy Loss")
    axes[0].set_title("LLM SFT Training Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if history.get("lr_steps"):
        axes[1].plot(history["lr_steps"], history["lr"], color="mediumpurple", linewidth=1.5)
        axes[1].set_xlabel("Global step")
        axes[1].set_ylabel("Learning rate")
        axes[1].set_title("LR Schedule (Cosine with Warmup)")
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = str(Path(output_dir) / "sft_training_curves.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[SFT] Training curves saved: {out_path}")


def compare_pre_post_llm(
    eval_qa_file: str,
    base_model_path: str,
    finetuned_model_path: str,
    output_dir: str,
    sample_n: int = 50,
) -> None:
    """Compare base vs fine-tuned LLM. Saves grouped bar chart + CSV."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = {}
    for name, path in [("Pre-trained", base_model_path), ("Fine-tuned", finetuned_model_path)]:
        if not Path(path).exists() and path != base_model_path:
            print(f"[SFT] {name} model not found at {path}, skipping.")
            continue
        print(f"\n[SFT] Evaluating {name}: {path}")
        from src.pipeline import RAGPipeline
        pipeline = RAGPipeline()
        metrics = evaluate_llm(path, eval_qa_file, pipeline=pipeline, sample_n=sample_n)
        results[name] = metrics
        print(f"  {name}: {metrics}")

    if len(results) < 2:
        print("[SFT] Not enough models for comparison.")
        return

    metric_labels = ["ROUGE-L", "BERTScore-F1"]
    metric_keys   = ["rouge_l", "bertscore_f1"]
    model_names   = list(results.keys())
    x      = np.arange(len(metric_labels))
    width  = 0.35
    colors = ["steelblue", "tomato"]

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, name in enumerate(model_names):
        vals = [results[name].get(k, 0) for k in metric_keys]
        ax.bar(x + i * width, vals, width, label=name, color=colors[i], alpha=0.85)
        for j, v in enumerate(vals):
            ax.text(x[j] + i * width, v + 0.003, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("LLM SFT: Pre-trained vs Fine-tuned")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    chart_path = str(Path(output_dir) / "comparison.png")
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"[SFT] Comparison chart saved: {chart_path}")

    csv_path = str(Path(output_dir) / "comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model"] + metric_labels)
        for name in model_names:
            row = [name] + [results[name].get(k, "") for k in metric_keys]
            writer.writerow(row)
    print(f"[SFT] Comparison CSV saved: {csv_path}")


# =============================================================================
# Main training orchestration
# =============================================================================

def run_sft(
    sft_file: str,
    output_dir: str,
    eval_dir: str,
    llamafactory_dir: Optional[str] = None,
    force: bool = False,
) -> None:
    """
    Full SFT pipeline with checkpoint/resume support.

    1. Check sentinel → skip training if already done (unless --force)
    2. Auto-detect partial checkpoints → resume LLaMA-Factory from checkpoint
    3. Launch training (subprocess)
    4. Write sentinel on success
    5. Parse TensorBoard logs + plot curves
    6. Pre/post comparison
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    sentinel_path = Path(output_dir) / "training_complete.json"
    dataset_dir   = str(Path(sft_file).parent)
    logging_dir   = str(Path(output_dir) / "runs")

    # --force: delete sentinel so training proceeds
    if force and sentinel_path.exists():
        sentinel_path.unlink()
        print(f"[SFT] --force: removed sentinel, retraining from scratch.")

    # Check if already complete
    if sentinel_path.exists():
        with open(sentinel_path) as f:
            info = json.load(f)
        print(f"[SFT] ✓ Training already complete (finished {info.get('completed_at', '?')}).")
        print(f"[SFT]   Skipping training. Use --force to retrain.")
    else:
        # Auto-detect partial checkpoints for resume
        existing_checkpoints = sorted(
            Path(output_dir).glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
        )
        resume_from_checkpoint = None
        if existing_checkpoints:
            resume_from_checkpoint = str(existing_checkpoints[-1])
            print(f"[SFT] Found partial checkpoint: {resume_from_checkpoint}")
            print(f"[SFT] Will resume LLaMA-Factory training from this checkpoint.")

        # Build config (with or without resume directive)
        build_dataset_info(sft_file, dataset_dir)
        config_path = build_llamafactory_config(
            output_dir=output_dir,
            logging_dir=logging_dir,
            dataset_dir=dataset_dir,
            resume_from_checkpoint=resume_from_checkpoint,
        )
        print(f"[SFT] Config written: {config_path}")

        # Launch training
        t0 = time.perf_counter()
        exit_code = run_llamafactory_training(config_path, llamafactory_dir)
        elapsed = time.perf_counter() - t0

        if exit_code != 0:
            print(f"[SFT] Training failed with exit code {exit_code}.")
            print(f"[SFT] Partial checkpoints (if any) are in: {output_dir}")
            print(f"[SFT] Re-run without --force to auto-resume from the last checkpoint.")
            return

        print(f"\n[SFT] Training finished in {elapsed/60:.1f} min.")

        # Write sentinel
        checkpoints = sorted(
            Path(output_dir).glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
        )
        latest_ckpt = str(checkpoints[-1]) if checkpoints else ""
        sentinel_data = {
            "completed_at":      time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_min":       round(elapsed / 60, 1),
            "latest_checkpoint": latest_ckpt,
        }
        with open(sentinel_path, "w", encoding="utf-8") as f:
            json.dump(sentinel_data, f, indent=2)
        print(f"[SFT] ✓ Sentinel written: {sentinel_path}")

    # Always parse logs + plot (cheap, idempotent)
    print("\n[SFT] Parsing TensorBoard logs...")
    history = parse_tensorboard_logs(logging_dir)
    if history:
        plot_sft_curves(history, eval_dir)

    # Pre/post comparison
    eval_qa_file = str(Path(constant.train_dir) / "eval_qa.jsonl")
    if Path(eval_qa_file).exists():
        checkpoints = sorted(
            Path(output_dir).glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
        )
        finetuned_path = str(checkpoints[-1]) if checkpoints else str(Path(output_dir) / "best")
        compare_pre_post_llm(
            eval_qa_file=eval_qa_file,
            base_model_path=constant.llm_local_path,
            finetuned_model_path=finetuned_path,
            output_dir=eval_dir,
        )

    print(f"\n[SFT] All done. Outputs in: {output_dir}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ImmunoBiology RAG — LLM SFT with LoRA + checkpoint/resume.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Checkpoint/resume behaviour
---------------------------
  training_complete.json is written when training finishes.
  On re-run, if this file exists, training is skipped automatically.

  If training was interrupted (no sentinel but checkpoint-N/ exists),
  re-running will automatically resume from the last checkpoint via
  LLaMA-Factory's resume_from_checkpoint directive.

  --force    Delete sentinel and retrain from scratch (ignores checkpoints)
        """,
    )
    parser.add_argument("--sft-file",         type=str, default=str(Path(constant.train_dir) / "sft_train.jsonl"))
    parser.add_argument("--output-dir",       type=str, default=constant.llm_finetuned_path)
    parser.add_argument("--eval-dir",         type=str, default=str(Path(constant.BASE_DIR) / "outputs" / "llm_eval"))
    parser.add_argument("--llamafactory-dir", type=str, default=None, help="Path to LLaMA-Factory root")
    parser.add_argument("--compare",          action="store_true", help="Run pre/post comparison only")
    parser.add_argument("--parse-logs",       action="store_true", help="Parse TensorBoard logs + plot only")
    parser.add_argument("--force",            action="store_true", help="Retrain from scratch even if already complete")

    args = parser.parse_args()

    if args.compare:
        eval_qa_file = str(Path(constant.train_dir) / "eval_qa.jsonl")
        checkpoints  = sorted(
            Path(args.output_dir).glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
        )
        finetuned_path = str(checkpoints[-1]) if checkpoints else str(Path(args.output_dir) / "best")
        compare_pre_post_llm(
            eval_qa_file=eval_qa_file,
            base_model_path=constant.llm_local_path,
            finetuned_model_path=finetuned_path,
            output_dir=args.eval_dir,
        )
    elif args.parse_logs:
        logging_dir = str(Path(args.output_dir) / "runs")
        history = parse_tensorboard_logs(logging_dir)
        if history:
            plot_sft_curves(history, args.eval_dir)
        else:
            print("[SFT] No TensorBoard logs found.")
    else:
        run_sft(
            sft_file=args.sft_file,
            output_dir=args.output_dir,
            eval_dir=args.eval_dir,
            llamafactory_dir=args.llamafactory_dir,
            force=args.force,
        )
