# -*- coding: utf-8 -*-
"""Joint multi-task training: sentiment + category trained simultaneously.

Strategy
--------
- Sentiment head : trained on main .apc + supplement data (all samples).
                   Uses weighted CrossEntropyLoss to handle class imbalance.
- Category  head : trained on main .apc samples ONLY.
                   Supplement samples are masked out via is_supplement flag.
- Both losses are summed: loss = sent_loss + cat_loss
- Early stopping monitors dev sentiment macro-F1.

Usage (from kltn/ parent directory):
    python thesis_apc_baseline/experiments/run_joint_experiments.py

Outputs (under runs_joint/):
    experiment_results_joint.txt   ← summary table (printed to screen + file)
    experiment_results_joint.csv   ← one row per config, all metrics
    <short_id>/best_model.pt       ← best checkpoint per config
"""

from __future__ import annotations

import contextlib
import copy
import csv
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ─── Tee: write to stdout AND a file simultaneously ───────────────────────────

class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data: str) -> None:
        for s in self._streams:
            s.write(data)
    def flush(self) -> None:
        for s in self._streams:
            s.flush()


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score, classification_report

from thesis_apc_baseline.dataset_utils import (
    ApcFileDataset,
    build_label_maps_from_apc,
    SENTIMENT_LABELS,
    SENTIMENT_MAP,
)
from thesis_apc_baseline.models.fast_lcf_bert_multitask import FastLcfBertMultiTask

# ─── Paths ────────────────────────────────────────────────────────────────────

DATASET_DIR    = ROOT / "thesis_apc_baseline" / "dataset"
TRAIN_APC      = DATASET_DIR / "train.apc"
DEV_APC        = DATASET_DIR / "dev.apc"
TEST_APC       = DATASET_DIR / "test.apc"
RUNS_DIR       = ROOT / "thesis_apc_baseline" / "runs_joint"
SUPPLEMENT_DIR = DATASET_DIR / "supplement"

SUPPLEMENT_FILES: List[str] = [
    str(SUPPLEMENT_DIR / "negative.tsv"),
    str(SUPPLEMENT_DIR / "neutral.tsv"),
]

# ─── Hyperparameters ──────────────────────────────────────────────────────────

PRETRAINED_BERT  = "bert-base-uncased"
SEED             = 42
NUM_EPOCHS       = 15
PATIENCE         = 4
BATCH_SIZE       = 16
LR               = 2e-5
MAX_SEQ_LEN      = 128
DROPOUT          = 0.1
NUM_HEADS        = 8
TOME_MERGE_STEPS = USE_MIXED_PRECISION = True  # Use torch.cuda.amp when running on GPU
# Default loss and early stopping weights
DEFAULT_TASK_WEIGHT_SENT = 1.0
DEFAULT_TASK_WEIGHT_CAT  = 1.0
DEFAULT_ES_WEIGHT_SENT   = 0.5  # For early stopping validation metric
DEFAULT_ES_WEIGHT_CAT    = 0.5

# Override weights per config (task_weight_sent loss, task_weight_cat, es_weight_sent, es_weight_cat)
WEIGHT_OVERRIDES = {
    "baseline_sent_focus": (1.2, 0.8, 0.6, 0.4),  # Emphasize sentiment
    "baseline_cat_focus":  (0.8, 1.2, 0.4, 0.6),  # Emphasize category
    "baseline_balanced":   (1.0, 1.0, 0.5, 0.5),  # Balanced
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# (use_lcf, use_tome, tome_resize, merge_strategy, display_name, short_id)
CONFIGS: List[Tuple[bool, bool, bool, str, str, str]] = [
    # Baseline with different loss/ES weights
    (False, False, True, "bipartite", "Baseline (Sent-focus)", "baseline_sent_focus"),
    (False, False, True, "bipartite", "Baseline (Cat-focus)",  "baseline_cat_focus"),
    (False, False, True, "bipartite", "Baseline (Balanced)",   "baseline_balanced"),
    # Other models (use default weights)
    (True,  False, True,  "bipartite", "LCF only",        "lcf_only"),
    (True,  True,  True,  "bipartite", "LCF+Bip (resize)",      "lcf_bip_resize"),
    (True,  True,  False, "bipartite", "LCF+Bip (compact)",     "lcf_bip_compact"),
    (False, True,  True,  "bipartite", "Bip (resize)",          "bip_resize"),
    (False, True,  False, "bipartite", "Bip (compact)",         "bip_compact"),
    (True,  True,  True,  "sequential_local", "LCF+Seq (resize)",      "lcf_seq_resize"),
    (True,  True,  False, "sequential_local", "LCF+Seq (compact)",     "lcf_seq_compact"),
    (False, True,  True,  "sequential_local", "Seq (resize)",          "seq_resize"),
    (False, True,  False, "sequential_local", "Seq (compact)",         "seq_compact"),
    (True,  True,  True,  "attention_weighted", "LCF+Attn (resize)",   "lcf_attn_resize"),
    (True,  True,  False, "attention_weighted", "LCF+Attn (compact)",  "lcf_attn_compact"),
    (False, True,  True,  "attention_weighted", "Attn (resize)",       "attn_resize"),
    (False, True,  False, "attention_weighted", "Attn (compact)",      "attn_compact"),
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_weights_for_config(short_id: str) -> Tuple[float, float, float, float]:
    """Get task and ES weights for a config. Returns (task_w_sent, task_w_cat, es_w_sent, es_w_cat)."""
    if short_id in WEIGHT_OVERRIDES:
        return WEIGHT_OVERRIDES[short_id]
    else:
        return (DEFAULT_TASK_WEIGHT_SENT, DEFAULT_TASK_WEIGHT_CAT, 
                DEFAULT_ES_WEIGHT_SENT, DEFAULT_ES_WEIGHT_CAT)


def compute_sentiment_class_weights(dataset: ApcFileDataset) -> torch.Tensor:
    counts = torch.zeros(len(SENTIMENT_MAP))
    for sample in dataset.samples:
        counts[sample["sentiment_label"]] += 1
    total   = counts.sum()
    n_cls   = len(SENTIMENT_MAP)
    weights = total / (n_cls * counts.clamp(min=1))
    print(f"  Sentiment class weights: ", end="")
    for lbl, idx in sorted(SENTIMENT_MAP.items(), key=lambda x: x[1]):
        print(f"{lbl}={weights[idx]:.3f}", end="  ")
    print()
    return weights.to(DEVICE)


def compute_category_class_weights(
    dataset: ApcFileDataset,
    aspect_cat_map: Dict[str, int],
) -> torch.Tensor:
    """Inverse-frequency weights for category classes.
    Only main samples (is_supplement=False) are counted.
    """
    num_cat = len(aspect_cat_map)
    counts  = torch.zeros(num_cat)
    for sample in dataset.samples:
        if not sample["is_supplement"].item():
            counts[sample["aspect_cat_label"]] += 1
    total   = counts.sum()
    weights = total / (num_cat * counts.clamp(min=1))
    id2cat  = {v: k for k, v in aspect_cat_map.items()}
    print(f"  Category  class weights: ", end="")
    for idx in range(num_cat):
        print(f"{id2cat[idx]}={weights[idx]:.3f}", end="  ")
    print()
    return weights.to(DEVICE)





# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    sent_criterion: nn.Module,
    cat_criterion:  nn.Module,
) -> Dict:
    model.eval()
    total_loss = 0.0
    sent_pred, sent_true = [], []
    cat_pred,  cat_true  = [], []

    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            lcf  = batch["lcf_vec"].to(DEVICE)
            y_s  = batch["sentiment_label"].to(DEVICE)
            y_c  = batch["aspect_cat_label"].to(DEVICE)

            main_mask = ~batch["is_supplement"].to(DEVICE)
            out       = model(ids, attn, lcf)

            sent_loss = sent_criterion(out["sentiment_logits"], y_s)
            cat_loss  = (
                cat_criterion(out["aspect_cat_logits"][main_mask], y_c[main_mask])
                if main_mask.any() else torch.tensor(0.0, device=DEVICE)
            )
            total_loss += (sent_loss + cat_loss).item()

            sent_pred += out["sentiment_logits"].argmax(-1).cpu().tolist()
            sent_true += y_s.cpu().tolist()
            cat_pred  += out["aspect_cat_logits"][main_mask].argmax(-1).cpu().tolist()
            cat_true  += y_c[main_mask].cpu().tolist()

    n = max(len(loader), 1)
    return {
        "loss":          round(total_loss / n, 4),
        "sentiment_acc": round(accuracy_score(sent_true, sent_pred) * 100, 2),
        "sentiment_f1":  round(f1_score(sent_true, sent_pred, average="macro",  zero_division=0) * 100, 2),
        "aspect_cat_f1": round(f1_score(cat_true,  cat_pred,  average="macro",  zero_division=0) * 100, 2),
        "sent_pred": sent_pred, "sent_true": sent_true,
        "cat_pred":  cat_pred,  "cat_true":  cat_true,
    }


# ─── Joint training ───────────────────────────────────────────────────────────

def train_joint(
    use_lcf:        bool,
    use_tome:       bool,
    tome_resize:    bool,
    merge_strategy: str,
    task_weight_sent: float,
    task_weight_cat: float,
    es_weight_sent: float,
    es_weight_cat: float,
    short_id:       str,
    train_ds:       ApcFileDataset,   # main + supplement
    dev_ds:         ApcFileDataset,
    test_ds:        ApcFileDataset,
    num_sentiment:  int,
    num_aspect_cat: int,
    aspect_cat_map: Dict[str, int],
) -> Dict:
    """Train both heads jointly.

    Sentiment loss  : all samples (main + supplement) — fixes class imbalance.
    Category loss   : main samples only (supplement masked via is_supplement).
    Early stopping  : dev sentiment macro-F1.

    Returns a dict with train_time_sec, per-class F1 for both tasks.
    """
    set_seed(SEED)

    g = torch.Generator(); g.manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  generator=g)
    dev_loader   = DataLoader(dev_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    bert = AutoModel.from_pretrained(PRETRAINED_BERT)
    model = FastLcfBertMultiTask(
        bert=bert,
        num_sentiment=num_sentiment,
        num_aspect_cat=num_aspect_cat,
        use_lcf=use_lcf,
        use_tome=use_tome,
        tome_resize=tome_resize,
        tome_merge_strategy=merge_strategy,
        dropout=DROPOUT,
        num_heads=NUM_HEADS,
        tome_merge_steps=TOME_MERGE_STEPS,
    ).to(DEVICE)

    optimiser = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_MIXED_PRECISION and DEVICE.type == "cuda")

    sent_weights   = compute_sentiment_class_weights(train_ds)
    sent_criterion = nn.CrossEntropyLoss(weight=sent_weights)

    # Category weights: main samples only (supplement has no category label)
    cat_weights   = compute_category_class_weights(train_ds, aspect_cat_map)
    cat_criterion = nn.CrossEntropyLoss(weight=cat_weights)

    ckpt_dir = RUNS_DIR / short_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_dev_f1 = -1.0
    best_epoch  = 0
    no_improve  = 0
    best_state  = None

    t0 = time.perf_counter()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            ids  = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            lcf  = batch["lcf_vec"].to(DEVICE)
            y_s  = batch["sentiment_label"].to(DEVICE)
            y_c  = batch["aspect_cat_label"].to(DEVICE)

            with torch.amp.autocast("cuda", enabled=USE_MIXED_PRECISION and DEVICE.type == "cuda"):
                out = model(ids, attn, lcf)

                # Sentiment: ALL samples (main + supplement)
                sent_loss = sent_criterion(out["sentiment_logits"], y_s)

                # Category: main samples ONLY
                main_mask = ~batch["is_supplement"].to(DEVICE)
                cat_loss  = (
                    cat_criterion(out["aspect_cat_logits"][main_mask], y_c[main_mask])
                    if main_mask.any() else torch.tensor(0.0, device=DEVICE)
                )

                # Weighted combination of losses
                loss = task_weight_sent * sent_loss + task_weight_cat * cat_loss

            optimiser.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)
        dev_m    = evaluate(model, dev_loader, sent_criterion, cat_criterion)

        # Combined F1 for early stopping based on ES weights
        dev_combined_f1 = es_weight_sent * dev_m["sentiment_f1"] + es_weight_cat * dev_m["aspect_cat_f1"]

        print(
            f"    [{short_id}] epoch {epoch:2d}/{NUM_EPOCHS}"
            f"  train_loss={avg_loss:.4f}"
            f"  dev_sent_f1={dev_m['sentiment_f1']:.1f}%"
            f"  dev_cat_f1={dev_m['aspect_cat_f1']:.1f}%"
            f"  dev_combined_f1={dev_combined_f1:.1f}%"
        )

        if dev_combined_f1 > best_dev_f1 + 0.01:
            best_dev_f1 = dev_combined_f1
            best_epoch  = epoch
            no_improve  = 0
            best_state  = copy.deepcopy(model.state_dict())
            torch.save(best_state, ckpt_dir / "best_model.pt")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"    [{short_id}] early stop at epoch {epoch}")
                break

    train_time = round(time.perf_counter() - t0, 2)

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = 1

    test_m = evaluate(model, test_loader, sent_criterion, cat_criterion)

    # ── Per-class F1 ─────────────────────────────────────────────────────────
    sent_f1_per = f1_score(
        test_m["sent_true"], test_m["sent_pred"],
        average=None, zero_division=0, labels=list(range(len(SENTIMENT_LABELS)))
    ) * 100

    cat_id2label     = {v: k for k, v in aspect_cat_map.items()}
    cat_labels_order = [cat_id2label[i] for i in sorted(cat_id2label)]
    cat_f1_per = f1_score(
        test_m["cat_true"], test_m["cat_pred"],
        average=None, zero_division=0, labels=list(range(len(aspect_cat_map)))
    ) * 100

    # ── Print reports ─────────────────────────────────────────────────────────
    print(f"\n  ── Test results [{short_id}] ──")
    print("  Sentiment classification report:")
    print(classification_report(
        test_m["sent_true"], test_m["sent_pred"],
        target_names=SENTIMENT_LABELS, zero_division=0,
    ))
    print("  Aspect-category classification report:")
    print(classification_report(
        test_m["cat_true"], test_m["cat_pred"],
        target_names=cat_labels_order, zero_division=0,
    ))

    result = {
        "train_time_sec":  train_time,
        "best_epoch":      best_epoch,
        "best_dev_f1":     round(best_dev_f1, 2),
        # Sentiment overall
        "sentiment_f1":    test_m["sentiment_f1"],
        "sentiment_acc":   test_m["sentiment_acc"],
        # Sentiment per-class
        **{f"sent_f1_{SENTIMENT_LABELS[i]}": round(float(sent_f1_per[i]), 2)
           for i in range(len(SENTIMENT_LABELS))},
        # Category overall
        "aspect_cat_f1":   test_m["aspect_cat_f1"],
        # Category per-class
        **{f"cat_f1_{cat_labels_order[i]}": round(float(cat_f1_per[i]), 2)
           for i in range(len(cat_labels_order))},
    }
    return result


# ─── Summary table ────────────────────────────────────────────────────────────

def print_summary_table(
    results: List[Dict],
    labels:  List[str],
    cat_labels_order: List[str],
) -> None:
    W = 112
    print(f"\n{'═' * W}")
    print("EXPERIMENT SUMMARY — Joint training (Sent: main+supp | Cat: main only)")
    print(f"{'═' * W}")
    print(
        f"  Seed: {SEED}  MaxEpochs: {NUM_EPOCHS}  Patience: {PATIENCE}"
        f"  Batch: {BATCH_SIZE}  LR: {LR}  Device: {DEVICE}"
        f"  UseSupp: True  UseWeights: True"
    )

    # ── Overall table ─────────────────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print(f"  {'Configuration':<26} {'LCF':>4} {'Strategy':<16} {'Resize':>6}"
          f" {'Time(s)':>8} {'BestEp':>7}"
          f" {'Sent-F1':>9} {'Cat-F1':>8} {'Acc':>7}")
    print(f"{'─' * W}")

    baseline_time = next(
        (r["train_time_sec"] for r in results if not r["use_tome"]),
        None,
    )
    for r, label in zip(results, labels):
        lcf_tag      = "Y" if r["use_lcf"] else "N"
        strategy_tag = r["merge_strategy"] if r["use_tome"] else "—"
        resize_tag   = "—" if not r["use_tome"] else ("yes" if r["tome_resize"] else "NO")
        speedup = ""
        if r["use_tome"] and not r["tome_resize"] and baseline_time:
            ratio = baseline_time / max(r["train_time_sec"], 1e-6)
            speedup = f"  x{ratio:.2f}"
        print(
            f"  {label:<26} {lcf_tag:>4} {strategy_tag:<16} {resize_tag:>6}"
            f" {r['train_time_sec']:>8.1f} {r['best_epoch']:>7d}"
            f" {r['sentiment_f1']:>8.2f}% {r['aspect_cat_f1']:>7.2f}%"
            f" {r['sentiment_acc']:>6.2f}%{speedup}"
        )

    # ── Sentiment per-class table ─────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print(f"  {'Configuration':<26} {'LCF':>4}", end="")
    for lbl in SENTIMENT_LABELS:
        print(f" {('F1-'+lbl):>12}", end="")
    print()
    print(f"{'─' * W}")
    for r, label in zip(results, labels):
        lcf_tag = "Y" if r["use_lcf"] else "N"
        print(f"  {label:<26} {lcf_tag:>4}", end="")
        for lbl in SENTIMENT_LABELS:
            print(f" {r.get(f'sent_f1_{lbl}', 0):>11.2f}%", end="")
        print()

    # ── Category per-class table ──────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print(f"  {'Configuration':<26} {'LCF':>4}", end="")
    for lbl in cat_labels_order:
        print(f" {('F1-'+lbl[:7]):>10}", end="")
    print()
    print(f"{'─' * W}")
    for r, label in zip(results, labels):
        lcf_tag = "Y" if r["use_lcf"] else "N"
        print(f"  {label:<26} {lcf_tag:>4}", end="")
        for lbl in cat_labels_order:
            print(f" {r.get(f'cat_f1_{lbl}', 0):>9.2f}%", end="")
        print()

    print(f"\n{'═' * W}")
    print("  Sent training : main .apc + supplement (negative.tsv + neutral.tsv)")
    print("  Cat  training : main .apc only — supplement masked out via is_supplement")
    print("  bipartite → ToMe CVPR 2023 | sequential_local → new neighbour merge")
    print(f"{'═' * W}\n")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    for p in [TRAIN_APC, DEV_APC, TEST_APC]:
        if not p.is_file():
            raise FileNotFoundError(f"Dataset file not found: {p}")

    set_seed(SEED)
    print(f"Device  : {DEVICE}")
    print(f"Model   : {PRETRAINED_BERT}")
    print(f"Seed    : {SEED}")
    print(f"Runs dir: {RUNS_DIR}\n")

    avail_supplements = [p for p in SUPPLEMENT_FILES if Path(p).is_file()]
    missing           = [p for p in SUPPLEMENT_FILES if not Path(p).is_file()]
    if missing:
        print(f"[warn] supplement files not found (skipped): {missing}")
    if avail_supplements:
        print(f"Supplement files: {[Path(p).name for p in avail_supplements]}")

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_BERT)
    sentiment_map, aspect_cat_map = build_label_maps_from_apc(
        str(TRAIN_APC), str(DEV_APC), str(TEST_APC),
    )
    cat_id2label     = {v: k for k, v in aspect_cat_map.items()}
    cat_labels_order = [cat_id2label[i] for i in sorted(cat_id2label)]

    print(f"Sentiment classes  ({len(sentiment_map)}): {sorted(sentiment_map, key=sentiment_map.get)}")
    print(f"Aspect-cat classes ({len(aspect_cat_map)}): {cat_labels_order}")

    # train_ds: main + supplement  (sentiment head uses ALL samples)
    #           is_supplement=True for supplement rows → category head ignores them
    # dev/test: main only
    print("\nBuilding datasets …")
    train_ds = ApcFileDataset(
        str(TRAIN_APC), tokenizer, aspect_cat_map, MAX_SEQ_LEN,
        supplement_paths=avail_supplements or None,
    )
    dev_ds  = ApcFileDataset(str(DEV_APC),  tokenizer, aspect_cat_map, MAX_SEQ_LEN)
    test_ds = ApcFileDataset(str(TEST_APC), tokenizer, aspect_cat_map, MAX_SEQ_LEN)

    from collections import Counter
    cnt   = Counter(int(s["sentiment_label"]) for s in train_ds.samples)
    id2s  = {v: k for k, v in sentiment_map.items()}
    total = sum(cnt.values())
    n_sup = sum(1 for s in train_ds.samples if s["is_supplement"].item())
    print(f"  Train: {len(train_ds)} samples ({len(train_ds)-n_sup} main + {n_sup} supplement)")
    print(f"  Dev: {len(dev_ds)} | Test: {len(test_ds)}")
    print("  Train sentiment distribution:")
    for idx in sorted(cnt):
        n = cnt[idx]
        print(f"    {id2s[idx]:<10}: {n:5d}  ({n/total*100:.1f}%)")

    results: List[Dict] = []
    labels:  List[str]  = []

    print(f"\n{'═' * 70}")
    print("Joint training: Sentiment (main+supp) + Category (main only)")
    print(f"{'═' * 70}")

    for use_lcf, use_tome, tome_resize, merge_strategy, label, short_id in CONFIGS:
        # Get weights for this config
        task_weight_sent, task_weight_cat, es_weight_sent, es_weight_cat = get_weights_for_config(short_id)
        
        strategy_tag = merge_strategy if use_tome else "—"
        resize_tag   = ("resize" if tome_resize else "compact") if use_tome else "—"
        print(f"\n{'─' * 70}")
        print(f"Config : {label}  "
              f"(lcf={use_lcf}, tome={use_tome}, "
              f"strategy={strategy_tag}, resize={resize_tag})")
        print(f"  Task weights: sent={task_weight_sent}, cat={task_weight_cat} | "
              f"ES weights: sent={es_weight_sent}, cat={es_weight_cat}")
        print(f"{'─' * 70}")

        r = train_joint(
            use_lcf=use_lcf,
            use_tome=use_tome,
            tome_resize=tome_resize,
            merge_strategy=merge_strategy,
            task_weight_sent=task_weight_sent,
            task_weight_cat=task_weight_cat,
            es_weight_sent=es_weight_sent,
            es_weight_cat=es_weight_cat,
            short_id=short_id,
            train_ds=train_ds,
            dev_ds=dev_ds,
            test_ds=test_ds,
            num_sentiment=len(sentiment_map),
            num_aspect_cat=len(aspect_cat_map),
            aspect_cat_map=aspect_cat_map,
        )
        r["label"]         = label
        r["use_lcf"]       = use_lcf
        r["use_tome"]      = use_tome
        r["tome_resize"]   = tome_resize
        r["merge_strategy"] = merge_strategy
        r["task_weight_sent"] = task_weight_sent
        r["task_weight_cat"] = task_weight_cat
        r["es_weight_sent"] = es_weight_sent
        r["es_weight_cat"] = es_weight_cat

        results.append(r)
        labels.append(label)

        print(f"\n  → Train time   : {r['train_time_sec']:.1f}s  | best epoch: {r['best_epoch']}")
        print(f"  → Sentiment F1 : {r['sentiment_f1']:.2f}%"
              f"  (pos={r.get('sent_f1_positive',0):.1f}%"
              f"  neg={r.get('sent_f1_negative',0):.1f}%"
              f"  neu={r.get('sent_f1_neutral',0):.1f}%)")
        print(f"  → Category  F1 : {r['aspect_cat_f1']:.2f}%")

    # ── Save outputs ───────────────────────────────────────────────────────────
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    txt_path = RUNS_DIR / "experiment_results_joint.txt"
    with open(txt_path, "w", encoding="utf-8") as txt_f:
        tee = _Tee(sys.stdout, txt_f)
        with contextlib.redirect_stdout(tee):
            print_summary_table(results, labels, cat_labels_order)

    csv_path = RUNS_DIR / "experiment_results_joint.csv"
    fieldnames = (
        ["label", "use_lcf", "use_tome", "tome_resize", "merge_strategy",
         "task_weight_sent", "task_weight_cat", "es_weight_sent", "es_weight_cat",
         "train_time_sec", "best_epoch",
         "sentiment_f1", "sentiment_acc"]
        + [f"sent_f1_{l}" for l in SENTIMENT_LABELS]
        + ["aspect_cat_f1"]
        + [f"cat_f1_{l}" for l in cat_labels_order]
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSummary table → {txt_path}")
    print(f"Full metrics   → {csv_path}")
    print(f"Best models    → {RUNS_DIR}/<config>/best_model.pt")


if __name__ == "__main__":
    main()
