# -*- coding: utf-8 -*-
"""Train and compare all 4 LCF × ToMe configurations on the .apc dataset.

Configurations:
    1. LCF=True,  ToMe=False  →  "LCF only"
    2. LCF=True,  ToMe=True   →  "LCF + ToMe"
    3. LCF=False, ToMe=True   →  "ToMe only"
    4. LCF=False, ToMe=False  →  "Baseline"

Usage (from the parent directory of thesis_apc_baseline, e.g. kltn/):
    python thesis_apc_baseline/experiments/run_all_experiments.py

Dataset:   thesis_apc_baseline/dataset/{train,dev,test}.apc  (4-line format)
Model:     FastLcfBertMultiTask  (BERT + optional LCF + optional ToMe)
Tasks:     sentiment (3 classes) + aspect_category (N classes, from data)

Training details:
    – Dev set used for early stopping (patience=PATIENCE epochs without
      improvement on sentiment macro-F1).
    – Best checkpoint per config is saved under runs/<label>/best_model.pt
    – Final metrics reported on test set using best dev checkpoint.
    – All 4 configs use identical seed, tokeniser, and data so results
      are directly comparable.

Outputs:
    – Per-epoch dev metrics printed live
    – Summary table printed at the end
    – runs/experiment_results.csv with all metrics
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
    """Write to multiple streams at once (stdout + file)."""
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

DATASET_DIR  = ROOT / "thesis_apc_baseline" / "dataset"
TRAIN_APC    = DATASET_DIR / "train.apc"
DEV_APC      = DATASET_DIR / "dev.apc"
TEST_APC     = DATASET_DIR / "test.apc"
RUNS_DIR     = ROOT / "thesis_apc_baseline" / "runs"
SUPPLEMENT_DIR = DATASET_DIR / "supplement"

# Supplement TSV files — appended to TRAINING ONLY (not dev/test)
# Each file contains samples for one under-represented sentiment class.
SUPPLEMENT_FILES: List[str] = [
    str(SUPPLEMENT_DIR / "negative.tsv"),
    str(SUPPLEMENT_DIR / "neutral.tsv"),
]
# No SUPPLEMENT_CATEGORY — supplement samples are flagged with is_supplement=True
# and excluded from category-head training via main_mask in the loss step.

# ─── Hyperparameters ──────────────────────────────────────────────────────────

PRETRAINED_BERT  = "bert-base-uncased"
SEED             = 42
NUM_EPOCHS       = 3           # max epochs; early stopping may exit sooner
PATIENCE         = 3           # stop if dev sentiment-F1 does not improve
BATCH_SIZE       = 16
LR               = 2e-5
MAX_SEQ_LEN      = 128
DROPOUT          = 0.1
NUM_HEADS        = 8
TOME_MERGE_STEPS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Each config: (use_lcf, use_tome, tome_resize, merge_strategy, display_name, short_id)
#
# merge_strategy:
#   "bipartite"        → ToMe (CVPR 2023): alternating-group cosine matching
#   "sequential_local" → new: left-to-right neighbour comparison
#
# tome_resize:
#   True  → interpolate back to L=128 after merging  (accuracy focus)
#   False → keep compact L'<128                       (real speedup)
#
# Joint multi-task configs: both heads trained together.
# (use_lcf, use_tome, tome_resize, merge_strategy, display_name, short_id)
CONFIGS: List[Tuple[bool, bool, bool, str, str, str]] = [
    # ── Reference configs (no merging) ────────────────────────────────────────
    (False, False, True,  "bipartite",        "Joint Baseline",        "baseline"),
    (True,  False, True,  "bipartite",        "Joint LCF only",        "lcf_only"),
    # ── Bipartite ToMe — with LCF ─────────────────────────────────────────────
    (True,  True,  True,  "bipartite",        "LCF+Bip (resize)",      "lcf_bip_resize"),
    (True,  True,  False, "bipartite",        "LCF+Bip (compact)",     "lcf_bip_compact"),
    # ── Bipartite ToMe — without LCF ─────────────────────────────────────────
    (False, True,  True,  "bipartite",        "Bip (resize)",          "bip_resize"),
    (False, True,  False, "bipartite",        "Bip (compact)",         "bip_compact"),
    # ── Sequential local-neighbour merge — with LCF ───────────────────────────
    (True,  True,  True,  "sequential_local", "LCF+Seq (resize)",      "lcf_seq_resize"),
    (True,  True,  False, "sequential_local", "LCF+Seq (compact)",     "lcf_seq_compact"),
    # ── Sequential local-neighbour merge — without LCF ───────────────────────
    (False, True,  True,  "sequential_local", "Seq (resize)",          "seq_resize"),
    (False, True,  False, "sequential_local", "Seq (compact)",         "seq_compact"),
]


# ─── Class-weight helpers ─────────────────────────────────────────────────────

def compute_sentiment_class_weights(dataset: ApcFileDataset) -> torch.Tensor:
    """Compute inverse-frequency weights for sentiment classes.

    Formula: weight[c] = total_samples / (n_classes * count[c])
    Only main samples (is_supplement=False) are counted — supplement samples
    are real sentiment signal but skew the distribution intentionally, so
    we count all samples (incl. supplement) to reflect the actual training mix.

    Returns a tensor of shape (n_sentiment_classes,) on DEVICE.
    """
    counts = torch.zeros(len(SENTIMENT_MAP))
    for sample in dataset.samples:
        counts[sample["sentiment_label"]] += 1
    total   = counts.sum()
    n_cls   = len(SENTIMENT_MAP)
    weights = total / (n_cls * counts.clamp(min=1))
    print(f"  Sentiment class weights: ", end="")
    for label, idx in sorted(SENTIMENT_MAP.items(), key=lambda x: x[1]):
        print(f"{label}={weights[idx]:.3f}", end="  ")
    print()
    return weights.to(DEVICE)


def compute_category_class_weights(
    dataset: ApcFileDataset,
    aspect_cat_map: Dict[str, int],
) -> torch.Tensor:
    """Compute inverse-frequency weights for aspect-category classes.

    Only MAIN samples (is_supplement=False) are counted because supplement
    samples have no valid category label and are excluded from category training.

    Formula: weight[c] = total_main / (n_classes * count[c])

    Returns a tensor of shape (num_aspect_cat,) on DEVICE.
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


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Evaluation helper ────────────────────────────────────────────────────────

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    sent_criterion: nn.Module,
    cat_criterion:  nn.Module,
) -> Dict[str, float]:
    """Run model on loader; return loss + macro-F1 for sentiment and aspect_cat."""
    model.eval()
    total_loss = 0.0
    sent_pred, sent_true = [], []
    cat_pred,  cat_true  = [], []

    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(DEVICE)
            attn  = batch["attention_mask"].to(DEVICE)
            lcf   = batch["lcf_vec"].to(DEVICE)
            y_s   = batch["sentiment_label"].to(DEVICE)
            y_c   = batch["aspect_cat_label"].to(DEVICE)

            # mask out supplement samples from category evaluation
            # (dev/test typically have none, but be defensive)
            main_mask = ~batch["is_supplement"].to(DEVICE)

            out      = model(ids, attn, lcf)
            sent_loss = sent_criterion(out["sentiment_logits"], y_s)
            if main_mask.any():
                cat_loss = cat_criterion(
                    out["aspect_cat_logits"][main_mask], y_c[main_mask]
                )
            else:
                cat_loss = torch.tensor(0.0, device=DEVICE)
            total_loss += (sent_loss + cat_loss).item()

            sent_pred += out["sentiment_logits"].argmax(-1).cpu().tolist()
            sent_true += y_s.cpu().tolist()
            # only record category predictions for main samples
            cat_pred  += out["aspect_cat_logits"][main_mask].argmax(-1).cpu().tolist()
            cat_true  += y_c[main_mask].cpu().tolist()

    n = max(len(loader), 1)
    return {
        "loss":          round(total_loss / n, 4),
        "sentiment_acc": round(accuracy_score(sent_true, sent_pred) * 100, 2),
        "sentiment_f1":  round(f1_score(sent_true, sent_pred, average="macro", zero_division=0) * 100, 2),
        "aspect_cat_f1": round(f1_score(cat_true,  cat_pred,  average="macro", zero_division=0) * 100, 2),
        "sent_pred": sent_pred,
        "sent_true": sent_true,
        "cat_pred":  cat_pred,
        "cat_true":  cat_true,
    }


# ─── Single-task training ──────────────────────────────────────────────────────

def train_one_task(
    task:              str,          # "sentiment" | "category"
    use_lcf:           bool,
    use_tome:          bool,
    tome_resize:       bool,
    merge_strategy:    str,
    short_id:          str,          # checkpoint sub-directory name
    train_ds:          ApcFileDataset,
    dev_ds:            ApcFileDataset,
    test_ds:           ApcFileDataset,
    num_sentiment:     int,
    num_aspect_cat:    int,
    aspect_cat_map:    Dict[str, int],
) -> Dict:
    """Train ONE task head with dev-based early stopping.

    task="sentiment":
        - Dataset : train_ds should contain main + supplement (all data).
        - Loss    : weighted CrossEntropyLoss on sentiment logits.
        - Monitor : dev sentiment macro-F1.
        - Reports : sentiment classification report on test set.

    task="category":
        - Dataset : train_ds should contain main .apc samples ONLY (no supplement).
        - Loss    : CrossEntropyLoss on aspect-category logits.
        - Monitor : dev aspect-category macro-F1.
        - Reports : category classification report on test set.

    Returns a dict with train_time_sec, best_epoch, and test metrics.
    """
    assert task in ("sentiment", "category"), f"task must be 'sentiment' or 'category', got {task!r}"

    set_seed(SEED)   # identical seed → fair comparison across configs

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  generator=g)
    dev_loader   = DataLoader(dev_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # Fresh BERT + both heads (only the relevant head's loss is used for backprop)
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

    # Criteria
    sent_weights   = compute_sentiment_class_weights(train_ds)
    sent_criterion = nn.CrossEntropyLoss(weight=sent_weights)

    # Category weights: computed from main samples only (supplement has no category)
    cat_weights   = compute_category_class_weights(train_ds, aspect_cat_map)
    cat_criterion = nn.CrossEntropyLoss(weight=cat_weights)

    ckpt_dir = RUNS_DIR / short_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "best_model.pt"

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

            out = model(ids, attn, lcf)

            if task == "sentiment":
                # ALL samples (main + supplement) train the sentiment head
                loss = sent_criterion(out["sentiment_logits"], y_s)
            else:
                # Category training: main samples only (supplement not in train_ds here)
                main_mask = ~batch["is_supplement"].to(DEVICE)
                if main_mask.any():
                    loss = cat_criterion(
                        out["aspect_cat_logits"][main_mask], y_c[main_mask]
                    )
                else:
                    loss = torch.tensor(0.0, device=DEVICE)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / max(len(train_loader), 1)

        dev_m = evaluate(model, dev_loader, sent_criterion, cat_criterion)
        monitor_f1 = dev_m["sentiment_f1"] if task == "sentiment" else dev_m["aspect_cat_f1"]
        monitor_tag = "sent_f1" if task == "sentiment" else "cat_f1"

        print(
            f"    [{short_id}] epoch {epoch:2d}/{NUM_EPOCHS}"
            f"  train_loss={avg_train_loss:.4f}"
            f"  dev_{monitor_tag}={monitor_f1:.1f}%"
        )

        if monitor_f1 > best_dev_f1 + 0.01:
            best_dev_f1 = monitor_f1
            best_epoch  = epoch
            no_improve  = 0
            best_state  = copy.deepcopy(model.state_dict())
            torch.save(best_state, best_ckpt_path)
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

    # ── Classification report for the trained task only ───────────────────────
    cat_id2label     = {v: k for k, v in aspect_cat_map.items()}
    cat_labels_order = [cat_id2label[i] for i in sorted(cat_id2label)]

    print(f"\n  ── Test results [{short_id}] ──")
    if task == "sentiment":
        print("  Sentiment classification report:")
        print(classification_report(
            test_m["sent_true"], test_m["sent_pred"],
            target_names=SENTIMENT_LABELS, zero_division=0,
        ))
    else:
        print("  Aspect-category classification report:")
        print(classification_report(
            test_m["cat_true"], test_m["cat_pred"],
            target_names=cat_labels_order, zero_division=0,
        ))

    return {
        "train_time_sec": train_time,
        "best_epoch":     best_epoch,
        "best_dev_f1":    round(best_dev_f1, 2),
        "sentiment_acc":  test_m["sentiment_acc"],
        "sentiment_f1":   test_m["sentiment_f1"],
        "aspect_cat_f1":  test_m["aspect_cat_f1"],
    }


# ─── Summary table ────────────────────────────────────────────────────────────

def print_summary_table(results: List[Dict], labels: List[str]) -> None:
    W = 110
    print(f"\n{'═' * W}")
    print("EXPERIMENT SUMMARY — Sentiment (main+supplement) vs Category (main only) | Separate training")
    print(f"{'═' * W}")
    print(
        f"  Seed: {SEED}  MaxEpochs: {NUM_EPOCHS}  Patience: {PATIENCE}"
        f"  Batch: {BATCH_SIZE}  LR: {LR}  Device: {DEVICE}"
    )
    print("─" * W)
    print(
        f"{'Configuration':<26} {'LCF':>4} {'Strategy':<16} {'Resize':>6}"
        f" | {'Sent-Time':>10} {'Sent-F1':>8} {'S-Ep':>5}"
        f" | {'Cat-Time':>9} {'Cat-F1':>8} {'C-Ep':>5}"
    )
    print("─" * W)

    # Baseline = no-LCF no-ToMe, for speedup ratio
    baseline = next((r for r in results if not r["use_tome"] and not r["use_lcf"]), None)

    for r, label in zip(results, labels):
        lcf_tag      = "Y" if r["use_lcf"] else "N"
        strategy_tag = r["merge_strategy"] if r["use_tome"] else "—"
        resize_tag   = "—" if not r["use_tome"] else ("yes" if r["tome_resize"] else "NO")

        speedup = ""
        if r["use_tome"] and not r["tome_resize"] and baseline:
            # speedup on category training time (shorter sequences → real speedup)
            bt = baseline.get("cat_train_time", baseline.get("sent_train_time", 1))
            ct = r.get("cat_train_time", 1)
            if ct > 0:
                speedup = f" x{bt/ct:.2f}"

        print(
            f"{label:<26} {lcf_tag:>4} {strategy_tag:<16} {resize_tag:>6}"
            f" | {r.get('sent_train_time', 0):>10.1f}s"
            f" {r.get('sentiment_f1', 0):>7.2f}%"
            f" {r.get('sent_best_epoch', 0):>5d}"
            f" | {r.get('cat_train_time', 0):>9.1f}s"
            f" {r.get('aspect_cat_f1', 0):>7.2f}%"
            f" {r.get('cat_best_epoch', 0):>5d}{speedup}"
        )
    print(f"{'═' * W}")
    print("  Sent training : main .apc + supplement (negative.tsv + neutral.tsv) — fixes class imbalance")
    print("  Cat  training : main .apc only — no supplement data")
    print("  bipartite → ToMe CVPR 2023 | sequential_local → new neighbour merge")
    print(f"{'═' * W}\n")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    # Sanity-check dataset files
    for p in [TRAIN_APC, DEV_APC, TEST_APC]:
        if not p.is_file():
            raise FileNotFoundError(
                f"Dataset file not found: {p}\n"
                f"Expected 4-line .apc files under {DATASET_DIR}"
            )

    set_seed(SEED)
    print(f"Device  : {DEVICE}")
    print(f"Seed    : {SEED}")
    print(f"Dataset : {DATASET_DIR}\n")

    # Check supplement files
    avail_supplements = [p for p in SUPPLEMENT_FILES if Path(p).is_file()]
    missing = [p for p in SUPPLEMENT_FILES if not Path(p).is_file()]
    if missing:
        print(f"[warn] supplement files not found (skipped): {missing}")
    if avail_supplements:
        print(f"Supplement files: {[Path(p).name for p in avail_supplements]}")

    # Build label maps from main .apc files only.
    # Supplement samples are flagged is_supplement=True and excluded from
    # category training, so no dummy category is needed.
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_BERT)
    sentiment_map, aspect_cat_map = build_label_maps_from_apc(
        str(TRAIN_APC), str(DEV_APC), str(TEST_APC),
    )
    print(
        f"Sentiment classes  ({len(sentiment_map)}): "
        f"{sorted(sentiment_map, key=sentiment_map.get)}"
    )
    print(
        f"Aspect-cat classes ({len(aspect_cat_map)}): "
        f"{sorted(aspect_cat_map, key=aspect_cat_map.get)}"
    )

    # ── Build datasets ─────────────────────────────────────────────────────────
    # train_sent : main .apc + supplement  → sentiment training (all data)
    # train_cat  : main .apc only          → category training  (no supplement)
    # dev / test : main .apc only          → evaluation (no supplement)
    print("\nBuilding datasets …")
    from collections import Counter

    train_sent_ds = ApcFileDataset(
        str(TRAIN_APC), tokenizer, aspect_cat_map, MAX_SEQ_LEN,
        supplement_paths=avail_supplements or None,
    )
    train_cat_ds = ApcFileDataset(
        str(TRAIN_APC), tokenizer, aspect_cat_map, MAX_SEQ_LEN,
        # no supplement_paths → main samples only
    )
    dev_ds  = ApcFileDataset(str(DEV_APC),  tokenizer, aspect_cat_map, MAX_SEQ_LEN)
    test_ds = ApcFileDataset(str(TEST_APC), tokenizer, aspect_cat_map, MAX_SEQ_LEN)

    print(
        f"  Train-Sent : {len(train_sent_ds)} samples (main + supplement)"
        f" | Train-Cat: {len(train_cat_ds)} samples (main only)"
        f"\n  Dev: {len(dev_ds)} | Test: {len(test_ds)}"
    )

    # Show sentiment distribution for each training set
    for tag, ds in [("Train-Sent", train_sent_ds), ("Train-Cat", train_cat_ds)]:
        cnt    = Counter(int(s["sentiment_label"]) for s in ds.samples)
        id2s   = {v: k for k, v in sentiment_map.items()}
        total  = sum(cnt.values())
        parts  = "  ".join(f"{id2s[i]}={cnt[i]/total*100:.1f}%" for i in sorted(cnt))
        print(f"  {tag} sentiment: {parts}")

    results: List[Dict] = []
    labels:  List[str]  = []

    # ── Per-config: train sentiment separately, then category separately ───────
    print(f"\n{'═' * 70}")
    print("Training: Sentiment (main+supplement) & Category (main only) — separately")
    print(f"{'═' * 70}")

    common_kwargs = dict(
        dev_ds=dev_ds,
        test_ds=test_ds,
        num_sentiment=len(sentiment_map),
        num_aspect_cat=len(aspect_cat_map),
        aspect_cat_map=aspect_cat_map,
    )

    for use_lcf, use_tome, tome_resize, merge_strategy, label, short_id in CONFIGS:
        strategy_tag = merge_strategy if use_tome else "—"
        resize_tag   = ("resize" if tome_resize else "compact") if use_tome else "—"
        print(f"\n{'─' * 70}")
        print(f"Config : {label}  (lcf={use_lcf}, tome={use_tome}, "
              f"strategy={strategy_tag}, resize={resize_tag})")
        print(f"{'─' * 70}")

        # ── 1. Sentiment training (main + supplement) ──────────────────────────
        print(f"\n  [Sentiment] {label} …")
        r_sent = train_one_task(
            task="sentiment",
            use_lcf=use_lcf, use_tome=use_tome,
            tome_resize=tome_resize, merge_strategy=merge_strategy,
            short_id=short_id + "_sent",
            train_ds=train_sent_ds,
            **common_kwargs,
        )
        print(f"  → Sent train time : {r_sent['train_time_sec']:.1f}s  "
              f"| best epoch: {r_sent['best_epoch']}  "
              f"| test Sent-F1: {r_sent['sentiment_f1']:.2f}%")

        # ── 2. Category training (main only) ───────────────────────────────────
        print(f"\n  [Category] {label} …")
        r_cat = train_one_task(
            task="category",
            use_lcf=use_lcf, use_tome=use_tome,
            tome_resize=tome_resize, merge_strategy=merge_strategy,
            short_id=short_id + "_cat",
            train_ds=train_cat_ds,
            **common_kwargs,
        )
        print(f"  → Cat  train time : {r_cat['train_time_sec']:.1f}s  "
              f"| best epoch: {r_cat['best_epoch']}  "
              f"| test Cat-F1: {r_cat['aspect_cat_f1']:.2f}%")

        # Combine into one result row per config
        result = {
            "label":           label,
            "use_lcf":         use_lcf,
            "use_tome":        use_tome,
            "tome_resize":     tome_resize,
            "merge_strategy":  merge_strategy,
            # Sentiment task
            "sent_train_time": r_sent["train_time_sec"],
            "sent_best_epoch": r_sent["best_epoch"],
            "sentiment_f1":    r_sent["sentiment_f1"],
            "sentiment_acc":   r_sent["sentiment_acc"],
            # Category task
            "cat_train_time":  r_cat["train_time_sec"],
            "cat_best_epoch":  r_cat["best_epoch"],
            "aspect_cat_f1":   r_cat["aspect_cat_f1"],
        }
        results.append(result)
        labels.append(label)

    # ── Save outputs ───────────────────────────────────────────────────────────
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Summary table → printed to screen AND saved to .txt
    txt_path = RUNS_DIR / "experiment_results.txt"
    with open(txt_path, "w", encoding="utf-8") as txt_f:
        tee = _Tee(sys.stdout, txt_f)
        with contextlib.redirect_stdout(tee):
            print_summary_table(results, labels)

    # 2. Full metrics → CSV (one row per config, easy to open in Excel/pandas)
    csv_path = RUNS_DIR / "experiment_results.csv"
    fieldnames = [
        "label", "use_lcf", "use_tome", "tome_resize", "merge_strategy",
        "sent_train_time", "sent_best_epoch", "sentiment_f1", "sentiment_acc",
        "cat_train_time",  "cat_best_epoch",  "aspect_cat_f1",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSummary table → {txt_path}")
    print(f"Full metrics   → {csv_path}")
    print(f"Best models    → {RUNS_DIR}/<config>/best_model.pt")


if __name__ == "__main__":
    main()
