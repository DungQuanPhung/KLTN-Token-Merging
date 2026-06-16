# -*- coding: utf-8 -*-
"""Evaluate joint model trên dataset/test.csv theo logic sentence-level.

Logic đánh giá:
    - Input mỗi row: unit (câu trích) + aspect_terms (ground truth term)
    - Model predict: categories + sentiments
    - Một unit ĐÚNG khi: predicted_category == gt_category
                          AND predicted_sentiment == gt_sentiment
      (aspect_term luôn được coi là "đúng" vì nó là input ground truth)
    - Một sentence ĐÚNG khi: TẤT CẢ các unit trong câu đó đều ĐÚNG
      (tức là không có unit nào sai category hoặc sentiment)

Metrics báo cáo:
    Unit-level  : accuracy, macro-F1 (category), macro-F1 (sentiment)
    Sentence-level : accuracy (% câu có tất cả unit đúng)
    Triplet exact-match: micro P/R/F1 — mỗi (aspect_terms, category, sentiment)
                         phải khớp chính xác trong câu tương ứng

Usage (từ thư mục gốc kltn/):
    python thesis_apc_baseline/experiments/eval_csv_sentence.py
    python thesis_apc_baseline/experiments/eval_csv_sentence.py --run-dir runs_joint/baseline
    python thesis_apc_baseline/experiments/eval_csv_sentence.py --list
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, T5EncoderModel, AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score, classification_report

from dataset_utils import SENTIMENT_MAP, SENTIMENT_LABELS
from models.fast_lcf_bert_multitask import FastLcfBertMultiTask

# ─── Paths & constants ────────────────────────────────────────────────────────

TEST_CSV  = ROOT / "dataset" / "test.csv"
RUNS_DIR  = ROOT / "runs_joint"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQ_LEN = 128
BATCH_SIZE  = 32

_DEFAULT_CFG = dict(dropout=0.1, num_heads=8, srd_threshold=5,
                    tome_merge_steps=2, pre_tome_merge_steps=1)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class CsvUnitDataset(Dataset):
    """Mỗi row trong test.csv là một sample.

    Encoding SPC: [CLS] unit [SEP] aspect_terms [SEP]
    LCF vector  : token_type_ids (segment B = aspect_terms → 1.0)
    """

    def __init__(self, df: pd.DataFrame, tokenizer, aspect_cat_map: Dict[str, int],
                 max_seq_len: int = 128) -> None:
        pad = tokenizer.pad_token or "[PAD]"
        self.samples: list = []
        self.meta: list = []  # (sentence, unit, aspect_terms, gt_category, gt_sentiment)

        skipped = 0
        for _, row in df.iterrows():
            unit          = str(row["unit"]).strip()
            aspect        = str(row["aspect_terms"]).strip() or pad
            sentiment_str = str(row["sentiments"]).strip().lower()
            category_str  = str(row["categories"]).strip().upper()
            sentence      = str(row["sentence"]).strip()

            if sentiment_str not in SENTIMENT_MAP:
                skipped += 1
                continue
            if category_str not in aspect_cat_map:
                skipped += 1
                continue

            enc = tokenizer(
                unit, aspect,
                max_length=max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids      = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            token_type_ids = enc.get(
                "token_type_ids", torch.zeros_like(input_ids)
            ).squeeze(0)
            lcf_vec = token_type_ids.float()

            self.samples.append({
                "input_ids":        input_ids,
                "attention_mask":   attention_mask,
                "lcf_vec":          lcf_vec,
                "sentiment_label":  torch.tensor(SENTIMENT_MAP[sentiment_str], dtype=torch.long),
                "aspect_cat_label": torch.tensor(aspect_cat_map[category_str],  dtype=torch.long),
            })
            self.meta.append((sentence, unit, aspect, category_str, sentiment_str))

        if skipped:
            print(f"[warn] Skipped {skipped} rows (unknown category or sentiment label)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


# ─── Load model ───────────────────────────────────────────────────────────────

def _load_encoder(model_type: str, pretrained: str):
    if model_type == "t5":
        return T5EncoderModel.from_pretrained(pretrained)
    return AutoModel.from_pretrained(pretrained)


def load_model_from_run(run_dir: Path):
    """Load best_model.pt + meta.json. Returns (model, cat_labels, tokenizer)."""
    meta_path = run_dir / "meta.json"
    ckpt_path = run_dir / "best_model.pt"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.json not found in {run_dir}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"best_model.pt not found in {run_dir}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cfg  = meta["config"]
    cat_labels: List[str] = meta["category_labels"]

    # Detect encoder type from pretrained name stored (fallback: t5)
    pretrained   = meta.get("pretrained", "t5-base")
    model_type   = "t5" if "t5" in pretrained.lower() else "bert"
    encoder      = _load_encoder(model_type, pretrained)
    tokenizer    = AutoTokenizer.from_pretrained(pretrained)

    model = FastLcfBertMultiTask(
        bert=encoder,
        num_sentiment=len(SENTIMENT_LABELS),
        num_aspect_cat=len(cat_labels),
        use_lcf=cfg.get("use_lcf", True),
        use_cdm=cfg.get("use_cdm", True),
        use_tome=cfg.get("use_tome", False),
        tome_resize=cfg.get("tome_resize", True),
        tome_merge_strategy=cfg.get("merge_strategy", "bipartite"),
        use_pre_tome=cfg.get("use_pre_tome", False),
        pre_tome_merge_steps=_DEFAULT_CFG["pre_tome_merge_steps"],
        pre_tome_merge_strategy=cfg.get("merge_strategy", "bipartite"),
        pre_tome_resize=cfg.get("tome_resize", True),
        dropout=_DEFAULT_CFG["dropout"],
        num_heads=_DEFAULT_CFG["num_heads"],
        tome_merge_steps=_DEFAULT_CFG["tome_merge_steps"],
        srd_threshold=_DEFAULT_CFG["srd_threshold"],
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model, cat_labels, tokenizer


# ─── Inference ────────────────────────────────────────────────────────────────

def run_inference(model, loader: DataLoader):
    """Return lists of predicted category idx and sentiment idx."""
    cat_preds, sent_preds = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            lcf  = batch["lcf_vec"].to(DEVICE)
            out  = model(ids, attn, lcf)
            cat_preds  += out["aspect_cat_logits"].argmax(-1).cpu().tolist()
            sent_preds += out["sentiment_logits"].argmax(-1).cpu().tolist()
    return cat_preds, sent_preds


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(
    dataset: CsvUnitDataset,
    cat_preds: List[int],
    sent_preds: List[int],
    cat_labels: List[str],
) -> None:
    cat_id2label = {i: lbl for i, lbl in enumerate(cat_labels)}
    cat_map      = {lbl: i for i, lbl in enumerate(cat_labels)}

    gt_cats  = [dataset.samples[i]["aspect_cat_label"].item() for i in range(len(dataset))]
    gt_sents = [dataset.samples[i]["sentiment_label"].item()  for i in range(len(dataset))]

    # ── Unit-level ────────────────────────────────────────────────────────────
    unit_correct = [
        cp == gc and sp == gs
        for cp, gc, sp, gs in zip(cat_preds, gt_cats, sent_preds, gt_sents)
    ]
    unit_acc = sum(unit_correct) / max(len(unit_correct), 1) * 100

    cat_acc  = accuracy_score(gt_cats,  cat_preds) * 100
    sent_acc = accuracy_score(gt_sents, sent_preds) * 100
    cat_f1   = f1_score(gt_cats,  cat_preds,  average="macro", zero_division=0) * 100
    sent_f1  = f1_score(gt_sents, sent_preds, average="macro", zero_division=0) * 100

    # ── Sentence-level ────────────────────────────────────────────────────────
    # Group by sentence; sentence correct = all units correct
    sent_unit_correct: Dict[str, List[bool]] = defaultdict(list)
    for i, (sentence, unit, aspect, gt_cat, gt_sent) in enumerate(dataset.meta):
        sent_unit_correct[sentence].append(unit_correct[i])

    n_sentences = len(sent_unit_correct)
    n_sent_correct = sum(all(v) for v in sent_unit_correct.values())
    sent_level_acc = n_sent_correct / max(n_sentences, 1) * 100

    # ── Triplet exact-match (micro P/R/F1) ────────────────────────────────────
    # Gold triplets per sentence: {sentence: {(aspect_term, category, sentiment)}}
    gold_by_sent: Dict[str, Set[Tuple[str, str, str]]] = defaultdict(set)
    pred_by_sent: Dict[str, Set[Tuple[str, str, str]]] = defaultdict(set)

    for i, (sentence, unit, aspect, gt_cat, gt_sent) in enumerate(dataset.meta):
        gold_by_sent[sentence].add((aspect.lower(), gt_cat.upper(), gt_sent.lower()))
        pred_cat  = cat_id2label[cat_preds[i]].upper()
        pred_sent = SENTIMENT_LABELS[sent_preds[i]].lower()
        pred_by_sent[sentence].add((aspect.lower(), pred_cat, pred_sent))

    total_tp = total_fp = total_fn = 0
    for sent in gold_by_sent:
        gold = gold_by_sent[sent]
        pred = pred_by_sent.get(sent, set())
        tp = len(gold & pred)
        total_tp += tp
        total_fp += len(pred) - tp
        total_fn += len(gold) - tp

    p  = total_tp / max(total_tp + total_fp, 1)
    r  = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)

    # ── Print ─────────────────────────────────────────────────────────────────
    W = 70
    print(f"\n{'═' * W}")
    print(f"  Samples (units) : {len(dataset)}")
    print(f"  Sentences       : {n_sentences}")
    print(f"{'─' * W}")

    print(f"\n  ── Unit-level ──")
    print(f"  Category  Acc    : {cat_acc:.2f}%   Macro-F1 : {cat_f1:.2f}%")
    print(f"  Sentiment Acc    : {sent_acc:.2f}%   Macro-F1 : {sent_f1:.2f}%")
    print(f"  Both correct     : {unit_acc:.2f}%  ({sum(unit_correct)}/{len(unit_correct)} units)")

    print(f"\n  ── Sentence-level (tất cả unit trong câu phải đúng) ──")
    print(f"  Sentence Acc     : {sent_level_acc:.2f}%"
          f"  ({n_sent_correct}/{n_sentences} sentences)")

    print(f"\n  ── Triplet exact-match micro P/R/F1 ──")
    print(f"  (aspect_term, category, sentiment) cả 3 phải khớp trong câu")
    print(f"  Precision : {p*100:.2f}%")
    print(f"  Recall    : {r*100:.2f}%")
    print(f"  F1        : {f1*100:.2f}%")
    print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")

    print(f"\n  ── Category classification report ──")
    print(classification_report(
        gt_cats, cat_preds,
        target_names=cat_labels, zero_division=0,
    ))

    print(f"  ── Sentiment classification report ──")
    print(classification_report(
        gt_sents, sent_preds,
        target_names=SENTIMENT_LABELS, zero_division=0,
    ))
    print(f"{'═' * W}")

    # ── Sentence breakdown (wrong sentences) ──────────────────────────────────
    wrong_sents = [s for s, v in sent_unit_correct.items() if not all(v)]
    if wrong_sents:
        print(f"\n  Sentences có ít nhất 1 unit sai: {len(wrong_sents)}")
        for s in wrong_sents[:5]:
            n_wrong = sum(1 for x in sent_unit_correct[s] if not x)
            n_total = len(sent_unit_correct[s])
            print(f"    [{n_wrong}/{n_total} unit sai] {s[:80]}{'…' if len(s)>80 else ''}")
        if len(wrong_sents) > 5:
            print(f"    ... và {len(wrong_sents)-5} câu khác")


# ─── Entry point ──────────────────────────────────────────────────────────────

def list_runs() -> None:
    runs = sorted(d for d in RUNS_DIR.iterdir()
                  if d.is_dir() and (d/"best_model.pt").is_file() and (d/"meta.json").is_file())
    if not runs:
        print(f"Không tìm thấy run nào trong {RUNS_DIR}")
        return
    print("Available runs:")
    for r in runs:
        meta = json.loads((r/"meta.json").read_text(encoding="utf-8"))
        print(f"  {r.name:<30}  best_dev_f1={meta.get('best_dev_f1','?')}%"
              f"  epoch={meta.get('best_epoch','?')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on test.csv (sentence-level)")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Path to run dir (default: first found in runs_joint/)")
    parser.add_argument("--list", action="store_true",
                        help="List available run directories and exit")
    args = parser.parse_args()

    if args.list:
        list_runs()
        return

    # Resolve run dir
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
    else:
        candidates = sorted(
            d for d in RUNS_DIR.iterdir()
            if d.is_dir() and (d/"best_model.pt").is_file() and (d/"meta.json").is_file()
        )
        if not candidates:
            print(f"[error] Không tìm thấy run nào trong {RUNS_DIR}")
            print("Chạy với --list để xem danh sách, hoặc --run-dir để chỉ định.")
            sys.exit(1)
        run_dir = candidates[0]

    if not TEST_CSV.is_file():
        print(f"[error] Không tìm thấy {TEST_CSV}")
        sys.exit(1)

    print(f"Device  : {DEVICE}")
    print(f"Run dir : {run_dir.relative_to(ROOT)}")
    print(f"Test CSV: {TEST_CSV.relative_to(ROOT)}")

    # Load model
    print("\nLoading model …")
    model, cat_labels, tokenizer = load_model_from_run(run_dir)
    cat_map = {lbl: i for i, lbl in enumerate(cat_labels)}
    print(f"  Category labels : {cat_labels}")
    print(f"  Sentiment labels: {SENTIMENT_LABELS}")

    # Load CSV
    df = pd.read_csv(TEST_CSV)
    df = df.dropna(subset=["unit", "sentiments", "categories"])
    print(f"\nLoaded {len(df)} rows from test.csv")

    dataset = CsvUnitDataset(df, tokenizer, cat_map, MAX_SEQ_LEN)
    print(f"  {len(dataset)} valid samples | "
          f"{len(set(m[0] for m in dataset.meta))} unique sentences")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Inference
    print("\nRunning inference …")
    cat_preds, sent_preds = run_inference(model, loader)

    # Metrics
    compute_metrics(dataset, cat_preds, sent_preds, cat_labels)


if __name__ == "__main__":
    main()
