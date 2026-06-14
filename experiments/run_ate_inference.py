# -*- coding: utf-8 -*-
"""Run T5 ATE inference on the test split and save predictions to CSV.

Output (runs_ate/test_ate_predictions.csv):
    sentence       - full sentence text
    predicted_term - one predicted aspect term per row
                     (multiple rows if sentence has multiple predicted terms)
    gold_terms     - semicolon-separated gold aspect terms (reference)

Usage (from thesis_apc_baseline/ directory):
    python experiments/run_ate_inference.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ate_dataset_utils import load_ate_records
from src.model import T5AspectExtractor
from src.inference import predict_aspects

# ─── Config ───────────────────────────────────────────────────────────────────

ATE_CKPT  = ROOT / "checkpoints/gas_t5_ate/best"
TEST_APC  = ROOT / "dataset" / "test.apc"
OUT_DIR   = ROOT / "runs_ate"
OUT_CSV   = OUT_DIR / "test_ate_predictions.csv"

MAX_INPUT_LENGTH = 128
NORMALIZE        = True   # Levenshtein post-processing

# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if not ATE_CKPT.is_dir():
        raise FileNotFoundError(f"ATE checkpoint not found: {ATE_CKPT}")
    if not TEST_APC.is_file():
        raise FileNotFoundError(f"Test file not found: {TEST_APC}")

    print(f"Loading ATE model from {ATE_CKPT} …")
    ate_model = T5AspectExtractor.from_pretrained(str(ATE_CKPT))
    print(f"  Device: {ate_model.device}")

    print(f"Loading test records from {TEST_APC} …")
    records = load_ate_records(str(TEST_APC))
    print(f"  {len(records)} unique sentences")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total_gold = 0
    total_pred = 0
    tp = 0

    rows: List[Dict[str, str]] = []

    for i, rec in enumerate(records, 1):
        sentence   = str(rec["input_text"])
        gold_terms: List[str] = [t.lower() for t in (rec.get("aspects") or [])]
        gold_set   = set(gold_terms)

        pred_terms = predict_aspects(
            ate_model,
            sentence,
            max_input_length=MAX_INPUT_LENGTH,
            normalize=NORMALIZE,
        )

        if not pred_terms:
            # No aspects predicted — still emit one row to record the sentence
            rows.append({
                "sentence":        sentence,
                "predicted_term":  "",
                "gold_terms":      "; ".join(sorted(gold_terms)),
            })
        else:
            for term in pred_terms:
                rows.append({
                    "sentence":        sentence,
                    "predicted_term":  term,
                    "gold_terms":      "; ".join(sorted(gold_terms)),
                })

        pred_set = set(t.lower() for t in pred_terms)
        tp_i = len(gold_set & pred_set)
        tp         += tp_i
        total_gold += len(gold_set)
        total_pred += len(pred_set)

        if i % 50 == 0 or i == len(records):
            print(f"  [{i}/{len(records)}] sentence processed")

    # ── Write CSV ──────────────────────────────────────────────────────────────
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sentence", "predicted_term", "gold_terms"])
        writer.writeheader()
        writer.writerows(rows)

    # ── Quick metrics ──────────────────────────────────────────────────────────
    prec = tp / max(total_pred, 1)
    rec  = tp / max(total_gold, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)

    print(f"\nATE results on test set:")
    print(f"  Gold terms  : {total_gold}")
    print(f"  Pred terms  : {total_pred}")
    print(f"  TP (exact)  : {tp}")
    print(f"  Precision   : {prec*100:.2f}%")
    print(f"  Recall      : {rec*100:.2f}%")
    print(f"  F1          : {f1*100:.2f}%")
    print(f"\nSaved {len(rows)} rows → {OUT_CSV}")


if __name__ == "__main__":
    main()
