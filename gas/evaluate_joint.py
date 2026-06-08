# -*- coding: utf-8 -*-
"""End-to-end 3-label evaluation: Aspect Term / Category / Sentiment / Joint.

Pipeline
--------
Step 1 — GAS model (T5):
    Input  : raw sentence
    Output : [(aspect_term, sentiment), ...]   (from GAS decoder)

Step 2 — BERT APC model (category predictor):
    Input  : (sentence, aspect_term)
    Output : category label

Step 3 — Combine:
    predicted  : (aspect_term, category, sentiment)  per sentence
    gold       : (aspect_term, category, sentiment)  from .apc file

Step 4 — Metrics (4 layers):
    Aspect Term   P/R/F1  — exact aspect term match
    Sentiment     P/R/F1  — (aspect_term, sentiment) pair match
    Category      P/R/F1  — (aspect_term, category) pair match
    Joint         P/R/F1  — all 3 labels correct simultaneously

Usage
-----
    python gas/evaluate_joint.py \\
        --gas-checkpoint  checkpoints/gas_t5/best \\
        --apc-checkpoint  runs_joint/lcf_bip_resize \\
        --data-dir        dataset \\
        --split           test

The GAS checkpoint must be a directory produced by ``gas/train_gas.py``
(contains T5 weights + tokenizer).

The APC checkpoint must be a directory produced by
``experiments/run_joint_experiments.py``
(contains best_model.pt + meta.json).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from gas.dataset import load_split_records
from gas.metrics import evaluate_all, format_report
from gas.model import GasT5Model
from pipeline_inference import APCPredictor


# ─── Core evaluation logic ────────────────────────────────────────────────────

def run_joint_evaluation(
    gas_model:  GasT5Model,
    apc_model:  APCPredictor,
    records:    List[Dict],
    normalize:  bool = True,
) -> Dict[str, Dict[str, float]]:
    """Run the two-stage pipeline over records and return all 4 metric dicts.

    Parameters
    ----------
    gas_model : trained GasT5Model
    apc_model : trained APCPredictor (BERT category classifier)
    records   : list of record dicts (from ``load_split_records``)
    normalize : apply Levenshtein aspect normalization to GAS output

    Returns
    -------
    Dict with keys ``aspect_term``, ``sentiment``, ``category``, ``joint``,
    each mapping to a dict with ``precision``, ``recall``, ``f1``, ``tp``,
    ``fp``, ``fn``.
    """
    # ── Stage 1: GAS → (aspect, sentiment) per sentence ──────────────────────
    pred_sent_pairs_list, _ = gas_model.predict_for_records(records, normalize=normalize)

    # ── Stage 2: BERT APC → category for each predicted aspect ───────────────
    pred_cat_pairs_list:  List[List[Tuple[str, str]]]          = []
    pred_triples_list:    List[List[Tuple[str, str, str]]]     = []

    for record, sent_pairs in zip(records, pred_sent_pairs_list):
        sentence  = str(record["input_text"])
        cat_pairs: List[Tuple[str, str]] = []
        triples:   List[Tuple[str, str, str]] = []

        for aspect, sentiment in sent_pairs:
            result   = apc_model.predict_one(sentence, aspect)
            category = result["category"]
            cat_pairs.append((aspect, category))
            triples.append((aspect, category, sentiment))

        pred_cat_pairs_list.append(cat_pairs)
        pred_triples_list.append(triples)

    # ── Gold labels ───────────────────────────────────────────────────────────
    gold_sent_pairs_list = [list(r["aspects_sentiments"]) for r in records]
    gold_aspects_list    = [[a for a, _ in pairs] for pairs in gold_sent_pairs_list]
    pred_aspects_list    = [[a for a, _ in pairs] for pairs in pred_sent_pairs_list]

    # Gold (aspect, category) pairs and (aspect, category, sentiment) triples
    gold_cat_pairs_list: List[List[Tuple[str, str]]] = [
        [(a, c) for a, c, s in r["gold_triples"]] for r in records
    ]
    gold_triples_list: List[List[Tuple[str, str, str]]] = [
        list(r["gold_triples"]) for r in records
    ]

    # ── Compute all 4 metrics ─────────────────────────────────────────────────
    return evaluate_all(
        pred_aspects=pred_aspects_list,
        gold_aspects=gold_aspects_list,
        pred_sent_pairs=pred_sent_pairs_list,
        gold_sent_pairs=gold_sent_pairs_list,
        pred_cat_pairs=pred_cat_pairs_list,
        gold_cat_pairs=gold_cat_pairs_list,
        pred_triples=pred_triples_list,
        gold_triples=gold_triples_list,
    )


# ─── Results serialisation ────────────────────────────────────────────────────

def save_results(
    results: Dict[str, Dict[str, float]],
    output_path: Path,
    meta: Optional[Dict] = None,
) -> None:
    """Save evaluation results to a JSON file."""
    payload = {"metrics": results}
    if meta:
        payload["meta"] = meta
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nResults saved → {output_path}")


def save_results_csv(
    results: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    """Save evaluation results to a CSV file (one row per metric label)."""
    rows = []
    for label, m in results.items():
        rows.append(
            {
                "label":     label,
                "precision": round(m["precision"], 6),
                "recall":    round(m["recall"],    6),
                "f1":        round(m["f1"],        6),
                "tp":        int(m["tp"]),
                "fp":        int(m["fp"]),
                "fn":        int(m["fn"]),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["label", "precision", "recall", "f1", "tp", "fp", "fn"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved      → {output_path}")


# ─── Pretty-print ─────────────────────────────────────────────────────────────

def print_results(
    results: Dict[str, Dict[str, float]],
    split: str,
    gas_checkpoint: str,
    apc_checkpoint: str,
) -> None:
    W = 72
    print(f"\n{'═' * W}")
    print("3-LABEL ABSA EVALUATION RESULTS")
    print(f"  Split          : {split}")
    print(f"  GAS checkpoint : {gas_checkpoint}")
    print(f"  APC checkpoint : {apc_checkpoint}")
    print(f"{'═' * W}")
    print(format_report(results))
    print()
    print("Metric definitions:")
    print("  Aspect Term  — exact match on extracted aspect strings")
    print("  Sentiment    — (aspect_term, sentiment) pair must both be correct")
    print("  Category     — (aspect_term, category) pair must both be correct")
    print("  Joint        — all 3 labels (aspect, category, sentiment) must match")
    print(f"{'═' * W}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "3-label ABSA evaluation: GAS (aspect+sentiment) + BERT APC (category). "
            "Reports P/R/F1 for aspect term, sentiment, category, and joint (all 3 correct)."
        )
    )
    parser.add_argument(
        "--gas-checkpoint",
        required=True,
        help="Path to trained GAS T5 checkpoint directory (best/ subdirectory)",
    )
    parser.add_argument(
        "--apc-checkpoint",
        required=True,
        help=(
            "Path to BERT APC checkpoint directory "
            "(e.g. runs_joint/lcf_bip_resize/).  "
            "Must contain best_model.pt + meta.json."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(ROOT / "dataset"),
        help="Directory containing train.apc / dev.apc / test.apc",
    )
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test"],
        default="test",
        help="Dataset split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--bert-name",
        type=str,
        default="bert-base-uncased",
        help="BERT variant used during APC training",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        default=False,
        help="Disable Levenshtein n-gram normalization for GAS aspect terms",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results JSON + CSV (default: same as gas-checkpoint)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device         : {device}")

    # ── Load GAS model ────────────────────────────────────────────────────────
    gas_ckpt = Path(args.gas_checkpoint)
    if not gas_ckpt.is_dir():
        print(f"[error] GAS checkpoint directory not found: {gas_ckpt}", file=sys.stderr)
        print(
            "  Train the GAS model first:\n"
            "    python gas/train_gas.py --output-dir checkpoints/gas_t5 ...",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nLoading GAS model from: {gas_ckpt}")
    gas_model = GasT5Model.from_pretrained(str(gas_ckpt), device=device)

    # ── Load BERT APC model ───────────────────────────────────────────────────
    apc_ckpt = Path(args.apc_checkpoint)
    if not apc_ckpt.is_dir():
        print(
            f"[error] APC checkpoint directory not found: {apc_ckpt}", file=sys.stderr
        )
        print(
            "  Train the APC model first:\n"
            "    python experiments/run_joint_experiments.py",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading APC model from: {apc_ckpt}")
    apc_model = APCPredictor.load(
        checkpoint_dir=str(apc_ckpt),
        bert_name=args.bert_name,
        device=device,
    )

    # ── Load evaluation data ──────────────────────────────────────────────────
    print(f"\nLoading {args.split} data from: {args.data_dir}")
    main_records, _ = load_split_records(args.split, args.data_dir)
    print(f"  {len(main_records)} sentences to evaluate")

    # ── Run evaluation ────────────────────────────────────────────────────────
    print(f"\nRunning 3-label evaluation on {args.split} split …")
    normalize = not args.no_normalize
    results   = run_joint_evaluation(
        gas_model=gas_model,
        apc_model=apc_model,
        records=main_records,
        normalize=normalize,
    )

    # ── Print results ─────────────────────────────────────────────────────────
    print_results(
        results,
        split=args.split,
        gas_checkpoint=str(gas_ckpt),
        apc_checkpoint=str(apc_ckpt),
    )

    # ── Save results ──────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir) if args.output_dir else gas_ckpt.parent
    meta = {
        "split":          args.split,
        "gas_checkpoint": str(gas_ckpt),
        "apc_checkpoint": str(apc_ckpt),
        "bert_name":      args.bert_name,
        "n_sentences":    len(main_records),
        "normalize":      normalize,
    }
    save_results(results,     out_dir / f"eval_joint_{args.split}.json", meta)
    save_results_csv(results, out_dir / f"eval_joint_{args.split}.csv")


if __name__ == "__main__":
    main()
