# -*- coding: utf-8 -*-
"""CLI entry point for training the GAS (Generative Aspect Sentiment) T5 model.

Usage
-----
    python gas/train_gas.py \\
        --data-dir   dataset \\
        --output-dir checkpoints/gas_t5 \\
        --supplement-dir dataset/supplement \\
        --epochs 20 \\
        --batch-size 16

The trained model is saved to <output-dir>/best/ (best dev checkpoint)
and <output-dir>/last/ (final epoch checkpoint).

To then run the full 3-label evaluation (GAS + BERT category):
    python gas/evaluate_joint.py \\
        --gas-checkpoint   checkpoints/gas_t5/best \\
        --apc-checkpoint   runs_joint/lcf_bip_resize \\
        --data-dir         dataset
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from gas.dataset import (
    DEFAULT_DATA_DIR,
    create_gas_dataloaders,
)
from gas.model import GasT5Model
from gas.trainer import GasTrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train GAS T5 model: generative aspect term + sentiment extraction. "
            "Supplement TSV files are used to improve minority-class recall."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing train.apc / dev.apc / test.apc",
    )
    parser.add_argument(
        "--supplement-dir",
        type=str,
        default=None,
        help=(
            "Directory containing supplement TSV files (negative.tsv, neutral.tsv). "
            "Defaults to <data-dir>/supplement if it exists."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "checkpoints" / "gas_t5"),
        help="Checkpoint output directory",
    )
    parser.add_argument("--model-name",       type=str,   default="t5-base")
    parser.add_argument("--batch-size",        type=int,   default=16)
    parser.add_argument("--learning-rate",     type=float, default=3e-4)
    parser.add_argument("--epochs",            type=int,   default=20)
    parser.add_argument("--max-input-length",  type=int,   default=128)
    parser.add_argument("--max-target-length", type=int,   default=128)
    parser.add_argument("--patience",          type=int,   default=5)
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--num-workers",       type=int,   default=0)
    return parser.parse_args()


def resolve_supplement_paths(args: argparse.Namespace) -> list:
    """Return supplement TSV paths if they exist, else empty list."""
    supp_dir: Path
    if args.supplement_dir:
        supp_dir = Path(args.supplement_dir)
    else:
        supp_dir = Path(args.data_dir) / "supplement"

    if not supp_dir.is_dir():
        print(f"[warn] Supplement directory not found: {supp_dir}  (no supplement)")
        return []

    paths = []
    for fname in ("negative.tsv", "neutral.tsv"):
        p = supp_dir / fname
        if p.is_file():
            paths.append(str(p))
        else:
            print(f"[warn] Supplement file not found (skipped): {p}")
    return paths


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    supplement_paths = resolve_supplement_paths(args)

    print(f"Data dir      : {args.data_dir}")
    print(f"Output dir    : {args.output_dir}")
    print(f"Model         : {args.model_name}")
    print(f"Device        : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Supplement    : {supplement_paths or 'none'}")

    (
        train_loader,
        dev_loader,
        test_loader,
        tokenizer,
        dev_records,
        test_records,
    ) = create_gas_dataloaders(
        data_dir=args.data_dir,
        supplement_paths=supplement_paths or None,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        num_workers=args.num_workers,
    )

    model = GasT5Model(model_name=args.model_name)
    model.tokenizer = tokenizer  # reuse already-loaded tokenizer

    trainer = GasTrainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        patience=args.patience,
        output_dir=args.output_dir,
        dev_records=dev_records,
        test_records=test_records,
    )

    result = trainer.train()

    print("\n=== Training complete ===")
    print(f"Best dev sent-F1  : {result['best_dev_f1']:.4f}")
    print(f"Best checkpoint   : {result['best_checkpoint']}")
    print(f"Wall time (sec)   : {result['wall_time_sec']}")

    tm = result.get("test_metrics", {})
    if tm:
        ate  = tm.get("ate",       {})
        sent = tm.get("sentiment", {})
        print(
            f"Test ATE  P/R/F1  : "
            f"{ate.get('precision', 0):.4f} / "
            f"{ate.get('recall',    0):.4f} / "
            f"{ate.get('f1',        0):.4f}"
        )
        print(
            f"Test Sent P/R/F1  : "
            f"{sent.get('precision', 0):.4f} / "
            f"{sent.get('recall',    0):.4f} / "
            f"{sent.get('f1',        0):.4f}"
        )

    print(
        f"\nNext step — full 3-label evaluation:\n"
        f"  python gas/evaluate_joint.py \\\n"
        f"    --gas-checkpoint  {result['best_checkpoint']} \\\n"
        f"    --apc-checkpoint  runs_joint/lcf_bip_resize \\\n"
        f"    --data-dir        {args.data_dir}"
    )


if __name__ == "__main__":
    main()
