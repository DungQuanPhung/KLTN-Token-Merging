# -*- coding: utf-8 -*-
"""Infer sentiment and category for one sentence and one aspect term."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.pipeline_inference import APCPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict sentiment and category for one sentence + one aspect term."
    )
    parser.add_argument(
        "--apc-checkpoint-dir",
        required=True,
        help="Directory containing best_model.pt + meta.json (output of experiments/run_joint_experiments.py)",
    )
    parser.add_argument(
        "--bert-name",
        default="bert-base-uncased",
        help="HuggingFace BERT variant used for APC model loading.",
    )
    parser.add_argument(
        "--sentence",
        required=True,
        help="Full input sentence containing the aspect term.",
    )
    parser.add_argument(
        "--aspect",
        required=True,
        help="Aspect term to classify within the sentence.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=128,
        help="Maximum sequence length for BERT tokenization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    apc = APCPredictor.load(
        checkpoint_dir=args.apc_checkpoint_dir,
        bert_name=args.bert_name,
        max_seq_len=args.max_seq_len,
    )

    result = apc.predict_one(args.sentence, args.aspect)

    print("Input sentence:", args.sentence)
    print("Aspect term:", args.aspect)
    print("Sentiment:", result["sentiment"])
    print("Category:", result["category"])


if __name__ == "__main__":
    main()
