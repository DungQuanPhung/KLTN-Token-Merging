# -*- coding: utf-8 -*-
"""GAS T5 inference — 3-label ABSA in a single decoder pass.

The model generates (aspect_term, CATEGORY, sentiment) for every aspect
in a sentence.

Modes
-----
Single sentence (--text):
    python gas/infer.py --checkpoint checkpoints/gas_t5/best \\
        --text "The food was great but service was slow."

Batch from file (--input-file):
    python gas/infer.py --checkpoint checkpoints/gas_t5/best \\
        --input-file sentences.txt --output results.json

Interactive REPL (no --text / --input-file):
    python gas/infer.py --checkpoint checkpoints/gas_t5/best

Output format
-------------
Each sentence produces a list of triples:
    [{"aspect_term": "food", "category": "FOOD", "sentiment": "positive"}, ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from gas.model import GasT5Model


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _triples_to_dicts(
    triples: List[Tuple[str, str, str]],
) -> List[Dict[str, str]]:
    return [
        {"aspect_term": a, "category": c, "sentiment": s}
        for a, c, s in triples
    ]


def _print_result(sentence: str, triples: List[Tuple[str, str, str]]) -> None:
    print(f"\nSentence : {sentence}")
    if not triples:
        print("  (no aspects detected)")
        return
    for i, (aspect, category, sentiment) in enumerate(triples, 1):
        print(f"  [{i}] aspect={aspect!r:20s}  category={category:<16}  sentiment={sentiment}")


def _load_sentences(path: str) -> List[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


# ─── Output writers ───────────────────────────────────────────────────────────

def _save_json(
    sentences: List[str],
    predictions: List[List[Tuple[str, str, str]]],
    path: Path,
    checkpoint: str,
) -> None:
    records = [
        {
            "sentence":   sent,
            "aspects":    _triples_to_dicts(triples),
            "raw_output": "; ".join(
                f"({a}, {c}, {s})" for a, c, s in triples
            ) if triples else "none",
        }
        for sent, triples in zip(sentences, predictions)
    ]
    payload = {
        "checkpoint": checkpoint,
        "n_sentences": len(sentences),
        "results": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nJSON saved -> {path}")


def _save_csv(
    sentences: List[str],
    predictions: List[List[Tuple[str, str, str]]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for sent, triples in zip(sentences, predictions):
        if not triples:
            rows.append(
                {"sentence": sent, "aspect_term": "", "category": "", "sentiment": ""}
            )
        else:
            for aspect, category, sentiment in triples:
                rows.append(
                    {
                        "sentence":    sent,
                        "aspect_term": aspect,
                        "category":    category,
                        "sentiment":   sentiment,
                    }
                )
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sentence", "aspect_term", "category", "sentiment"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV  saved -> {path}")


def _save_results(
    sentences: List[str],
    predictions: List[List[Tuple[str, str, str]]],
    output_path: str,
    checkpoint: str,
) -> None:
    p = Path(output_path)
    if p.suffix.lower() == ".csv":
        _save_csv(sentences, predictions, p)
    else:
        _save_json(sentences, predictions, p, checkpoint)


# ─── Interactive REPL ─────────────────────────────────────────────────────────

def _interactive_loop(model: GasT5Model, normalize: bool) -> None:
    print("\nInteractive mode — type a sentence and press Enter (Ctrl-C to quit).")
    print("Type 'quit' or 'exit' to stop.\n")
    while True:
        try:
            sentence = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if not sentence:
            continue
        if sentence.lower() in {"quit", "exit"}:
            break
        triples = model.predict_one(sentence, normalize=normalize)
        _print_result(sentence, triples)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "GAS T5 inference: extract (aspect_term, CATEGORY, sentiment) triples "
            "from free text in a single decoder pass."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        metavar="DIR",
        help="Path to a trained GAS T5 checkpoint directory (e.g. checkpoints/gas_t5/best).",
    )
    parser.add_argument(
        "--text",
        metavar="SENTENCE",
        default=None,
        help="Single sentence to analyse (wrap in quotes if it contains spaces).",
    )
    parser.add_argument(
        "--input-file",
        metavar="FILE",
        default=None,
        help="Plain-text file with one sentence per line.",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help=(
            "Where to save results. Extension determines format: "
            ".csv → CSV, anything else → JSON. "
            "Defaults to <checkpoint_dir>/predictions.json when --input-file is used."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Inference batch size for --input-file mode (default: 16).",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        default=False,
        help="Disable Levenshtein n-gram normalization on generated aspect terms.",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help="Beam search width (default: 4).",
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=128,
        help="Maximum tokenizer input length (default: 128).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_dir():
        print(
            f"[error] Checkpoint not found: {ckpt}\n"
            "  Train the GAS model first:\n"
            "    python gas/train_gas.py --output-dir checkpoints/gas_t5",
            file=sys.stderr,
        )
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Checkpoint : {ckpt}")

    from gas.model import GenerationConfig
    gen_cfg = GenerationConfig(num_beams=args.num_beams)
    model   = GasT5Model.from_pretrained(str(ckpt), device=device, generation=gen_cfg)

    normalize = not args.no_normalize

    # ── Single sentence ───────────────────────────────────────────────────────
    if args.text is not None:
        triples = model.predict_one(
            args.text,
            normalize=normalize,
            max_input_length=args.max_input_length,
        )
        _print_result(args.text, triples)

        if args.output:
            _save_results(
                [args.text], [triples], args.output, str(ckpt)
            )
        return

    # ── Batch from file ───────────────────────────────────────────────────────
    if args.input_file is not None:
        input_path = Path(args.input_file)
        if not input_path.is_file():
            print(f"[error] Input file not found: {input_path}", file=sys.stderr)
            sys.exit(1)

        sentences = _load_sentences(str(input_path))
        print(f"Input file : {input_path}  ({len(sentences)} sentences)")

        predictions = model.predict_batch(
            sentences,
            batch_size=args.batch_size,
            normalize=normalize,
            max_input_length=args.max_input_length,
        )

        for sent, triples in zip(sentences, predictions):
            _print_result(sent, triples)

        out_path = args.output or str(ckpt.parent / "predictions.json")
        _save_results(sentences, predictions, out_path, str(ckpt))
        return

    # ── Interactive REPL ──────────────────────────────────────────────────────
    _interactive_loop(model, normalize)


if __name__ == "__main__":
    main()
