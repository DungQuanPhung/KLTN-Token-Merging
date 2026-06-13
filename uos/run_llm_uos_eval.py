#!/usr/bin/env python3
"""UOS sentence splitting via Ollama LLM.

Dataset format (.txt) — 4 lines per sample, blank line between samples:
    sentence
    aspect_terms
    categories
    sentiments

Usage:
    python run_llm_uos_eval.py --data_path data/reviews.txt
    python run_llm_uos_eval.py --data_path data/reviews.json --text_column review
    python run_llm_uos_eval.py --data_path data/reviews.txt --max_rows 10 --output_dir out/
    python run_llm_uos_eval.py --data_path data/reviews.txt --resume

Output (in --output_dir):
    results.jsonl  — one row per sentence
    metrics.json   — summary stats
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uos.io import iter_records
from uos.pipeline import run_corpus
from uos.records import export_row, summarize
from uos.segmenter import (
    PROMPT_VERSION,
    check_ollama,
    create_llm_segmenter,
    resolve_model_name,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "UOS splitting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--data_path", required=True, help=".txt (4-line format) or .json")
    p.add_argument("--text_column", default="", help="Column name for JSON input")
    p.add_argument("--max_rows", type=int, default=0, help="Limit rows processed (0 = all)")
    p.add_argument("--llm_model", default="Qwen3-8B-Instruct")
    p.add_argument("--ollama_host", default="http://localhost:11434")
    p.add_argument("--output_dir", default="output")
    p.add_argument("--resume", action="store_true", help="Skip already-processed rows")
    p.add_argument("--quiet", action="store_true", help="No progress output")
    p.add_argument("--save_raw_response", action="store_true", help="Keep raw LLM text in output")
    return p.parse_args()


def _row_key(row: dict) -> str:
    return f"{row.get('entity_id', '')}|{row.get('sentence_idx', '')}"


def _load_existing(results_path: Path) -> tuple[list[dict], set[str]]:
    if not results_path.exists():
        return [], set()
    rows: list[dict] = []
    keys: set[str] = set()
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows.append(row)
            keys.add(_row_key(row))
    return rows, keys


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise SystemExit(f"File not found: {data_path}")

    model = resolve_model_name(args.llm_model)
    check_ollama(args.ollama_host, model)
    segmenter = create_llm_segmenter(model, host=args.ollama_host)

    print(f"UOS | model={model} | prompt={PROMPT_VERSION}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"

    existing_rows, completed = _load_existing(results_path) if args.resume else ([], set())
    if completed:
        print(f"Resume: skipping {len(completed)} already-processed rows")

    all_records = list(iter_records(
        data_path, text_column=args.text_column, max_rows=args.max_rows
    ))
    pending = [r for r in all_records if _row_key(r) not in completed]
    print(f"Sentences: {len(all_records)} total, {len(pending)} to process")

    pbar = None
    progress = None
    if not args.quiet:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(pending), unit="sentence")

            def progress(done, total, latency_s, sentence):
                pbar.update(1)
                pbar.set_postfix_str(f"{latency_s:.1f}s | {sentence[:40]}", refresh=True)
        except ImportError:
            pass

    new_rows: list[dict] = []

    def write_row(row: dict) -> None:
        payload = export_row(row)
        if not args.save_raw_response:
            payload["raw_response"] = ""
        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        new_rows.append(row)

    run_corpus(segmenter, pending, on_progress=progress, on_record=write_row)

    if pbar is not None:
        pbar.close()

    all_rows = existing_rows + new_rows
    summary = summarize(all_rows)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {"model": model, "prompt_version": PROMPT_VERSION, "summary": summary},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"\nResults : {results_path}  ({len(all_rows)} rows)")
    print(f"Metrics : {metrics_path}")


if __name__ == "__main__":
    main()
