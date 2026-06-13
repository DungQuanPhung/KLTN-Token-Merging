"""UOS segmentation pipeline."""
from __future__ import annotations

import sys
import time
from typing import Callable, Iterable, List, Optional

from uos.records import build_record

ProgressFn = Callable[[int, Optional[int], float, str], None]
RecordFn = Callable[[dict], None]


def _default_progress(done: int, total: Optional[int], latency_s: float, sentence: str) -> None:
    preview = sentence[:60] + ("..." if len(sentence) > 60 else "")
    total_str = str(total) if total is not None else "?"
    print(f"  [{done}/{total_str}] {latency_s:.1f}s | {preview}", flush=True)


def segment_one(segmenter, sentence: str) -> tuple[List[str], str]:
    if hasattr(segmenter, "segment_with_raw"):
        return segmenter.segment_with_raw(sentence)
    return list(segmenter.segment(sentence)), ""


def run_corpus(
    segmenter,
    records: Iterable[dict],
    *,
    on_progress: Optional[ProgressFn] = _default_progress,
    on_record: Optional[RecordFn] = None,
    total: Optional[int] = None,
) -> List[dict]:
    rows: List[dict] = []
    for i, meta in enumerate(records, start=1):
        t0 = time.perf_counter()
        try:
            units, raw = segment_one(segmenter, meta["sentence"])
        except Exception as exc:
            units = [meta["sentence"]]
            raw = f"ERROR: {exc}"
            print(f"  [ERROR] row {i}: {exc}", file=sys.stderr, flush=True)
        latency = time.perf_counter() - t0
        row = build_record(meta, units, latency, raw_response=raw)
        rows.append(row)
        if on_record:
            on_record(row)
        if on_progress:
            on_progress(i, total, latency, meta["sentence"])
    return rows
