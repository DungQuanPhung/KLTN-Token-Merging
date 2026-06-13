"""Build output records and corpus summaries."""
from __future__ import annotations

import statistics
from typing import Sequence

from uos.metrics import compute_sentence_unit_metrics
from uos.parsing import validate_units
from uos.validation import detect_warnings, unsupported_token_rate

_ABSA_FIELDS = ("raw_sentence", "aspect_terms", "categories", "sentiments")


def build_record(
    meta: dict,
    units: Sequence[str],
    latency_s: float,
    raw_response: str = "",
    *,
    ok: bool | None = None,
) -> dict:
    units = validate_units(meta["sentence"], list(units))
    token_recall, char_coverage, was_split = compute_sentence_unit_metrics(
        meta["sentence"], units
    )
    record: dict = {
        "entity_id": meta["entity_id"],
        "review_id": meta["review_id"],
        "sentence_idx": meta["sentence_idx"],
        "sentence": meta["sentence"],
        "units": units,
        "warnings": detect_warnings(meta["sentence"], units),
        "raw_response": raw_response,
        "metrics": {
            "n_units": len(units),
            "was_split": was_split,
            "token_recall": round(token_recall, 4),
            "char_coverage": round(char_coverage, 4),
            "information_loss_token": round(1 - token_recall, 4),
            "information_loss_char": round(1 - char_coverage, 4),
            "unsupported_token_rate": round(
                unsupported_token_rate(meta["sentence"], units), 4
            ),
            "latency_s": round(latency_s, 3),
        },
    }
    for field in _ABSA_FIELDS:
        if field in meta:
            record[field] = meta[field]
    if ok is not None:
        record["metrics"]["ok"] = ok
    return record


def export_row(record: dict) -> dict:
    metrics = {k: v for k, v in record["metrics"].items() if k != "ok"}
    row: dict = {
        "entity_id": record["entity_id"],
        "review_id": record["review_id"],
        "sentence_idx": record["sentence_idx"],
        "sentence": record["sentence"],
        "units": record["units"],
        "warnings": record["warnings"],
        "raw_response": record.get("raw_response") or "",
        "metrics": metrics,
    }
    for field in _ABSA_FIELDS:
        if field in record:
            row[field] = record[field]
    return row


def summarize(rows: Sequence[dict]) -> dict:
    if not rows:
        return {}
    n = len(rows)
    return {
        "n_sentences": n,
        "split_rate": round(sum(1 for r in rows if r["metrics"]["was_split"]) / n, 4),
        "warning_rate": round(sum(1 for r in rows if r.get("warnings")) / n, 4),
        "units_per_sentence_mean": round(
            sum(r["metrics"]["n_units"] for r in rows) / n, 4
        ),
        "token_recall_mean": round(
            statistics.mean(r["metrics"]["token_recall"] for r in rows), 4
        ),
        "char_coverage_mean": round(
            statistics.mean(r["metrics"]["char_coverage"] for r in rows), 4
        ),
        "latency_s_mean": round(
            statistics.mean(r["metrics"]["latency_s"] for r in rows), 4
        ),
    }
