# -*- coding: utf-8 -*-
"""Evaluation metrics for 3-label ABSA: aspect term / category / sentiment / joint.

All metrics use **micro-averaged** precision, recall, and F1 computed over
individual predictions — consistent with the standard ABSA evaluation protocol
(GAS, PARAPHRASE, MvP, etc.).

The single-stage GAS model generates all three labels simultaneously, so all
four metrics are derived from the same set of predictions.

Metric definitions
------------------
aspect_term : TP when predicted aspect string exactly matches a gold term.
category    : TP when (aspect_term, category) pair exactly matches gold.
sentiment   : TP when (aspect_term, sentiment) pair exactly matches gold.
joint       : TP when (aspect_term, category, sentiment) triple all match gold.
              Getting 2 out of 3 correct counts as 0 TP (pure FP + FN).

All comparisons use .strip() normalisation; no case folding.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


# ─── Core PRF ─────────────────────────────────────────────────────────────────

def compute_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Return {precision, recall, f1, tp, fp, fn} from raw counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return {
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "tp":        float(tp),
        "fp":        float(fp),
        "fn":        float(fn),
    }


def _micro_prf(all_pred_sets, all_gold_sets) -> Dict[str, float]:
    total_tp = total_fp = total_fn = 0
    for pred_set, gold_set in zip(all_pred_sets, all_gold_sets):
        total_tp += len(pred_set & gold_set)
        total_fp += len(pred_set - gold_set)
        total_fn += len(gold_set - pred_set)
    return compute_prf(total_tp, total_fp, total_fn)


# ─── Derived label sets ────────────────────────────────────────────────────────
# All four metrics are derived from the same List[List[Tuple[str,str,str]]].

def _aspect_sets(triples_per_sent):
    return [
        {t[0].strip() for t in triples}
        for triples in triples_per_sent
    ]

def _category_pair_sets(triples_per_sent):
    return [
        {(t[0].strip(), t[1].strip()) for t in triples}
        for triples in triples_per_sent
    ]

def _sentiment_pair_sets(triples_per_sent):
    return [
        {(t[0].strip(), t[2].strip()) for t in triples}
        for triples in triples_per_sent
    ]

def _triple_sets(triples_per_sent):
    return [
        {(t[0].strip(), t[1].strip(), t[2].strip()) for t in triples}
        for triples in triples_per_sent
    ]


# ─── Per-label evaluators ─────────────────────────────────────────────────────

def evaluate_aspect_term(
    pred_triples: List[List[Tuple[str, str, str]]],
    gold_triples: List[List[Tuple[str, str, str]]],
) -> Dict[str, float]:
    """Micro P/R/F1 on aspect term extraction.

    Only the first field (aspect_term) of each triple is compared;
    category and sentiment are ignored.
    """
    return _micro_prf(_aspect_sets(pred_triples), _aspect_sets(gold_triples))


def evaluate_category(
    pred_triples: List[List[Tuple[str, str, str]]],
    gold_triples: List[List[Tuple[str, str, str]]],
) -> Dict[str, float]:
    """Micro P/R/F1 on (aspect_term, category) pairs.

    A prediction is TP only when both the aspect term AND the category
    match a gold pair simultaneously.
    """
    return _micro_prf(
        _category_pair_sets(pred_triples),
        _category_pair_sets(gold_triples),
    )


def evaluate_sentiment(
    pred_triples: List[List[Tuple[str, str, str]]],
    gold_triples: List[List[Tuple[str, str, str]]],
) -> Dict[str, float]:
    """Micro P/R/F1 on (aspect_term, sentiment) pairs.

    A prediction is TP only when both the aspect term AND the sentiment
    match a gold pair simultaneously.
    """
    return _micro_prf(
        _sentiment_pair_sets(pred_triples),
        _sentiment_pair_sets(gold_triples),
    )


def evaluate_joint(
    pred_triples: List[List[Tuple[str, str, str]]],
    gold_triples: List[List[Tuple[str, str, str]]],
) -> Dict[str, float]:
    """Micro P/R/F1 on (aspect_term, category, sentiment) triples.

    A prediction is TP **only** when all three labels match a gold triple
    simultaneously.  Partial matches (2 out of 3) contribute 0 TP.
    """
    return _micro_prf(_triple_sets(pred_triples), _triple_sets(gold_triples))


# ─── Composite evaluation ─────────────────────────────────────────────────────

def evaluate_all(
    pred_triples: List[List[Tuple[str, str, str]]],
    gold_triples: List[List[Tuple[str, str, str]]],
) -> Dict[str, Dict[str, float]]:
    """Run all four evaluations from a single set of (aspect, category, sentiment) triples.

    Parameters
    ----------
    pred_triples : predicted triples, one list per sentence
    gold_triples : gold triples, one list per sentence

    Returns
    -------
    Dict with keys ``aspect_term``, ``category``, ``sentiment``, ``joint``.
    """
    return {
        "aspect_term": evaluate_aspect_term(pred_triples, gold_triples),
        "category":    evaluate_category(   pred_triples, gold_triples),
        "sentiment":   evaluate_sentiment(  pred_triples, gold_triples),
        "joint":       evaluate_joint(      pred_triples, gold_triples),
    }


# ─── Formatting ───────────────────────────────────────────────────────────────

def format_metric(label: str, m: Dict[str, float]) -> str:
    """Single-line metric summary."""
    return (
        f"{label:<16}"
        f"P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}"
        f"  (TP={int(m['tp'])} FP={int(m['fp'])} FN={int(m['fn'])})"
    )


def format_report(results: Dict[str, Dict[str, float]]) -> str:
    """Format all 4 metrics as a multi-line report string."""
    order  = ["aspect_term", "category", "sentiment", "joint"]
    labels = {
        "aspect_term": "Aspect Term",
        "category":    "Category",
        "sentiment":   "Sentiment",
        "joint":       "Joint (all 3)",
    }
    lines = ["-" * 72]
    for key in order:
        if key in results:
            lines.append(format_metric(labels[key], results[key]))
    lines.append("-" * 72)
    return "\n".join(lines)
