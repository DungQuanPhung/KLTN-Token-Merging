# -*- coding: utf-8 -*-
"""Evaluation metrics for 3-label ABSA (aspect term / category / sentiment).

All metrics use **micro-averaged** precision, recall, and F1 computed over
individual predictions (not per-sentence averages) — consistent with the
standard ABSA evaluation protocol used in GAS, PARAPHRASE, etc.

Metric definitions
------------------
aspect_term  : TP when predicted aspect term exactly matches a gold term.
sentiment    : TP when (aspect_term, sentiment) pair exactly matches gold.
category     : TP when (aspect_term, category) pair exactly matches gold.
joint        : TP when (aspect_term, category, sentiment) triple all match gold.
               A prediction that gets 2 out of 3 labels correct contributes 0 TP.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Set, Tuple


# ─── Core PRF computation ─────────────────────────────────────────────────────

def compute_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Return precision / recall / F1 from raw counts."""
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


def _micro_counts(
    all_pred: Iterable[Iterable],
    all_gold: Iterable[Iterable],
) -> Tuple[int, int, int]:
    """Sum TP / FP / FN across all sentences using set intersection."""
    total_tp = total_fp = total_fn = 0
    for preds, golds in zip(all_pred, all_gold):
        pred_set = set(preds)
        gold_set = set(golds)
        total_tp += len(pred_set & gold_set)
        total_fp += len(pred_set - gold_set)
        total_fn += len(gold_set - pred_set)
    return total_tp, total_fp, total_fn


# ─── Per-label evaluators ─────────────────────────────────────────────────────

def evaluate_aspect_term(
    all_pred_aspects: List[List[str]],
    all_gold_aspects: List[List[str]],
) -> Dict[str, float]:
    """Micro P/R/F1 on aspect term extraction (exact string match, stripped).

    Args:
        all_pred_aspects : one list of predicted aspect strings per sentence.
        all_gold_aspects : one list of gold aspect strings per sentence.
    """
    def _normalize(aspects):
        return (a.strip() for a in aspects if a and a.strip())

    tp, fp, fn = _micro_counts(
        (_normalize(p) for p in all_pred_aspects),
        (_normalize(g) for g in all_gold_aspects),
    )
    return compute_prf(tp, fp, fn)


def evaluate_sentiment_pairs(
    all_pred: List[List[Tuple[str, str]]],
    all_gold: List[List[Tuple[str, str]]],
) -> Dict[str, float]:
    """Micro P/R/F1 on (aspect_term, sentiment) pairs.

    A prediction is TP only when both the aspect term AND the sentiment match
    a gold pair exactly.

    Args:
        all_pred : one list of (aspect, sentiment) tuples per sentence.
        all_gold : one list of (aspect, sentiment) gold tuples per sentence.
    """
    def _norm_pairs(pairs):
        return ((a.strip(), s.strip()) for a, s in pairs)

    tp, fp, fn = _micro_counts(
        (_norm_pairs(p) for p in all_pred),
        (_norm_pairs(g) for g in all_gold),
    )
    return compute_prf(tp, fp, fn)


def evaluate_category_pairs(
    all_pred: List[List[Tuple[str, str]]],
    all_gold: List[List[Tuple[str, str]]],
) -> Dict[str, float]:
    """Micro P/R/F1 on (aspect_term, category) pairs.

    A prediction is TP only when both the aspect term AND the category match.
    Category is predicted by the BERT APC model; this metric thus captures
    *both* the ATE quality and the category classification quality together.

    Args:
        all_pred : one list of (aspect, category) tuples per sentence.
        all_gold : one list of (aspect, category) gold tuples per sentence.
    """
    def _norm_pairs(pairs):
        return ((a.strip(), c.strip()) for a, c in pairs)

    tp, fp, fn = _micro_counts(
        (_norm_pairs(p) for p in all_pred),
        (_norm_pairs(g) for g in all_gold),
    )
    return compute_prf(tp, fp, fn)


def evaluate_joint_triples(
    all_pred: List[List[Tuple[str, str, str]]],
    all_gold: List[List[Tuple[str, str, str]]],
) -> Dict[str, float]:
    """Micro P/R/F1 on (aspect_term, category, sentiment) triples.

    A prediction is TP **only** when all three labels match a gold triple
    simultaneously.  Getting 2 out of 3 correct still counts as FP + FN.

    Args:
        all_pred : one list of (aspect, category, sentiment) tuples per sentence.
        all_gold : one list of (aspect, category, sentiment) gold tuples per sentence.
    """
    def _norm_triples(triples):
        return ((a.strip(), c.strip(), s.strip()) for a, c, s in triples)

    tp, fp, fn = _micro_counts(
        (_norm_triples(p) for p in all_pred),
        (_norm_triples(g) for g in all_gold),
    )
    return compute_prf(tp, fp, fn)


# ─── Composite evaluation ─────────────────────────────────────────────────────

def evaluate_all(
    pred_aspects:    List[List[str]],
    gold_aspects:    List[List[str]],
    pred_sent_pairs: List[List[Tuple[str, str]]],
    gold_sent_pairs: List[List[Tuple[str, str]]],
    pred_cat_pairs:  List[List[Tuple[str, str]]],
    gold_cat_pairs:  List[List[Tuple[str, str]]],
    pred_triples:    List[List[Tuple[str, str, str]]],
    gold_triples:    List[List[Tuple[str, str, str]]],
) -> Dict[str, Dict[str, float]]:
    """Run all four evaluations and return a structured report dict."""
    return {
        "aspect_term": evaluate_aspect_term(pred_aspects, gold_aspects),
        "sentiment":   evaluate_sentiment_pairs(pred_sent_pairs, gold_sent_pairs),
        "category":    evaluate_category_pairs(pred_cat_pairs,  gold_cat_pairs),
        "joint":       evaluate_joint_triples(pred_triples,     gold_triples),
    }


# ─── Formatting ───────────────────────────────────────────────────────────────

def format_metric(label: str, m: Dict[str, float]) -> str:
    """Single-line metric summary."""
    return (
        f"{label:<14} "
        f"P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}"
        f"  (TP={int(m['tp'])} FP={int(m['fp'])} FN={int(m['fn'])})"
    )


def format_report(results: Dict[str, Dict[str, float]]) -> str:
    """Format all 4 metrics as a multi-line string."""
    lines = ["─" * 72]
    order = ["aspect_term", "sentiment", "category", "joint"]
    labels = {
        "aspect_term": "Aspect Term",
        "sentiment":   "Sentiment",
        "category":    "Category",
        "joint":       "Joint (all 3)",
    }
    for key in order:
        if key in results:
            lines.append(format_metric(labels[key], results[key]))
    lines.append("─" * 72)
    return "\n".join(lines)
