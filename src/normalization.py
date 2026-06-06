# -*- coding: utf-8 -*-
"""Post-processing normalization for GAS-style ATE outputs."""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Set

import Levenshtein

_ASPECT_PATTERN = re.compile(r"\(([^)]*)\)")


def build_ngram_vocabulary(sentence: str) -> Set[str]:
    """Build candidate vocabulary V from all word n-grams in the sentence."""
    words = sentence.split()
    candidates: Set[str] = set()
    n = len(words)
    for start in range(n):
        for end in range(start + 1, n + 1):
            span = " ".join(words[start:end])
            if span:
                candidates.add(span)
    return candidates


def normalize_aspect(term: str, vocabulary: Set[str]) -> str:
    """Map an aspect to the closest n-gram in V (identity if already present)."""
    cleaned = term.strip()
    if not cleaned or not vocabulary:
        return cleaned
    if cleaned in vocabulary:
        return cleaned

    best = cleaned
    best_dist = None
    for candidate in vocabulary:
        dist = Levenshtein.distance(cleaned, candidate)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = candidate
    return best


def normalize_aspects(aspects: Sequence[str], sentence: str) -> List[str]:
    """Normalize each predicted aspect against sentence n-grams."""
    vocab = build_ngram_vocabulary(sentence)
    return [normalize_aspect(a, vocab) for a in aspects if a and a.strip()]


def decode_target_text(text: str) -> List[str]:
    """Decode GAS extraction-style output into aspect strings.

    Examples:
        "(pizza); (service)" -> ["pizza", "service"]
        "(pizza)"            -> ["pizza"]
        "none"               -> []
        "pizza" (no parens)  -> []
    """
    stripped = text.strip()
    if not stripped or stripped.lower() == "none":
        return []

    matches = _ASPECT_PATTERN.findall(stripped)
    if not matches:
        return []

    aspects: List[str] = []
    for match in matches:
        aspect = match.strip()
        if aspect:
            aspects.append(aspect)
    return aspects


def decode_and_normalize(text: str, sentence: str) -> List[str]:
    """Decode model output and apply Levenshtein n-gram normalization."""
    raw = decode_target_text(text)
    return normalize_aspects(raw, sentence)


def format_aspects_for_display(aspects: Iterable[str]) -> str:
    cleaned = [a.strip() for a in aspects if a and a.strip()]
    if not cleaned:
        return "none"
    return "; ".join(f"({a})" for a in cleaned)
