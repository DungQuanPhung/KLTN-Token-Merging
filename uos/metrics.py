"""Per-sentence UOS quality metrics."""
from __future__ import annotations

import re
from typing import Sequence


def _content_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1}


def compute_sentence_unit_metrics(
    original: str,
    unit_texts: Sequence[str],
) -> tuple[float, float, bool]:
    if not unit_texts:
        return 0.0, 0.0, False
    orig = _content_tokens(original)
    covered: set[str] = set()
    for unit in unit_texts:
        covered |= _content_tokens(unit)
    token_recall = len(covered & orig) / len(orig) if orig else 1.0
    char_coverage = min(1.0, sum(len(u) for u in unit_texts) / max(1, len(original)))
    return token_recall, char_coverage, len(unit_texts) > 1
