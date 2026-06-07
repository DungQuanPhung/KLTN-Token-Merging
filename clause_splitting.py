# -*- coding: utf-8 -*-
"""Clause-level preprocessing for APC.

Splits a sentence into clauses and extracts the clause that contains
the aspect term.  Only that clause is passed to the APC model, reducing
noise from unrelated sentence parts.

Clause boundaries recognised (conservative — avoids splitting within
common aspect-term phrases):
    - comma  ,
    - semicolon  ;
    - adversative conjunction: but, yet
    - discourse marker: however
    - concessive/contrastive subordinators: although, though, whereas

Conjunctions like "and", "or", "so", "because", "since", "if" are
intentionally excluded — they appear too often inside aspect terms or
closely tied sub-clauses (e.g. "fish and chips", "clean and tidy").

Usage::

    from clause_splitting import extract_aspect_clause

    clause, new_start, new_end = extract_aspect_clause(
        text="The food was great, but the service was slow",
        aspect_char_start=28,
        aspect_char_end=34,
    )
    # → ("but the service was slow", 8, 14)
"""

from __future__ import annotations

import re
from typing import List, Tuple

# ── Boundary pattern ──────────────────────────────────────────────────────────

_BOUNDARY_RE = re.compile(
    r",\s*"                     # comma
    r"|;\s*"                    # semicolon
    r"|\s+but\s+"              # adversative conjunction
    r"|\s+yet\s+"              # adversative conjunction
    r"|\s+however[,\s]+"       # discourse marker (followed by comma or space)
    r"|\s+although\s+"         # concessive subordinator
    r"|\s+though\s+"           # concessive subordinator
    r"|\s+whereas\s+",         # contrastive subordinator
    re.IGNORECASE,
)

# Minimum number of characters a clause must have to be considered valid.
# Fragments shorter than this (e.g. lone punctuation artefacts) are skipped.
_MIN_CLAUSE_CHARS = 3


# ── Public API ────────────────────────────────────────────────────────────────

def _get_segment_bounds(text: str) -> List[Tuple[int, int]]:
    """Return (start, end) char ranges for each clause (exclusive of boundary tokens).

    Boundary tokens themselves are excluded so clause text stays clean
    (no trailing comma, no leading "but", etc.).
    """
    bounds: List[Tuple[int, int]] = []
    prev_end = 0
    for m in _BOUNDARY_RE.finditer(text):
        bounds.append((prev_end, m.start()))   # clause before boundary
        prev_end = m.end()                     # next clause starts after boundary
    bounds.append((prev_end, len(text)))       # last (or only) clause
    return bounds


def split_into_clauses(text: str) -> List[Tuple[str, int]]:
    """Split *text* into clauses at recognised boundary tokens.

    Boundary tokens (commas, conjunctions) are excluded from clause text.

    Returns a list of ``(clause_text, start_offset)`` tuples where
    *start_offset* is the character position of the clause's first
    non-whitespace character in the original *text*.

    Falls back to ``[(text.strip(), 0)]`` when no boundary is found.
    """
    clauses: List[Tuple[str, int]] = []
    for raw_start, raw_end in _get_segment_bounds(text):
        raw_seg  = text[raw_start:raw_end]
        stripped = raw_seg.strip()
        if len(stripped) < _MIN_CLAUSE_CHARS:
            continue
        leading = len(raw_seg) - len(raw_seg.lstrip())
        clauses.append((stripped, raw_start + leading))

    return clauses if clauses else [(text.strip(), 0)]


def extract_aspect_clause(
    text: str,
    aspect_char_start: int,
    aspect_char_end: int,
) -> Tuple[str, int, int]:
    """Return the clause that fully contains the aspect term.

    The returned character offsets are relative to the extracted clause,
    not the original sentence.  Boundary tokens (commas, conjunctions)
    are excluded from the returned clause text.

    Args:
        text:              Full sentence text (after ``$T$`` replacement).
        aspect_char_start: Start char offset of the aspect in *text* (inclusive).
        aspect_char_end:   End char offset of the aspect in *text* (inclusive).

    Returns:
        ``(clause_text, new_aspect_char_start, new_aspect_char_end)``

        If no single clause fully contains the aspect (e.g. the aspect
        straddles a boundary, or the sentence has no recognised boundaries),
        the original *text* and positions are returned unchanged.
    """
    # Nothing to do when there is no valid aspect span.
    if aspect_char_start < 0 or aspect_char_end < aspect_char_start:
        return text, aspect_char_start, aspect_char_end

    for raw_start, raw_end in _get_segment_bounds(text):
        # Aspect must be fully inside this segment (raw, before stripping).
        if raw_start <= aspect_char_start and aspect_char_end < raw_end:
            raw_seg  = text[raw_start:raw_end]
            stripped = raw_seg.strip()

            if len(stripped) < _MIN_CLAUSE_CHARS:
                break  # degenerate clause — fall back to full sentence

            leading          = len(raw_seg) - len(raw_seg.lstrip())
            clause_start     = raw_start + leading
            new_aspect_start = aspect_char_start - clause_start
            new_aspect_end   = aspect_char_end   - clause_start
            return stripped, new_aspect_start, new_aspect_end

    # Fallback: return the original sentence and positions unchanged.
    return text, aspect_char_start, aspect_char_end
