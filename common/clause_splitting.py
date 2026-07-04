# -*- coding: utf-8 -*-
"""Clause-level preprocessing for APC.

Two modes are available, selected via CLAUSE_SPLIT_MODE or the ``mode``
parameter on each public function:

    "uos"       LLM-based segmentation via OllamaUOSSegmenter.  Decomposes a
                sentence into semantically self-contained, aspect-focused units
                (one aspect + one sentiment per unit).  Requires a running
                Ollama instance.  Falls back to the full sentence on
                connection/model errors.

    "rulebase"  Fast regex splitter: splits on comma, semicolon, and a fixed
                set of adversative/concessive conjunctions (but, yet, however,
                although, though, whereas).  No external service required.

Public API:

    from common.clause_splitting import extract_aspect_clause, split_into_clauses

    clauses = split_into_clauses("The room was clean but breakfast was cold.")
    # → [("The room was clean", 0), ("breakfast was cold.", 23)]  (rulebase)

    clause, new_start, new_end = extract_aspect_clause(
        text="The food was great, but the service was slow",
        aspect_char_start=28,
        aspect_char_end=34,
    )
    # → ("the service was slow", 4, 10)

    # Override mode per-call (ignores CLAUSE_SPLIT_MODE):
    clause, s, e = extract_aspect_clause(text, s, e, mode="rulebase")

Fallback behaviour (UOS mode only)
-----------------------------------
If Ollama is unreachable, the requested model is missing, or any runtime
error occurs, both functions degrade gracefully:

* ``split_into_clauses`` returns ``[(text.strip(), 0)]``.
* ``extract_aspect_clause`` returns ``(text, aspect_char_start, aspect_char_end)``.

Caching
-------
UOS results are memoised per sentence text (LRU, 2048 entries) so that
multiple aspects within the same sentence share one LLM call — important
when ``extract_aspect_clause`` is called in a tight loop over training
samples.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import List, Optional, Tuple

# ── Module-level mode config ──────────────────────────────────────────────────
# Controls which splitter is used when `mode` is not passed explicitly.
# Valid values: "uos" | "rulebase"
CLAUSE_SPLIT_MODE: str = "uos"

# ── Lazy-initialised global segmenter ─────────────────────────────────────────

_segmenter: Optional[object] = None
_segmenter_unavailable: bool = False   # set after first failed init attempt


def _get_segmenter() -> Optional[object]:
    """Return the cached OllamaUOSSegmenter, or None if unavailable."""
    global _segmenter, _segmenter_unavailable
    if _segmenter_unavailable:
        return None
    if _segmenter is None:
        try:
            from uos.segmenter import OllamaUOSSegmenter
            _segmenter = OllamaUOSSegmenter()
        except Exception as exc:
            print(
                f"[clause_splitting] UOS segmenter unavailable ({exc})"
                " — falling back to full sentence"
            )
            _segmenter_unavailable = True
            return None
    return _segmenter


@lru_cache(maxsize=2048)
def _cached_segment(text: str) -> Optional[tuple]:
    """Call UOS and return units as a tuple (hashable for lru_cache).

    Returns ``None`` on any error so callers can distinguish "no units"
    from "empty units".
    """
    seg = _get_segmenter()
    if seg is None:
        return None
    try:
        units: List[str] = seg.segment(text)
        return tuple(units) if units else None
    except Exception as exc:
        global _segmenter_unavailable
        # Mark permanently unavailable on connection errors so subsequent
        # calls to new texts skip the connection attempt entirely.
        msg = str(exc)
        if "Cannot connect" in msg or "Connection refused" in msg:
            _segmenter_unavailable = True
        print(f"[clause_splitting] UOS segment call failed: {exc}")
        return None


def _uos_units(text: str) -> Optional[List[str]]:
    result = _cached_segment(text)
    return list(result) if result is not None else None


# ── Rulebase helpers ──────────────────────────────────────────────────────────
# Boundaries: comma, semicolon, and a fixed set of adversative/concessive
# conjunctions that commonly separate independent opinion clauses.

_RULEBASE_BOUNDARY = re.compile(
    r'[,;]\s*|\s+(?:but|yet|however|although|though|whereas)\b\s*',
    re.IGNORECASE,
)


def _rulebase_split(text: str) -> List[Tuple[str, int]]:
    """Split *text* at comma/semicolon/conjunction boundaries (no LLM)."""
    result: List[Tuple[str, int]] = []
    last_end = 0
    for m in _RULEBASE_BOUNDARY.finditer(text):
        chunk = text[last_end : m.start()]
        stripped = chunk.strip()
        if stripped:
            leading = len(chunk) - len(chunk.lstrip())
            result.append((stripped, last_end + leading))
        last_end = m.end()
    trailing = text[last_end:]
    stripped = trailing.strip()
    if stripped:
        leading = len(trailing) - len(trailing.lstrip())
        result.append((stripped, last_end + leading))
    return result if result else [(text.strip(), 0)]


def _rulebase_extract_clause(
    text: str, aspect_char_start: int, aspect_char_end: int
) -> Tuple[str, int, int]:
    """Return the rulebase clause that fully contains the aspect span."""
    clauses = _rulebase_split(text)
    if len(clauses) <= 1:
        return text, aspect_char_start, aspect_char_end
    for clause_text, clause_offset in clauses:
        end_offset = clause_offset + len(clause_text) - 1
        if clause_offset <= aspect_char_start and aspect_char_end <= end_offset:
            return (
                clause_text,
                aspect_char_start - clause_offset,
                aspect_char_end - clause_offset,
            )
    return text, aspect_char_start, aspect_char_end


# ── Offset helpers ─────────────────────────────────────────────────────────────

def _find_unit_offset(text: str, unit: str) -> int:
    """Return the start char position of *unit* inside *text*.

    ``normalize_unit`` (in ``uos.parsing``) appends a period to units that
    lack sentence-final punctuation.  We therefore try searching for the
    unit verbatim, then with the trailing punctuation stripped, then
    case-insensitively.  Returns 0 as a safe fallback.
    """
    candidates = [unit, unit.rstrip(".!?"), unit.rstrip(".!?").strip()]
    for candidate in candidates:
        if not candidate:
            continue
        pos = text.find(candidate)
        if pos >= 0:
            return pos
        pos = text.lower().find(candidate.lower())
        if pos >= 0:
            return pos
    return 0


# ── Public API ─────────────────────────────────────────────────────────────────

def split_into_clauses(
    text: str,
    mode: Optional[str] = None,
) -> List[Tuple[str, int]]:
    """Segment *text* into clauses.

    Args:
        text: Input sentence.
        mode: ``"rulebase"`` for the regex splitter, ``"uos"`` for the LLM
              segmenter.  Defaults to :data:`CLAUSE_SPLIT_MODE` when ``None``.

    Returns a list of ``(clause_text, start_offset)`` tuples.  *start_offset*
    is the best-effort character position of the clause inside the original
    *text*; callers that only need the text may ignore it::

        for clause_text, _ in split_into_clauses(sentence):
            ...

    Falls back to ``[(text.strip(), 0)]`` when UOS is unavailable or returns
    no usable units.
    """
    effective_mode = mode if mode is not None else CLAUSE_SPLIT_MODE

    if effective_mode == "rulebase":
        return _rulebase_split(text)

    # UOS (default)
    units = _uos_units(text)
    if not units:
        return [(text.strip(), 0)]

    result: List[Tuple[str, int]] = [
        (unit, _find_unit_offset(text, unit)) for unit in units
    ]
    return result if result else [(text.strip(), 0)]


def extract_aspect_clause(
    text: str,
    aspect_char_start: int,
    aspect_char_end: int,
    mode: Optional[str] = None,
) -> Tuple[str, int, int]:
    """Return the clause that contains the aspect term.

    Returned character offsets are relative to the extracted clause, not the
    original sentence.

    Args:
        text:              Full sentence text.
        aspect_char_start: Start char offset of the aspect (inclusive).
        aspect_char_end:   End char offset of the aspect (inclusive).
        mode:              ``"rulebase"`` or ``"uos"``.  Defaults to
                           :data:`CLAUSE_SPLIT_MODE` when ``None``.

    Returns:
        ``(clause_text, new_aspect_start, new_aspect_end)``

        Falls back to ``(text, aspect_char_start, aspect_char_end)`` when
        the splitter is unavailable, returns only one clause, or no clause
        contains the aspect term.
    """
    if aspect_char_start < 0 or aspect_char_end < aspect_char_start:
        return text, aspect_char_start, aspect_char_end

    effective_mode = mode if mode is not None else CLAUSE_SPLIT_MODE

    if effective_mode == "rulebase":
        return _rulebase_extract_clause(text, aspect_char_start, aspect_char_end)

    # UOS path ────────────────────────────────────────────────────────────────
    aspect_term = text[aspect_char_start : aspect_char_end + 1]
    units = _uos_units(text)

    if not units or len(units) == 1:
        return text, aspect_char_start, aspect_char_end

    # Pass 1 — case-sensitive verbatim match
    for unit in units:
        pos = unit.find(aspect_term)
        if pos >= 0:
            return unit, pos, pos + len(aspect_term) - 1

    # Pass 2 — case-insensitive match (handles capitalisation from UOS rewrites)
    aspect_lower = aspect_term.lower()
    for unit in units:
        pos = unit.lower().find(aspect_lower)
        if pos >= 0:
            return unit, pos, pos + len(aspect_term) - 1

    # No unit contains the aspect term — return the original sentence unchanged.
    return text, aspect_char_start, aspect_char_end
