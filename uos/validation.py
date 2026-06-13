"""Rule-based warnings and gold-case checks."""
from __future__ import annotations

import re
from typing import List, Sequence

CONTRAST_MARKERS = (
    "but", "however", "although", "though", "while",
    "whereas", "yet", "except", "unfortunately",
)


def simple_tokens(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text.lower())


def unsupported_token_rate(sentence: str, units: Sequence[str]) -> float:
    original = set(simple_tokens(sentence))
    generated = simple_tokens(" ".join(units))
    allowed = {
        "the", "a", "an", "is", "are", "was", "were",
        "it", "they", "this", "that", "had", "has", "have",
        "be", "been", "very", "really",
    }
    if not generated:
        return 0.0
    unsupported = [t for t in generated if t not in original and t not in allowed]
    return len(unsupported) / len(generated)


def detect_warnings(sentence: str, units: Sequence[str]) -> List[str]:
    warnings: List[str] = []
    lower = sentence.lower()
    if not units:
        warnings.append("empty_units")
    if len(units) > 5:
        warnings.append("possible_over_split_too_many_units")
    has_contrast = any(re.search(rf"\b{re.escape(m)}\b", lower) for m in CONTRAST_MARKERS)
    logistics_but = re.search(r"\b(arrived|checked in|waited|got there)\b.*\bbut\b", lower)
    if has_contrast and len(units) == 1 and not logistics_but:
        warnings.append("possible_under_split_contrast")
    if re.search(r"\bloved\b.*\bwhole experience\b.*\bfrom\b", lower) and len(units) > 3:
        warnings.append("possible_over_split_global_experience")
    if re.search(r"\bthere (is|are|was|were)\b", lower) and len(units) > 1:
        warnings.append("possible_over_split_factual_sentence")
    layout_complaint = (
        re.search(r"\bcould(?:n't| not)\s+\w+\b.*\bbecause\b", lower)
        and re.search(r"\b(no|not enough)\s+space\b|\bphysically no space\b", lower)
    )
    if layout_complaint and len(units) > 1:
        warnings.append("possible_over_split_causal_layout_complaint")
    for unit in units:
        if len(unit.split()) > max(12, len(sentence.split()) * 1.5):
            warnings.append("unit_too_long")
    return sorted(set(warnings))


def check_case(case: dict, units: Sequence[str]) -> bool:
    joined = " ".join(units).lower()
    if "expect_exact_units" in case and len(units) != case["expect_exact_units"]:
        return False
    if "expect_max_units" in case and len(units) > case["expect_max_units"]:
        return False
    if "expect_min_units" in case and len(units) < case["expect_min_units"]:
        return False
    for needle in case.get("must_include", []):
        if needle.lower() not in joined:
            return False
    for needle in case.get("forbid_include", []):
        if needle.lower() in joined:
            return False
    return True
