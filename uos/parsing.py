"""Parse and normalize LLM UOS outputs."""
from __future__ import annotations

import json
import re
from typing import List


def strip_code_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```", "", text).strip()
    return re.sub(r"```$", "", text).strip()


def _strip_thinking(text: str) -> str:
    for open_tag, close_tag in (
        ("<" + "think" + ">", "</" + "think" + ">"),
        ("<think>", "</think>"),
    ):
        pattern = re.escape(open_tag) + r"[\s\S]*?" + re.escape(close_tag)
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def normalize_unit(unit: str) -> str:
    unit = re.sub(r"\s+", " ", unit).strip()
    if not unit:
        return ""
    if unit[-1] not in ".!?":
        unit += "."
    return unit


def _units_from_list(items: list) -> List[str]:
    units: List[str] = []
    for item in items:
        if isinstance(item, str) and item.strip():
            units.append(normalize_unit(item))
        elif isinstance(item, dict) and item.get("unit_sentence"):
            units.append(normalize_unit(str(item["unit_sentence"])))
    return [u for u in units if u]


def parse_units(text: str) -> List[str]:
    text = strip_code_fence(_strip_thinking(text))
    candidates = [text]
    for pattern in (r"\[[\s\S]*\]", r"\{[\s\S]*\}"):
        match = re.search(pattern, text)
        if match:
            candidates.insert(0, match.group(0))
    for chunk in candidates:
        try:
            parsed = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            units = _units_from_list(parsed)
            if units:
                return units
        if isinstance(parsed, dict):
            uos = parsed.get("uos", [])
            if isinstance(uos, list):
                units = _units_from_list(uos)
                if units:
                    return units
    return []


def validate_units(sentence: str, units: List[str]) -> List[str]:
    cleaned = [normalize_unit(u) for u in units if str(u).strip()]
    return cleaned if cleaned else [normalize_unit(sentence)]
