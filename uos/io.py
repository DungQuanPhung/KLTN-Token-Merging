"""Load input records for UOS segmentation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List, Union


def iter_records_txt(path: Union[str, Path]) -> Iterator[dict]:
    """Read 4-line-per-block ABSA dataset (.apc format).

    Format (each block of 4 non-empty lines):
        sentence  (may contain $T$ as aspect placeholder)
        aspect_term
        category
        sentiment
    Blocks may be separated by blank lines.

    $T$ in the sentence is replaced with the aspect term before segmentation
    so the LLM receives a complete, readable sentence.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    buf: list[str] = []
    idx = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            buf = []
            continue
        buf.append(stripped)
        if len(buf) == 4:
            raw_sentence, aspect_term, category, sentiment = buf
            sentence = raw_sentence.replace("$T$", aspect_term)
            yield {
                "entity_id": str(idx),
                "review_id": str(idx),
                "sentence_idx": idx,
                "sentence": sentence,
                "raw_sentence": raw_sentence,
                "aspect_terms": aspect_term,
                "categories": category,
                "sentiments": sentiment,
                "text_column": "sentence",
            }
            idx += 1
            buf = []


def iter_records_json(
    path: Union[str, Path],
    *,
    text_column: str = "",
    max_rows: int = 0,
) -> Iterator[dict]:
    """Read flat JSON array of objects."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")

    fallbacks = ("sentence", "review", "caption", "text", "comment")
    sample = data[0] if isinstance(data[0], dict) else {}
    col = text_column or next((k for k in fallbacks if k in sample), "sentence")
    id_fallbacks = ("entity_id", "review_id", "id", "json_file")
    id_col = next((k for k in id_fallbacks if k in sample), "")

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        raw = item.get(col, "")
        if not isinstance(raw, str) or not raw.strip():
            continue
        row_id = str(item.get(id_col, i)) if id_col else str(i)
        yield {
            "entity_id": row_id,
            "review_id": row_id,
            "sentence_idx": i,
            "sentence": raw.strip(),
            "text_column": col,
        }
        if max_rows and i + 1 >= max_rows:
            break


def iter_records(
    path: Union[str, Path],
    *,
    text_column: str = "",
    max_rows: int = 0,
) -> Iterator[dict]:
    p = Path(path)
    if p.suffix in (".txt", ".apc", ".seg"):
        records = iter_records_txt(p)
        if max_rows:
            for i, r in enumerate(records):
                if i >= max_rows:
                    break
                yield r
        else:
            yield from records
    else:
        yield from iter_records_json(p, text_column=text_column, max_rows=max_rows)


def load_records(
    path: Union[str, Path],
    *,
    text_column: str = "",
    max_rows: int = 0,
) -> List[dict]:
    return list(iter_records(path, text_column=text_column, max_rows=max_rows))
