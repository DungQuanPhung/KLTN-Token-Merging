# -*- coding: utf-8 -*-
"""Dataset utilities for GAS (Generative Aspect Sentiment) training.

Target format  : "(aspect1, positive); (aspect2, negative)"
Input sources  :
    - 4-line .apc files  (main data; has aspect_term, aspect_category, sentiment)
    - TSV supplement     (text / term / sentiment only; no aspect_category)

Main .apc samples are grouped by sentence so that each unique sentence
produces one record with all its aspects listed in the target string.

Supplement rows each produce an independent single-aspect record.
Both sources contribute to GAS training because GAS predicts (aspect, sentiment)
and does NOT need the aspect_category label — category is handled separately
by the BERT APC model.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer, T5Tokenizer

# ─── Constants ────────────────────────────────────────────────────────────────

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "dataset"
SPLIT_FILES = {"train": "train.apc", "dev": "dev.apc", "test": "test.apc"}
VALID_SENTIMENTS: Set[str] = {"positive", "negative", "neutral"}

# Matches "(some aspect term, sentiment)" — sentiment must be one of the 3 classes.
# Using a non-greedy group for the aspect to correctly handle commas at the end.
_GAS_RE = re.compile(
    r"\(\s*(.+?),\s*(positive|negative|neutral)\s*\)",
    re.IGNORECASE,
)


# ─── Target string helpers ────────────────────────────────────────────────────

def aspects_sentiments_to_target(pairs: List[Tuple[str, str]]) -> str:
    """Convert [(aspect, sentiment), ...] → GAS target string.

    Example:
        [("food", "positive"), ("service", "negative")]
        -> "(food, positive); (service, negative)"
    """
    cleaned = [
        (a.strip(), s.strip().lower())
        for a, s in pairs
        if a and a.strip() and s and s.strip().lower() in VALID_SENTIMENTS
    ]
    if not cleaned:
        return "none"
    return "; ".join(f"({a}, {s})" for a, s in cleaned)


def parse_gas_target(text: str) -> List[Tuple[str, str]]:
    """Parse a GAS target string back to a list of (aspect_term, sentiment) pairs.

    Handles "none" → [].
    """
    stripped = text.strip()
    if not stripped or stripped.lower() == "none":
        return []
    return [
        (m.group(1).strip(), m.group(2).strip().lower())
        for m in _GAS_RE.finditer(stripped)
    ]


# ─── .apc file parser ─────────────────────────────────────────────────────────

def parse_apc_file_for_gas(path: str | Path) -> List[Dict[str, str]]:
    """Parse a 4-line .apc file into raw sample dicts.

    Each 4-line block → one dict:
        text            — sentence with $T$ replaced by aspect_term
        aspect_term     — the aspect term
        aspect_category — category label (e.g. FOOD, SERVICE)
        sentiment       — lowercase sentiment label
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    samples: List[Dict[str, str]] = []
    i = 0
    while i + 3 < len(lines):
        sentence    = lines[i].strip()
        aspect_term = lines[i + 1].strip()
        aspect_cat  = lines[i + 2].strip()
        sentiment   = lines[i + 3].strip().lower()
        i += 4

        if not sentence or not aspect_term or sentiment not in VALID_SENTIMENTS:
            continue

        text = sentence.replace("$T$", aspect_term).strip()
        samples.append(
            {
                "text":             text,
                "aspect_term":      aspect_term,
                "aspect_category":  aspect_cat or "UNKNOWN",
                "sentiment":        sentiment,
            }
        )
    return samples


# ─── Supplement TSV parser ────────────────────────────────────────────────────

def parse_supplement_tsv_for_gas(path: str | Path) -> List[Dict[str, str]]:
    """Parse a TSV supplement file (text, term, sentiment) for GAS training.

    Supplement files have no aspect_category column, which is fine for GAS
    because GAS only predicts (aspect_term, sentiment).
    """
    samples: List[Dict[str, str]] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text_raw  = str(row.get("text",  "")).strip()
            term      = str(row.get("term",  "")).strip()
            sentiment = str(row.get("sentiment", "")).strip().lower()

            if not text_raw or not term or sentiment not in VALID_SENTIMENTS:
                continue

            text = text_raw.replace("$T$", term).strip()
            samples.append(
                {
                    "text":         text,
                    "aspect_term":  term,
                    "sentiment":    sentiment,
                }
            )
    return samples


# ─── Record builder ───────────────────────────────────────────────────────────

def load_gas_records(
    path: str | Path,
    supplement_paths: Optional[List[str | Path]] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """Build GAS records from a .apc file plus optional supplement TSV files.

    Main .apc samples are *grouped by sentence* (identical text after $T$
    substitution) so each sentence produces one record covering all its aspects.

    Supplement rows are kept as *independent* records (each row = its own
    sentence context after $T$ substitution → cannot be meaningfully grouped).

    Returns
    -------
    main_records : List[Dict]
        One record per unique sentence from the .apc file.
        Includes ``gold_triples`` with (aspect, category, sentiment) for evaluation.
    all_records : List[Dict]
        main_records + supplement records; use as the training dataset.

    Record schema
    -------------
    input_text          : str  — the sentence
    target_text         : str  — GAS target "(aspect, sentiment); ..."
    aspects             : List[str]  — aspect terms (for ATE evaluation)
    aspects_sentiments  : List[Tuple[str, str]]  — (aspect, sentiment) pairs
    gold_triples        : List[Tuple[str, str, str]]  — (aspect, category, sentiment)
                           (empty for supplement records — no category)
    """
    raw = parse_apc_file_for_gas(path)

    # Group main .apc samples by sentence text
    grouped: Dict[str, List] = {}
    for s in raw:
        text = s["text"]
        if text not in grouped:
            grouped[text] = []
        grouped[text].append(s)

    main_records: List[Dict] = []
    for text in sorted(grouped):
        samples = grouped[text]

        # Deduplicate while preserving insertion order
        seen_pairs: Set[Tuple[str, str]] = set()
        unique_pairs: List[Tuple[str, str]] = []
        seen_triples: Set[Tuple[str, str, str]] = set()
        unique_triples: List[Tuple[str, str, str]] = []

        for s in samples:
            pair = (s["aspect_term"], s["sentiment"])
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                unique_pairs.append(pair)

            triple = (s["aspect_term"], s["aspect_category"], s["sentiment"])
            if triple not in seen_triples:
                seen_triples.add(triple)
                unique_triples.append(triple)

        main_records.append(
            {
                "input_text":         text,
                "target_text":        aspects_sentiments_to_target(unique_pairs),
                "aspects":            [p[0] for p in unique_pairs],
                "aspects_sentiments": unique_pairs,
                "gold_triples":       unique_triples,
            }
        )

    # Supplement records (independent single-aspect samples)
    supplement_records: List[Dict] = []
    if supplement_paths:
        for sp in supplement_paths:
            sp_path = Path(sp)
            if not sp_path.is_file():
                print(f"[GAS] supplement file not found (skipped): {sp_path}")
                continue
            for s in parse_supplement_tsv_for_gas(sp_path):
                pair = (s["aspect_term"], s["sentiment"])
                supplement_records.append(
                    {
                        "input_text":         s["text"],
                        "target_text":        aspects_sentiments_to_target([pair]),
                        "aspects":            [s["aspect_term"]],
                        "aspects_sentiments": [pair],
                        "gold_triples":       [],  # no category in supplement
                    }
                )

    n_main = len(main_records)
    n_supp = len(supplement_records)
    print(
        f"[GAS] {Path(path).name}: {n_main} main sentences"
        + (f" + {n_supp} supplement = {n_main + n_supp} total" if n_supp else "")
    )

    all_records = main_records + supplement_records
    return main_records, all_records


def load_split_records(
    split: str,
    data_dir: Optional[str | Path] = None,
    supplement_paths: Optional[List[str | Path]] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """Load main + all records for train / dev / test split.

    Supplement is only appended for the train split.
    """
    if split not in SPLIT_FILES:
        raise ValueError(f"Unknown split {split!r}; expected one of {list(SPLIT_FILES)}")
    root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    path = root / SPLIT_FILES[split]
    if not path.is_file():
        raise FileNotFoundError(f"Missing split file: {path}")
    supp = supplement_paths if split == "train" else None
    return load_gas_records(path, supp)


# ─── PyTorch Dataset ──────────────────────────────────────────────────────────

class GASDataset(Dataset):
    """Tokenised (sentence → aspect+sentiment sequence) pairs for T5 seq2seq."""

    def __init__(
        self,
        records: Sequence[Dict],
        tokenizer: PreTrainedTokenizer,
        max_input_length: int = 128,
        max_target_length: int = 128,
    ) -> None:
        self.tokenizer         = tokenizer
        self.max_input_length  = max_input_length
        self.max_target_length = max_target_length
        self.records           = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row         = self.records[idx]
        input_text  = str(row["input_text"])
        target_text = str(row["target_text"])

        model_inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        label_ids = labels["input_ids"].squeeze(0)
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      model_inputs["input_ids"].squeeze(0),
            "attention_mask": model_inputs["attention_mask"].squeeze(0),
            "labels":         label_ids,
        }


# ─── Tokenizer & DataLoader builders ─────────────────────────────────────────

def build_tokenizer(model_name: str = "t5-base") -> T5Tokenizer:
    return T5Tokenizer.from_pretrained(model_name)


def create_gas_dataloaders(
    data_dir: Optional[str | Path] = None,
    supplement_paths: Optional[List[str | Path]] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    batch_size: int = 16,
    max_input_length: int = 128,
    max_target_length: int = 128,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, PreTrainedTokenizer, List[Dict], List[Dict]]:
    """Build train / dev / test DataLoaders for GAS.

    Returns
    -------
    train_loader, dev_loader, test_loader, tokenizer, dev_main_records, test_main_records

    ``dev_main_records`` and ``test_main_records`` are the ungrouped record lists
    used by the trainer for generative evaluation (no supplement records included).
    """
    tok = tokenizer or build_tokenizer()

    train_main, train_all = load_split_records("train", data_dir, supplement_paths)
    dev_main,   _         = load_split_records("dev",   data_dir)
    test_main,  _         = load_split_records("test",  data_dir)

    def _loader(records: Sequence[Dict], shuffle: bool) -> DataLoader:
        ds = GASDataset(records, tok, max_input_length, max_target_length)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return (
        _loader(train_all, shuffle=True),
        _loader(dev_main,  shuffle=False),
        _loader(test_main, shuffle=False),
        tok,
        dev_main,
        test_main,
    )
