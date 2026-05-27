# -*- coding: utf-8 -*-
"""Dataset utilities for multi-task APC (aspect_category + sentiment).

Supports three input formats:
    - CSV files        (legacy, via ApcCSVDataset)
    - 4-line .apc files (main format, via ApcFileDataset)
    - TSV supplement   (text+$T$, term, sentiment — no category)

4-line .apc format (one sample = 4 lines):
    Line 1: sentence with $T$ placeholder
    Line 2: aspect_term  (replaces $T$ in sentence)
    Line 3: aspect_category  (e.g. SERVICE, FOOD, ROOM)
    Line 4: sentiment  (Positive / Negative / Neutral)

TSV supplement format (tab-separated, with header row):
    text    term    sentiment
    Used to reduce class imbalance; no aspect_category column.
    Supplement samples are flagged with is_supplement=True so the
    category head is NOT trained on them — only the sentiment head
    uses these samples.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

# ─── Label maps ───────────────────────────────────────────────────────────────

SENTIMENT_MAP: Dict[str, int] = {"positive": 0, "negative": 1, "neutral": 2}
SENTIMENT_LABELS = ["positive", "negative", "neutral"]


# ─── .apc file parser ─────────────────────────────────────────────────────────

def parse_apc_file(path: str) -> List[Dict[str, str]]:
    """Parse a 4-line .apc file into a list of sample dicts.

    Each sample occupies exactly 4 consecutive lines:
        sentence (with $T$), aspect_term, aspect_category, sentiment

    The $T$ placeholder is replaced by the aspect_term to produce the
    final ``text`` field used for tokenisation.

    Returns:
        List of dicts with keys: text, aspect_term, aspect_category, sentiment
        (sentiment is normalised to lowercase)
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    samples: List[Dict[str, str]] = []
    i = 0
    while i + 3 < len(lines):
        sentence      = lines[i].strip()
        aspect_term   = lines[i + 1].strip()
        aspect_cat    = lines[i + 2].strip()
        sentiment     = lines[i + 3].strip().lower()
        i += 4

        if not sentence or not sentiment:
            continue

        # Replace $T$ placeholder with the actual aspect term
        text = sentence.replace("$T$", aspect_term).strip()

        samples.append(
            {
                "text": text,
                "aspect_term": aspect_term,
                "aspect_category": aspect_cat if aspect_cat else "UNKNOWN",
                "sentiment": sentiment,
            }
        )
    return samples


def parse_supplement_tsv(path: str) -> List[Dict[str, str]]:
    """Parse a TSV supplement file (text / term / sentiment, no aspect_category).

    The ``$T$`` placeholder in ``text`` is replaced by ``term`` to produce the
    final ``text`` field used for tokenisation, matching the .apc convention.

    Supplement samples have no aspect_category — the returned dicts intentionally
    omit that key.  The category head must NOT be trained on supplement samples;
    use the ``is_supplement`` flag added by ``ApcFileDataset`` to mask them out.

    Args:
        path : path to the .tsv file (tab-separated, has header row).

    Returns:
        List of dicts with keys: text, aspect_term, sentiment
        (sentiment is normalised to lowercase; no aspect_category key)
    """
    samples: List[Dict[str, str]] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text_raw  = str(row.get("text",  "")).strip()
            term      = str(row.get("term",  "")).strip()
            sentiment = str(row.get("sentiment", "")).strip().lower()

            if not text_raw or not sentiment:
                continue

            text = text_raw.replace("$T$", term).strip()
            samples.append(
                {
                    "text":        text,
                    "aspect_term": term,
                    "sentiment":   sentiment,
                }
            )
    return samples


def build_label_maps_from_apc(
    *paths: str,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Build label maps from one or more .apc files (union of all categories).

    Only the main .apc files are used — supplement TSV files are intentionally
    excluded because supplement samples are not used to train the category head.

    Args:
        *paths : .apc file paths (train / dev / test).

    Returns:
        sentiment_map : {"positive": 0, "negative": 1, "neutral": 2}
        aspect_cat_map: {"AMENITY": 0, "BRANDING": 1, ...}  (alphabetical)
    """
    cats: set = set()
    for path in paths:
        for sample in parse_apc_file(path):
            cats.add(sample["aspect_category"])
    aspect_cat_map = {c: i for i, c in enumerate(sorted(cats))}
    return SENTIMENT_MAP, aspect_cat_map


# ─── CSV label maps (legacy) ──────────────────────────────────────────────────

def build_label_maps(
    train_csv: str, test_csv: str
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Build label maps from the union of train + test CSV aspect categories.

    Returns:
        sentiment_map : {"positive": 0, "negative": 1, "neutral": 2}
        aspect_cat_map: {"AMENITY": 0, "BRANDING": 1, ...}  (alphabetical order)
    """
    cats: set = set()
    for path in [train_csv, test_csv]:
        df = pd.read_csv(path)
        cats.update(df["aspect_category"].dropna().astype(str).unique())
    aspect_cat_map = {c: i for i, c in enumerate(sorted(cats))}
    return SENTIMENT_MAP, aspect_cat_map


# ─── .apc Dataset ─────────────────────────────────────────────────────────────

class ApcFileDataset(Dataset):
    """Tokenise samples from a 4-line .apc file for multi-task APC.

    Tokenisation format (SPC):
        "[CLS] text [SEP] aspect_term [SEP]"
    LCF vector:
        token_type_ids  → 1.0 for aspect-segment tokens (segment B),
                          0.0 for text-segment tokens (segment A).
        This naturally marks aspect positions for LCF masking.

    Each sample dict contains an ``is_supplement`` bool tensor:
        False → main .apc sample  (trains BOTH sentiment and category heads)
        True  → supplement sample (trains sentiment head ONLY)
    The caller is responsible for masking out supplement samples before
    computing the category loss.
    """

    def __init__(
        self,
        apc_path: str,
        tokenizer: PreTrainedTokenizer,
        aspect_cat_map: Dict[str, int],
        max_seq_len: int = 128,
        supplement_paths: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            apc_path         : path to the main 4-line .apc file.
            tokenizer        : HuggingFace tokenizer.
            aspect_cat_map   : {category_str: int} mapping (main categories only,
                               no SUPPLEMENT placeholder).
            max_seq_len      : max tokenisation length.
            supplement_paths : optional list of TSV supplement files to append
                               to training data (typically only for train split).
                               These samples set is_supplement=True and are
                               excluded from category-head training.
        """
        raw = parse_apc_file(apc_path)

        # Build is_supplement flag list in parallel with raw
        n_main = len(raw)
        is_supp_flags: List[bool] = [False] * n_main

        # Append supplement samples (only passed for training, not dev/test)
        if supplement_paths:
            for tsv_path in supplement_paths:
                extra = parse_supplement_tsv(tsv_path)
                raw.extend(extra)
                is_supp_flags.extend([True] * len(extra))
            n_supp = len(raw) - n_main
            print(
                f"[ApcFileDataset] {Path(apc_path).name}:"
                f" {n_main} main + {n_supp} supplement = {len(raw)} total samples"
                f" (supplement trains sentiment head only)"
            )

        pad_or_unk = tokenizer.pad_token or tokenizer.unk_token or "[PAD]"

        self.samples: list = []
        skipped = 0
        for row, is_supp in zip(raw, is_supp_flags):
            text          = row["text"]
            aspect        = row["aspect_term"] or pad_or_unk
            sentiment_str = row["sentiment"]

            if sentiment_str not in SENTIMENT_MAP:
                skipped += 1
                continue

            # SPC: text (segment A) + aspect_term (segment B)
            enc = tokenizer(
                text,
                aspect,
                max_length=max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            input_ids      = enc["input_ids"].squeeze(0)           # (L,)
            attention_mask = enc["attention_mask"].squeeze(0)      # (L,)
            token_type_ids = enc.get(
                "token_type_ids", torch.zeros_like(input_ids)
            ).squeeze(0)
            lcf_vec = token_type_ids.float()                       # (L,)

            # aspect_cat_label: real category for main samples; 0 (placeholder)
            # for supplement samples — will be masked out in the loss computation.
            if is_supp:
                cat_label = 0
            else:
                aspect_cat = row["aspect_category"]
                cat_label  = aspect_cat_map.get(aspect_cat, 0)

            self.samples.append(
                {
                    "input_ids":        input_ids,
                    "attention_mask":   attention_mask,
                    "lcf_vec":          lcf_vec,
                    "sentiment_label":  torch.tensor(
                        SENTIMENT_MAP[sentiment_str], dtype=torch.long
                    ),
                    "aspect_cat_label": torch.tensor(cat_label, dtype=torch.long),
                    "is_supplement":    torch.tensor(is_supp,   dtype=torch.bool),
                }
            )

        if skipped:
            print(f"[ApcFileDataset] {apc_path}: skipped {skipped} samples "
                  f"(unknown sentiment label)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


# ─── CSV Dataset (legacy) ─────────────────────────────────────────────────────

class ApcCSVDataset(Dataset):
    """Tokenise CSV rows for multi-task APC (aspect_category + sentiment).

    Input format (SPC):
        "[CLS] text [SEP] aspect_term [SEP]"
    LCF vector:
        token_type_ids  →  1.0 for aspect-segment tokens, 0.0 for text-segment
        (segment B = aspect term; naturally marks aspect positions for LCF)

    Expected CSV columns: text, aspect_term, aspect_category, sentiment
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer: PreTrainedTokenizer,
        aspect_cat_map: Dict[str, int],
        max_seq_len: int = 128,
    ) -> None:
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["text", "sentiment"])
        df["aspect_term"] = df["aspect_term"].fillna("").astype(str)
        df["aspect_category"] = df["aspect_category"].fillna("UNKNOWN").astype(str)

        pad_or_unk = tokenizer.pad_token or tokenizer.unk_token or "[PAD]"

        self.samples: list = []
        for _, row in df.iterrows():
            text          = str(row["text"]).strip()
            aspect        = str(row["aspect_term"]).strip() or pad_or_unk
            sentiment_str = str(row["sentiment"]).strip().lower()
            aspect_cat    = str(row["aspect_category"]).strip()

            if sentiment_str not in SENTIMENT_MAP:
                continue

            enc = tokenizer(
                text,
                aspect,
                max_length=max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            input_ids      = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            token_type_ids = enc.get(
                "token_type_ids", torch.zeros_like(input_ids)
            ).squeeze(0)
            lcf_vec = token_type_ids.float()

            self.samples.append(
                {
                    "input_ids":        input_ids,
                    "attention_mask":   attention_mask,
                    "lcf_vec":          lcf_vec,
                    "sentiment_label":  torch.tensor(
                        SENTIMENT_MAP[sentiment_str], dtype=torch.long
                    ),
                    "aspect_cat_label": torch.tensor(
                        aspect_cat_map.get(aspect_cat, 0), dtype=torch.long
                    ),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]
