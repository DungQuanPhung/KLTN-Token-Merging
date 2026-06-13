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

from clause_splitting import extract_aspect_clause

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

        # Locate aspect span in the final text (first occurrence; matches LCF-ATEPC).
        aspect_stripped = (aspect_term or "").strip()
        if aspect_stripped:
            aspect_char_start = text.find(aspect_stripped)
        else:
            aspect_char_start = -1
        aspect_char_end = (
            aspect_char_start + len(aspect_stripped) - 1
            if aspect_char_start >= 0 else -1
        )

        samples.append(
            {
                "text": text,
                "aspect_term": aspect_term,
                "aspect_category": aspect_cat if aspect_cat else "UNKNOWN",
                "sentiment": sentiment,
                "aspect_char_start": aspect_char_start,
                "aspect_char_end":   aspect_char_end,
            }
        )
    return samples


def build_lcf_vector(
    hidden_states: torch.Tensor,
    aspect_start: int,
    aspect_end: int,
    srd_threshold: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build an LCF vector using the original CDW formulation from LCF-ATEPC.

    Args:
        hidden_states: Tensor of shape [seq_len, hidden_dim].
        aspect_start:  Start index of the aspect span (inclusive).
        aspect_end:    End index of the aspect span (inclusive).
        srd_threshold: SRD threshold for full-weight region.

    Returns:
        lcf_vec : Tensor [hidden_dim] containing the weighted mean vector.
        weights : Tensor [seq_len] of CDW weights for each token.
        srd     : Tensor [seq_len] of SRD values for each token.
    """
    seq_len = hidden_states.size(0)

    # Aspect center is the midpoint of the aspect span.
    center = (aspect_start + aspect_end) / 2.0

    # Effective half-length of the aspect span used in SRD calculation.
    aspect_len = aspect_end - aspect_start + 1
    half_aspect = torch.floor(torch.tensor(aspect_len / 2.0, dtype=torch.float32))

    # Token positions from 0 to seq_len-1.
    positions = torch.arange(seq_len, dtype=torch.float32, device=hidden_states.device)

    # SRD_i = abs(i - center) - floor(aspect_len / 2), clamped at 0.
    srd = torch.abs(positions - center) - half_aspect
    srd = torch.clamp(srd, min=0.0)

    # CDW weight: 1.0 inside the threshold zone, then linearly decays.
    weights = torch.where(
        srd <= srd_threshold,
        torch.ones_like(srd),
        (seq_len - (srd - srd_threshold)) / float(seq_len),
    )
    weights = torch.clamp(weights, min=0.0, max=1.0)

    # Apply weights to hidden states and average to produce local-context vector.
    weighted_hidden = hidden_states * weights.unsqueeze(-1)
    lcf_vec = weighted_hidden.mean(dim=0)

    return lcf_vec, weights, srd


def build_lcf_vector_cdm(
    hidden_states: torch.Tensor,
    aspect_start: int,
    aspect_end: int,
    srd_threshold: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build an LCF vector using Context Dynamic Mask (CDM) formulation.

    Args:
        hidden_states: Tensor of shape [seq_len, hidden_dim].
        aspect_start:  Start index of the aspect span (inclusive).
        aspect_end:    End index of the aspect span (inclusive).
        srd_threshold: SRD threshold for binary mask region (default 5).

    Returns:
        lcf_vec : Tensor [hidden_dim] containing the masked mean vector.
        weights : Tensor [seq_len] of CDM mask {0, 1} for each token.
        srd     : Tensor [seq_len] of SRD values for each token.
    """
    seq_len = hidden_states.size(0)

    # Aspect center is the midpoint of the aspect span.
    center = (aspect_start + aspect_end) / 2.0

    # Effective half-length of the aspect span used in SRD calculation.
    aspect_len = aspect_end - aspect_start + 1
    half_aspect = torch.floor(torch.tensor(aspect_len / 2.0, dtype=torch.float32))

    # Token positions from 0 to seq_len-1.
    positions = torch.arange(seq_len, dtype=torch.float32, device=hidden_states.device)

    # SRD_i = abs(i - center) - floor(aspect_len / 2), clamped at 0.
    srd = torch.abs(positions - center) - half_aspect
    srd = torch.clamp(srd, min=0.0)

    # CDM mask: 1.0 inside threshold, 0.0 outside (binary).
    weights = torch.where(
        srd <= srd_threshold,
        torch.ones_like(srd),
        torch.zeros_like(srd),
    )

    # Apply weights to hidden states and average to produce local-context vector.
    weighted_hidden = hidden_states * weights.unsqueeze(-1)
    lcf_vec = weighted_hidden.mean(dim=0)

    return lcf_vec, weights, srd


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

            term_stripped = term.strip()
            if term_stripped:
                aspect_char_start = text.find(term_stripped)
            else:
                aspect_char_start = -1
            aspect_char_end = (
                aspect_char_start + len(term_stripped) - 1
                if aspect_char_start >= 0 else -1
            )

            samples.append(
                {
                    "text":              text,
                    "aspect_term":       term,
                    "sentiment":         sentiment,
                    "aspect_char_start": aspect_char_start,
                    "aspect_char_end":   aspect_char_end,
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
        clause_split_mode: str = "none",
    ) -> None:
        """
        Args:
            apc_path          : path to the main 4-line .apc file.
            tokenizer         : HuggingFace tokenizer.
            aspect_cat_map    : {category_str: int} mapping (main categories only,
                                no SUPPLEMENT placeholder).
            max_seq_len       : max tokenisation length.
            supplement_paths  : optional list of TSV supplement files to append
                                to training data (typically only for train split).
                                These samples set is_supplement=True and are
                                excluded from category-head training.
            clause_split_mode : ``"none"`` — use full sentences (default);
                                ``"rulebase"`` — regex split on comma/semicolon/
                                conjunctions; ``"uos"`` — LLM-based UOS segmenter.
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

        # Clause-level preprocessing: narrow each sentence to the clause
        # containing its aspect term and adjust character offsets accordingly.
        if clause_split_mode != "none":
            n_shortened = 0
            for r in raw:
                clause, new_cs, new_ce = extract_aspect_clause(
                    r["text"], r["aspect_char_start"], r["aspect_char_end"],
                    mode=clause_split_mode,
                )
                if clause != r["text"]:
                    n_shortened += 1
                r["text"]              = clause
                r["aspect_char_start"] = new_cs
                r["aspect_char_end"]   = new_ce
            print(
                f"[ApcFileDataset] clause_split_mode={clause_split_mode!r}:"
                f" {n_shortened}/{len(raw)} samples shortened to aspect clause"
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
                return_offsets_mapping=True,
            )

            input_ids      = enc["input_ids"].squeeze(0)           # (L,)
            attention_mask = enc["attention_mask"].squeeze(0)      # (L,)
            token_type_ids = enc.get(
                "token_type_ids", torch.zeros_like(input_ids)
            ).squeeze(0)
            offsets        = enc["offset_mapping"].squeeze(0)      # (L, 2)

            # ── Build LCF aspect indicator (binary, 1.0 at aspect subwords) ──
            # Mark tokens in segment A whose char offsets overlap the aspect span.
            # The downstream model converts this binary mask into CDW weights
            # (LCF-ATEPC, Zeng 2019).  Falls back to segment B (aspect copy) if
            # the aspect was truncated out of segment A.
            asp_cs = row.get("aspect_char_start", -1)
            asp_ce = row.get("aspect_char_end",   -1)
            lcf_vec = torch.zeros_like(input_ids, dtype=torch.float32)
            if asp_cs >= 0 and asp_ce >= asp_cs:
                for k in range(input_ids.size(0)):
                    if attention_mask[k].item() == 0:
                        continue
                    if token_type_ids[k].item() != 0:        # only segment A
                        continue
                    tok_s = int(offsets[k, 0].item())
                    tok_e = int(offsets[k, 1].item())
                    if tok_s == 0 and tok_e == 0:            # special tokens
                        continue
                    if tok_e > asp_cs and tok_s <= asp_ce:   # overlap
                        lcf_vec[k] = 1.0
            if lcf_vec.sum().item() == 0:
                # Aspect missing from segment A (e.g. truncated) — fall back to
                # the segment-B aspect copy so the local stream still has signal.
                lcf_vec = token_type_ids.float()

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
                return_offsets_mapping=True,
            )

            input_ids      = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            token_type_ids = enc.get(
                "token_type_ids", torch.zeros_like(input_ids)
            ).squeeze(0)
            offsets        = enc["offset_mapping"].squeeze(0)

            asp_stripped = aspect.strip()
            asp_cs = text.find(asp_stripped) if asp_stripped else -1
            asp_ce = asp_cs + len(asp_stripped) - 1 if asp_cs >= 0 else -1
            lcf_vec = torch.zeros_like(input_ids, dtype=torch.float32)
            if asp_cs >= 0:
                for k in range(input_ids.size(0)):
                    if attention_mask[k].item() == 0:
                        continue
                    if token_type_ids[k].item() != 0:
                        continue
                    tok_s = int(offsets[k, 0].item())
                    tok_e = int(offsets[k, 1].item())
                    if tok_s == 0 and tok_e == 0:
                        continue
                    if tok_e > asp_cs and tok_s <= asp_ce:
                        lcf_vec[k] = 1.0
            if lcf_vec.sum().item() == 0:
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
