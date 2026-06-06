# -*- coding: utf-8 -*-
"""ATE dataset loader for generative T5 training (GAS extraction-style)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer, T5Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

from thesis_apc_baseline.dataset_utils import parse_apc_file

DEFAULT_DATA_DIR = PROJECT_ROOT / "dataset"
SPLIT_FILES = {
    "train": "train.apc",
    "dev": "dev.apc",
    "test": "test.apc",
}


def aspects_to_target(aspects: Sequence[str]) -> str:
    """Convert aspect terms to GAS extraction-style target text."""
    cleaned = [a.strip() for a in aspects if a and a.strip()]
    if not cleaned:
        return "none"
    return "; ".join(f"({a})" for a in cleaned)


def load_ate_records(apc_path: str | Path) -> List[Dict[str, object]]:
    """Load ATE samples from a 4-line .apc file.

    Groups rows by final sentence text and merges aspect terms so one sentence
    can yield multiple aspects in the target string.
    """
    path = Path(apc_path)
    grouped: Dict[str, set[str]] = {}

    for row in parse_apc_file(str(path)):
        text = row["text"].strip()
        aspect = (row.get("aspect_term") or "").strip()
        if text not in grouped:
            grouped[text] = set()
        if aspect:
            grouped[text].add(aspect)

    records: List[Dict[str, object]] = []
    for text in sorted(grouped.keys()):
        aspects = sorted(grouped[text])
        records.append(
            {
                "input_text": text,
                "target_text": aspects_to_target(aspects),
                "aspects": aspects,
            }
        )
    return records


def load_split_records(
    split: str,
    data_dir: Optional[str | Path] = None,
) -> List[Dict[str, object]]:
    """Load train/dev/test split from the googomg .apc files (no supplement)."""
    if split not in SPLIT_FILES:
        raise ValueError(f"Unknown split {split!r}; expected one of {list(SPLIT_FILES)}")

    root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    apc_path = root / SPLIT_FILES[split]
    if not apc_path.is_file():
        raise FileNotFoundError(f"Missing split file: {apc_path}")

    records = load_ate_records(apc_path)
    print(f"[ATE dataset] {split}: {len(records)} sentences from {apc_path.name}")
    return records


class ATEDataset(Dataset):
    """Tokenised sentence → aspect-sequence pairs for T5."""

    def __init__(
        self,
        records: Sequence[Dict[str, object]],
        tokenizer: PreTrainedTokenizer,
        max_input_length: int = 128,
        max_target_length: int = 64,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.records[idx]
        input_text = str(row["input_text"])
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

        input_ids = model_inputs["input_ids"].squeeze(0)
        attention_mask = model_inputs["attention_mask"].squeeze(0)
        label_ids = labels["input_ids"].squeeze(0)
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_ids,
        }


def build_tokenizer(model_name: str = "t5-base") -> T5Tokenizer:
    return T5Tokenizer.from_pretrained(model_name)


def create_dataloaders(
    data_dir: Optional[str | Path] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    batch_size: int = 16,
    max_input_length: int = 128,
    max_target_length: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, PreTrainedTokenizer]:
    """Build train/dev/test DataLoaders from googomg .apc splits."""
    tok = tokenizer or build_tokenizer()

    train_records = load_split_records("train", data_dir)
    dev_records = load_split_records("dev", data_dir)
    test_records = load_split_records("test", data_dir)

    def _loader(records: Sequence[Dict[str, object]], shuffle: bool) -> DataLoader:
        ds = ATEDataset(
            records,
            tok,
            max_input_length=max_input_length,
            max_target_length=max_target_length,
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return (
        _loader(train_records, shuffle=True),
        _loader(dev_records, shuffle=False),
        _loader(test_records, shuffle=False),
        tok,
    )


def get_raw_split_records(
    split: str,
    data_dir: Optional[str | Path] = None,
) -> List[Dict[str, object]]:
    """Return raw records (with input_text / target_text / aspects) for a split."""
    return load_split_records(split, data_dir)
