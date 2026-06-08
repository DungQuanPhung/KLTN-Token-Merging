# -*- coding: utf-8 -*-
"""T5 wrapper for Generative Aspect Sentiment (GAS) extraction.

Input  : raw sentence
Output : "(aspect1, positive); (aspect2, negative)"

Aspect terms in the generated string are post-processed with Levenshtein
n-gram normalization (same technique used by the ATE T5 model) to correct
minor generation errors.  Sentiment labels are validated against the fixed
vocabulary {positive, negative, neutral}.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gas.dataset import parse_gas_target
from src.normalization import build_ngram_vocabulary, normalize_aspect


@dataclass
class GenerationConfig:
    max_length:    int  = 128
    num_beams:     int  = 4
    early_stopping: bool = True


class GasT5Model:
    """T5-based generative model that extracts (aspect_term, sentiment) pairs.

    Architecture is identical to T5AspectExtractor in ``src/model.py``, but
    the target format is "(aspect, sentiment)" instead of "(aspect)" only,
    and the output parser understands that format.
    """

    def __init__(
        self,
        model_name: str = "t5-base",
        device: Optional[torch.device] = None,
        generation: Optional[GenerationConfig] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.generation = generation or GenerationConfig()
        self.tokenizer  = T5Tokenizer.from_pretrained(model_name)
        self.model      = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, output_dir: str | Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(out))
        self.tokenizer.save_pretrained(str(out))

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        device: Optional[torch.device] = None,
        generation: Optional[GenerationConfig] = None,
    ) -> "GasT5Model":
        obj = cls.__new__(cls)
        obj.model_name = str(model_dir)
        obj.device     = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        obj.generation = generation or GenerationConfig()
        obj.tokenizer  = T5Tokenizer.from_pretrained(str(model_dir))
        obj.model      = T5ForConditionalGeneration.from_pretrained(str(model_dir))
        obj.model.to(obj.device)
        obj.model.eval()
        return obj

    # ── Generation ────────────────────────────────────────────────────────────

    def _generate_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **override_kwargs,
    ) -> torch.Tensor:
        gen_kwargs = {
            "max_length":     self.generation.max_length,
            "num_beams":      self.generation.num_beams,
            "early_stopping": self.generation.early_stopping,
        }
        gen_kwargs.update(override_kwargs)
        self.model.eval()
        with torch.no_grad():
            return self.model.generate(
                input_ids=input_ids.to(self.device),
                attention_mask=(
                    attention_mask.to(self.device)
                    if attention_mask is not None else None
                ),
                **gen_kwargs,
            )

    def _encode(self, sentence: str, max_input_length: int = 128) -> dict:
        return self.tokenizer(
            sentence,
            max_length=max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_one(
        self,
        sentence: str,
        normalize: bool = True,
        max_input_length: int = 128,
    ) -> List[Tuple[str, str]]:
        """Return [(aspect_term, sentiment), ...] for one sentence."""
        enc       = self._encode(sentence, max_input_length)
        gen_ids   = self._generate_ids(enc["input_ids"], enc["attention_mask"])
        raw_text  = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        return _decode_and_normalize(raw_text, sentence, normalize)

    def predict_for_records(
        self,
        records: Sequence[dict],
        normalize: bool = True,
        max_input_length: int = 128,
    ) -> Tuple[List[List[Tuple[str, str]]], List[List[Tuple[str, str]]]]:
        """Run GAS inference over a list of records.

        Returns
        -------
        predictions : one List[(aspect, sentiment)] per record
        golds       : gold ``aspects_sentiments`` from each record
        """
        predictions: List[List[Tuple[str, str]]] = []
        golds:       List[List[Tuple[str, str]]] = []

        for row in records:
            sentence   = str(row["input_text"])
            gold_pairs = list(row.get("aspects_sentiments") or [])
            pred_pairs = self.predict_one(sentence, normalize, max_input_length)
            predictions.append(pred_pairs)
            golds.append(gold_pairs)

        return predictions, golds

    def predict_batch(
        self,
        sentences: Sequence[str],
        batch_size: int = 16,
        normalize:  bool = True,
        max_input_length: int = 128,
    ) -> List[List[Tuple[str, str]]]:
        """Run GAS inference on a list of sentences with mini-batching."""
        all_preds: List[List[Tuple[str, str]]] = []
        self.model.eval()

        for start in range(0, len(sentences), batch_size):
            chunk = list(sentences[start : start + batch_size])
            enc = self.tokenizer(
                chunk,
                max_length=max_input_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            gen_ids  = self._generate_ids(enc["input_ids"], enc["attention_mask"])
            decoded  = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

            for sentence, raw_text in zip(chunk, decoded):
                all_preds.append(_decode_and_normalize(raw_text, sentence, normalize))

        return all_preds


# ─── Output parsing & normalization ──────────────────────────────────────────

def _decode_and_normalize(
    raw_text: str,
    sentence: str,
    normalize: bool,
) -> List[Tuple[str, str]]:
    """Parse raw generated text and optionally normalize aspect terms."""
    pairs = parse_gas_target(raw_text)
    if not normalize or not pairs:
        return pairs
    vocab = build_ngram_vocabulary(sentence)
    return [
        (normalize_aspect(aspect, vocab), sentiment)
        for aspect, sentiment in pairs
    ]
