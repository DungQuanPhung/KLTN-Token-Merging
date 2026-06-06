# -*- coding: utf-8 -*-
"""T5 wrapper for generative Aspect Term Extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


DEFAULT_MODEL_NAME = "t5-base"


@dataclass
class GenerationConfig:
    max_length: int = 64
    num_beams: int = 4
    early_stopping: bool = True


class T5AspectExtractor:
    """Load T5-base and expose training / generation helpers."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[torch.device] = None,
        generation: Optional[GenerationConfig] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.generation = generation or GenerationConfig()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)

    @property
    def config(self) -> GenerationConfig:
        return self.generation

    def save(self, output_dir: str) -> None:
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        device: Optional[torch.device] = None,
        generation: Optional[GenerationConfig] = None,
    ) -> "T5AspectExtractor":
        obj = cls.__new__(cls)
        obj.model_name = model_dir
        obj.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        obj.generation = generation or GenerationConfig()
        obj.tokenizer = T5Tokenizer.from_pretrained(model_dir)
        obj.model = T5ForConditionalGeneration.from_pretrained(model_dir)
        obj.model.to(obj.device)
        obj.model.eval()
        return obj

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **override_kwargs,
    ) -> torch.Tensor:
        gen_kwargs = {
            "max_length": self.generation.max_length,
            "num_beams": self.generation.num_beams,
            "early_stopping": self.generation.early_stopping,
        }
        gen_kwargs.update(override_kwargs)

        self.model.eval()
        with torch.no_grad():
            return self.model.generate(
                input_ids=input_ids.to(self.device),
                attention_mask=(
                    attention_mask.to(self.device) if attention_mask is not None else None
                ),
                **gen_kwargs,
            )
