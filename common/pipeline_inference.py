# -*- coding: utf-8 -*-
"""End-to-end inference pipeline: ATE (T5) → APC (category + sentiment).

Step 1 — Aspect Term Extraction (ATE):
    Input : raw sentence
    Model : T5AspectExtractor (fine-tuned T5)
    Output: list of aspect terms

Step 2 — Aspect Polarity Classification (APC):
    Input : sentence + each extracted aspect term
    Model : FastLcfBertMultiTask (BERT multitask)
    Output: sentiment label + aspect category for each aspect

Usage
-----
    from common.pipeline_inference import PipelineInference

    pipe = PipelineInference.load(
        ate_checkpoint="checkpoints/ate/best",
        apc_checkpoint="checkpoints/apc/best/model.pt",
        apc_meta="checkpoints/apc/best/meta.json",
        bert_name="bert-base-uncased",
    )

    results = pipe.predict("The food was amazing but the service was slow")
    for r in results:
        print(r)
    # {"aspect": "food",    "sentiment": "positive", "category": "FOOD"}
    # {"aspect": "service", "sentiment": "negative",  "category": "SERVICE"}

CLI (from repo root):
    python common/pipeline_inference.py --ate-checkpoint ... --apc-checkpoint-dir ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import T5AspectExtractor
from src.inference import predict_aspects
from common.clause_splitting import extract_aspect_clause, split_into_clauses


SENTIMENT_LABELS = ["positive", "negative", "neutral"]


# ─── APC wrapper ──────────────────────────────────────────────────────────────

class APCPredictor:
    """Load a FastLcfBertMultiTask checkpoint and predict per-aspect labels."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        sentiment_labels: List[str],
        category_labels: List[str],
        device: torch.device,
        max_seq_len: int = 128,
        clause_split_mode: str = "none",
    ) -> None:
        self.model             = model
        self.tokenizer         = tokenizer
        self.sentiment_labels  = sentiment_labels
        self.category_labels   = category_labels
        self.device            = device
        self.max_seq_len       = max_seq_len
        self.clause_split_mode = clause_split_mode
        self.model.to(device)
        self.model.eval()

    @classmethod
    def load(
        cls,
        checkpoint_dir: str | Path,
        bert_name: str = "bert-base-uncased",
        device: Optional[torch.device] = None,
        max_seq_len: int = 128,
    ) -> "APCPredictor":
        """Load model weights + label maps from a checkpoint directory.

        Looks for (in order):
            <checkpoint_dir>/best_model.pt   (saved by run_joint_experiments.py)
            <checkpoint_dir>/model.pt        (saved by APCMultiTaskTrainer)
        and reads <checkpoint_dir>/meta.json for label maps + model config.

        Args:
            checkpoint_dir : directory containing model weights + meta.json
            bert_name      : HuggingFace BERT variant used during training
        """
        dev  = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = Path(checkpoint_dir)

        # Resolve weight file (best_model.pt first, then model.pt)
        weight_path = None
        for fname in ("best_model.pt", "model.pt"):
            candidate = ckpt / fname
            if candidate.is_file():
                weight_path = candidate
                break
        if weight_path is None:
            raise FileNotFoundError(
                f"No model weights found in {ckpt}. "
                "Expected best_model.pt or model.pt."
            )

        # Load meta.json
        meta: Dict = {}
        meta_path = ckpt / "meta.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))

        category_labels: List[str] = meta.get("category_labels", [])
        sentiment_labels: List[str] = meta.get("sentiment_labels", SENTIMENT_LABELS)
        num_aspect_cat = len(category_labels) if category_labels else meta.get("num_aspect_cat", 1)

        # Reconstruct model with same config used during training
        cfg = meta.get("config", {})
        from models.fast_lcf_bert_multitask import FastLcfBertMultiTask

        bert  = AutoModel.from_pretrained(bert_name)
        model = FastLcfBertMultiTask(
            bert=bert,
            num_sentiment=len(sentiment_labels),
            num_aspect_cat=num_aspect_cat,
            use_lcf=cfg.get("use_lcf", True),
            use_cdm=cfg.get("use_cdm", False),
            use_tome=cfg.get("use_tome", False),
            tome_resize=cfg.get("tome_resize", True),
            tome_merge_strategy=cfg.get("merge_strategy", "bipartite"),
        )
        state = torch.load(str(weight_path), map_location=dev)
        model.load_state_dict(state)

        # Read clause_split_mode from meta.json; fall back to legacy bool field.
        raw_mode = cfg.get("clause_split_mode")
        if raw_mode is not None:
            clause_split_mode: str = str(raw_mode)
        elif cfg.get("clause_split", False):
            clause_split_mode = "uos"
        else:
            clause_split_mode = "none"

        tokenizer = AutoTokenizer.from_pretrained(bert_name)
        print(f"[APC] Loaded weights from {weight_path.name}")
        print(f"[APC] sentiment={sentiment_labels}  categories={category_labels}")
        print(f"[APC] clause_split_mode={clause_split_mode!r}")

        return cls(
            model=model,
            tokenizer=tokenizer,
            sentiment_labels=sentiment_labels,
            category_labels=category_labels,
            device=dev,
            max_seq_len=max_seq_len,
            clause_split_mode=clause_split_mode,
        )

    @torch.no_grad()
    def predict_one(self, text: str, aspect: str) -> Dict[str, str]:
        """Predict sentiment + category for one (text, aspect) pair.

        Tokenisation follows SPC format: [CLS] text [SEP] aspect [SEP]
        LCF vec: binary 1.0 at aspect subword positions in segment A.

        When ``self.clause_split_mode`` is not ``"none"`` (read from meta.json),
        the sentence is narrowed to the clause containing the aspect before
        tokenisation — matching the preprocessing applied during training.
        """
        asp_stripped = aspect.strip()
        asp_cs = text.find(asp_stripped)
        asp_ce = asp_cs + len(asp_stripped) - 1 if asp_cs >= 0 else -1

        if self.clause_split_mode != "none" and asp_cs >= 0:
            text, asp_cs, asp_ce = extract_aspect_clause(
                text, asp_cs, asp_ce, mode=self.clause_split_mode
            )

        enc = self.tokenizer(
            text,
            aspect,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        input_ids      = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids)).squeeze(0)
        offsets        = enc["offset_mapping"].squeeze(0)

        # Build LCF vec: mark aspect tokens in segment A by char overlap
        lcf_vec = torch.zeros(input_ids.shape[-1], dtype=torch.float32)

        if asp_cs >= 0:
            for k in range(input_ids.shape[-1]):
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

        lcf_vec = lcf_vec.unsqueeze(0).to(self.device)

        out = self.model(input_ids, attention_mask, lcf_vec)
        sent_idx = int(out["sentiment_logits"].argmax(dim=-1).item())
        cat_idx  = int(out["aspect_cat_logits"].argmax(dim=-1).item())

        sentiment = (
            self.sentiment_labels[sent_idx]
            if sent_idx < len(self.sentiment_labels) else str(sent_idx)
        )
        category = (
            self.category_labels[cat_idx]
            if cat_idx < len(self.category_labels) else str(cat_idx)
        )
        return {"sentiment": sentiment, "category": category}

    def predict_batch(
        self,
        pairs: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """Predict for a list of {"text": ..., "aspect": ...} dicts."""
        return [self.predict_one(p["text"], p["aspect"]) for p in pairs]


# ─── Full pipeline ────────────────────────────────────────────────────────────

class PipelineInference:
    """Two-stage pipeline: ATE (T5) → APC (BERT multitask).

    When ``clause_split_mode`` is not ``"none"`` the sentence is split into
    clauses first; each clause is processed by ATE and APC independently, so
    aspects are always classified against the clause they belong to.

    Pipeline.predict(sentence) returns one dict per extracted aspect term:
        [
            {"aspect": "food",    "sentiment": "positive", "category": "FOOD"},
            {"aspect": "service", "sentiment": "negative",  "category": "SERVICE"},
        ]
    If ATE finds no aspects in any clause, returns [].
    """

    def __init__(
        self,
        ate_model: T5AspectExtractor,
        apc_model: APCPredictor,
        clause_split_mode: str = "none",
    ) -> None:
        self.ate               = ate_model
        self.apc               = apc_model
        self.clause_split_mode = clause_split_mode

    @classmethod
    def load(
        cls,
        ate_checkpoint: str | Path,
        apc_checkpoint_dir: str | Path,
        bert_name: str = "bert-base-uncased",
        device: Optional[torch.device] = None,
        max_seq_len: int = 128,
        clause_split_mode: str = "none",
    ) -> "PipelineInference":
        """Load both checkpoints.

        Args:
            ate_checkpoint      : directory saved by T5AspectExtractor.save()
            apc_checkpoint_dir  : directory containing best_model.pt (or model.pt)
                                  + meta.json — output of run_joint_experiments.py
                                  e.g. runs_joint/lcf_only/
            bert_name           : BERT variant used for APC training
            clause_split_mode   : ``"none"``, ``"rulebase"``, or ``"uos"``
        """
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[pipeline] Loading ATE model from {ate_checkpoint}")
        ate = T5AspectExtractor.from_pretrained(str(ate_checkpoint), device=dev)

        print(f"[pipeline] Loading APC model from {apc_checkpoint_dir}")
        apc = APCPredictor.load(
            checkpoint_dir=apc_checkpoint_dir,
            bert_name=bert_name,
            device=dev,
            max_seq_len=max_seq_len,
        )

        print(f"[pipeline] clause_split_mode={clause_split_mode!r}")
        return cls(ate_model=ate, apc_model=apc, clause_split_mode=clause_split_mode)

    # Pronouns that should be resolved to the most recent aspect from the
    # previous clause rather than being treated as a new aspect term.
    _RESOLVE_PRONOUNS: frozenset = frozenset({"it"})

    @staticmethod
    def _is_pronoun(aspect: str) -> bool:
        return aspect.strip().lower() in PipelineInference._RESOLVE_PRONOUNS

    def _predict_on_text(self, text: str) -> List[Dict[str, str]]:
        """Run ATE → APC on a single text (full sentence or one clause)."""
        aspects = predict_aspects(self.ate, text)
        return [{"aspect": a, **self.apc.predict_one(text, a)} for a in aspects]

    def predict(self, sentence: str) -> List[Dict[str, str]]:
        """Run the full pipeline on one sentence.

        If ``clause_split_mode`` is not ``"none"``: split sentence → each
        clause → ATE → APC.  Pronoun resolution: if ATE returns ``"it"`` for
        a clause, it is replaced by the last valid aspect term extracted from a
        previous clause.  If no previous aspect is available, the pronoun is
        skipped.

        Otherwise: ATE → APC on the full sentence (``"it"`` aspects are
        skipped because no clause context exists to resolve them).
        """
        if self.clause_split_mode != "none":
            results: List[Dict[str, str]] = []
            # Last non-pronoun aspect seen in any *previous* clause.
            prev_clause_aspect: Optional[str] = None

            for clause_text, _ in split_into_clauses(sentence, mode=self.clause_split_mode):
                clause_aspects = predict_aspects(self.ate, clause_text)
                # Track the last valid aspect within this clause separately so
                # that prev_clause_aspect only advances after the clause ends.
                current_clause_last: Optional[str] = None

                for aspect in clause_aspects:
                    if self._is_pronoun(aspect):
                        if prev_clause_aspect is None:
                            # Cannot resolve pronoun — no previous aspect.
                            continue
                        resolved = prev_clause_aspect
                    else:
                        resolved = aspect
                        current_clause_last = aspect

                    results.append({
                        "aspect": resolved,
                        **self.apc.predict_one(clause_text, resolved),
                    })

                if current_clause_last is not None:
                    prev_clause_aspect = current_clause_last

            return results

        # No clause splitting — skip unresolvable pronoun aspects.
        return [
            r for r in self._predict_on_text(sentence)
            if not self._is_pronoun(r["aspect"])
        ]

    def predict_batch(
        self,
        sentences: List[str],
        ate_batch_size: int = 16,
    ) -> List[List[Dict[str, str]]]:
        """Run pipeline on multiple sentences."""
        if self.clause_split_mode != "none":
            return [self.predict(s) for s in sentences]

        from src.inference import predict_batch as ate_predict_batch

        all_aspects = ate_predict_batch(
            self.ate, sentences, batch_size=ate_batch_size
        )

        all_results = []
        for sentence, aspects in zip(sentences, all_aspects):
            if not aspects:
                all_results.append([])
                continue
            row = [
                {"aspect": a, **self.apc.predict_one(sentence, a)}
                for a in aspects
                if not self._is_pronoun(a)
            ]
            all_results.append(row)
        return all_results


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="ATE -> APC pipeline inference (aspect term + category + sentiment)"
    )
    parser.add_argument("--ate-checkpoint", required=True,
                        help="Directory saved by T5AspectExtractor.save()")
    parser.add_argument("--apc-checkpoint-dir", required=True,
                        help="Directory containing best_model.pt + meta.json "
                             "(output of run_joint_experiments.py, e.g. runs_joint/lcf_only/)")
    parser.add_argument("--bert-name", default="bert-base-uncased")
    parser.add_argument("--sentence", default=None,
                        help="Single sentence to predict (interactive if omitted)")
    parser.add_argument(
        "--clause-split-mode",
        default="none",
        choices=["none", "rulebase", "uos"],
        help="Clause splitting mode: 'none' (full sentence), "
             "'rulebase' (regex boundaries), 'uos' (LLM via Ollama). "
             "Default: none",
    )
    args = parser.parse_args()

    pipe = PipelineInference.load(
        ate_checkpoint=args.ate_checkpoint,
        apc_checkpoint_dir=args.apc_checkpoint_dir,
        bert_name=args.bert_name,
        clause_split_mode=args.clause_split_mode,
    )

    if args.sentence:
        sentences = [args.sentence]
    else:
        print("Enter sentences (empty line to quit):")
        sentences = []
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                break
            sentences.append(line)

    for sentence in sentences:
        print(f"\nInput: {sentence}")
        results = pipe.predict(sentence)
        if not results:
            print("  (no aspects found)")
        for r in results:
            print(f"  aspect={r['aspect']!r:30s}  sentiment={r['sentiment']:10s}  category={r['category']}")


if __name__ == "__main__":
    main()
