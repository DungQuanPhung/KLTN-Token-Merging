# -*- coding: utf-8 -*-
"""Training loop for the single-stage GAS T5 model.

The model generates "(aspect, CATEGORY, sentiment)" for each aspect in the
sentence.  Both main .apc data and supplement TSV data are mixed into the
training loader; supplement rows use the NULL category placeholder.

Early stopping monitors **dev joint F1** (all three labels correct) — the
most demanding signal that captures aspect extraction, category prediction,
and sentiment classification simultaneously.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gas.model import GasT5Model
from gas.metrics import evaluate_all, format_metric


class GasTrainer:
    """Fine-tune GasT5Model with standard seq2seq CrossEntropyLoss.

    Supplement data (NULL category) is already mixed into ``train_loader``
    by ``gas.dataset.create_gas_dataloaders``; no special masking needed
    at the loss level.

    Parameters
    ----------
    model         : GasT5Model instance (untrained)
    train_loader  : includes main + supplement samples
    dev_loader    : main samples only (not used for generative eval — kept for API compat)
    test_loader   : main samples only
    dev_records   : raw main record dicts for generative evaluation on dev set
    test_records  : raw main record dicts for generative evaluation on test set
    patience      : early-stopping patience in epochs (on dev joint-F1)
    """

    def __init__(
        self,
        model: GasT5Model,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        *,
        learning_rate: float = 3e-4,
        num_epochs: int = 20,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        patience: int = 5,
        output_dir: Optional[str | Path] = None,
        dev_records:  Optional[Sequence[Dict]] = None,
        test_records: Optional[Sequence[Dict]] = None,
    ) -> None:
        self.model         = model
        self.train_loader  = train_loader
        self.dev_loader    = dev_loader
        self.test_loader   = test_loader
        self.num_epochs    = num_epochs
        self.max_grad_norm = max_grad_norm
        self.patience      = patience
        self.output_dir    = Path(output_dir) if output_dir else None
        self.dev_records   = list(dev_records)  if dev_records  else None
        self.test_records  = list(test_records) if test_records else None

        self.device    = model.device
        self.optimizer = torch.optim.AdamW(
            model.model.parameters(), lr=learning_rate
        )
        total_steps  = max(1, len(train_loader) * num_epochs)
        warmup_steps = int(total_steps * warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        self.use_amp = torch.cuda.is_available()
        self.scaler  = amp.GradScaler(enabled=self.use_amp)

        self.best_dev_f1         = -1.0
        self.best_checkpoint_dir: Optional[Path] = None
        self.history: List[Dict] = []

    # ── Training ──────────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> float:
        self.model.model.train()
        running_loss = 0.0
        n_batches    = 0

        for batch in self.train_loader:
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["labels"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            with amp.autocast(enabled=self.use_amp):
                loss = self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                ).loss

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.model.parameters(), self.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            running_loss += float(loss.item())
            n_batches    += 1

        avg_loss = running_loss / max(n_batches, 1)
        print(f"[epoch {epoch + 1}/{self.num_epochs}] train_loss={avg_loss:.4f}")
        return avg_loss

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _eval_records(
        self,
        model: GasT5Model,
        records: Sequence[Dict],
        split_name: str,
    ) -> Dict[str, Dict]:
        """Run single-stage GAS inference and compute all 4 metrics."""
        pred_triples_list, gold_triples_list = model.predict_for_records(records)

        results = evaluate_all(pred_triples_list, gold_triples_list)

        # Print one line per metric
        label_map = {
            "aspect_term": "ATE",
            "category":    "Category",
            "sentiment":   "Sentiment",
            "joint":       "Joint",
        }
        for key, lbl in label_map.items():
            print(f"[{split_name}] {format_metric(lbl, results[key])}")

        return results

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self) -> Dict:
        """Run the full training loop.

        Returns
        -------
        dict with: best_dev_f1, best_checkpoint, test_metrics, wall_time_sec, history
        """
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        t0         = time.perf_counter()
        no_improve = 0

        for epoch in range(self.num_epochs):
            train_loss = self._train_epoch(epoch)

            dev_metrics: Dict = {}
            if self.dev_records is not None:
                dev_metrics = self._eval_records(self.model, self.dev_records, "dev")

            # Monitor joint F1: all 3 labels must be correct
            dev_joint_f1 = float(
                dev_metrics.get("joint", {}).get("f1", -1.0)
            )

            self.history.append(
                {"epoch": epoch + 1, "train_loss": train_loss, "dev": dev_metrics}
            )

            if self.output_dir is not None and dev_joint_f1 >= self.best_dev_f1:
                self.best_dev_f1 = dev_joint_f1
                best_dir = self.output_dir / "best"
                self.model.save(str(best_dir))
                self.best_checkpoint_dir = best_dir
                no_improve = 0
                print(
                    f"  -> Saved best checkpoint "
                    f"(dev joint-F1={dev_joint_f1:.4f}) to {best_dir}"
                )
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(
                        f"  -> Early stop at epoch {epoch + 1} "
                        f"(patience={self.patience})"
                    )
                    break

        # Save final checkpoint
        if self.output_dir is not None:
            last_dir = self.output_dir / "last"
            self.model.save(str(last_dir))
            print(f"  -> Saved last checkpoint to {last_dir}")

        # Evaluate on test using best (or last) checkpoint
        test_metrics: Dict = {}
        if self.test_records is not None:
            ckpt = self.best_checkpoint_dir or (
                self.output_dir / "last" if self.output_dir else None
            )
            eval_model = (
                GasT5Model.from_pretrained(str(ckpt), device=self.device)
                if ckpt is not None and Path(ckpt).is_dir()
                else self.model
            )
            test_metrics = self._eval_records(eval_model, self.test_records, "test")

        elapsed = time.perf_counter() - t0
        return {
            "history":         self.history,
            "best_dev_f1":     self.best_dev_f1,
            "best_checkpoint": (
                str(self.best_checkpoint_dir) if self.best_checkpoint_dir else None
            ),
            "test_metrics":    test_metrics,
            "wall_time_sec":   round(elapsed, 3),
        }
