# -*- coding: utf-8 -*-
"""Multi-task FAST_LCF_BERT: aspect_category + sentiment classification.

LCF semantics (LCF-ATEPC, Zeng et al. 2019):
    The input ``lcf_vec`` is a binary aspect indicator (1.0 at aspect subword
    positions in segment A, 0.0 elsewhere).  This model converts it into a
    Context Dynamic Weight (CDW) vector inside the forward pass:

        SRD_i = max(0, |i - aspect_center| - floor(aspect_len / 2))
        w_i   = 1                              if SRD_i ≤ α
                (L - (SRD_i - α)) / L          otherwise

    The CDW vector multiplies the BERT hidden states to produce the local
    stream; the global stream uses the unmasked hidden states.  This matches
    the two-stream design from the original paper.

Architecture:
    BERT  →  [ToMe]  →  split:
        local  = hidden * CDW(lcf_vec)   →  SA
        global = hidden
    →  cat(local, global)  →  Linear(2H→H)  →  Dropout
    →  SA  →  BertPooler  →  [sentiment head | aspect_cat head]

Constructor flags:
    use_lcf       (bool) – enable Local Context Focus (CDW-weighted local stream).
    use_tome      (bool) – apply Token Merging (ToMe) after BERT backbone.
    tome_resize   (bool) – True: interpolate merged tokens back to original L.
                           False: keep compact length L' ≤ L (real speedup).
    srd_threshold (int)  – CDW full-weight radius α (paper default = 5).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPooler

from thesis_apc_baseline.token_merging.tome_1d import ToMeSequenceMerger


def _compute_cdw_weights(
    aspect_indicator: torch.Tensor,
    attention_mask: torch.Tensor,
    srd_threshold: int = 5,
) -> torch.Tensor:
    """LCF-ATEPC CDW weights from a binary aspect indicator.

    Args:
        aspect_indicator : (B, L) float — 1.0 at aspect subword positions.
        attention_mask   : (B, L) float — 1.0 at valid (non-padding) positions.
        srd_threshold    : full-weight local-context radius α (paper default = 5).

    Returns:
        (B, L) float CDW weights in [0, 1] with padding positions zeroed.
    """
    B, L = aspect_indicator.shape
    device = aspect_indicator.device
    dtype = aspect_indicator.dtype
    positions = torch.arange(L, device=device, dtype=torch.float32)
    cdw = torch.zeros(B, L, device=device, dtype=dtype)

    for b in range(B):
        asp = torch.nonzero(aspect_indicator[b] > 0.5, as_tuple=False).squeeze(-1)
        if asp.numel() == 0:
            cdw[b] = torch.ones(L, device=device, dtype=dtype)
            continue
        a_start = asp.min().float()
        a_end   = asp.max().float()
        center  = (a_start + a_end) / 2.0
        half_aspect = torch.floor((a_end - a_start + 1.0) / 2.0)
        srd = torch.clamp(torch.abs(positions - center) - half_aspect, min=0.0)
        w = torch.where(
            srd <= float(srd_threshold),
            torch.ones_like(srd),
            (float(L) - (srd - float(srd_threshold))) / float(L),
        )
        cdw[b] = torch.clamp(w, min=0.0, max=1.0).to(dtype)

    return cdw * attention_mask.to(dtype)


def _compute_cdm_weights(
    aspect_indicator: torch.Tensor,
    attention_mask: torch.Tensor,
    srd_threshold: int = 5,
) -> torch.Tensor:
    """LCF-ATEPC CDM (Context Dynamic Mask) from a binary aspect indicator.

    Args:
        aspect_indicator : (B, L) float — 1.0 at aspect subword positions.
        attention_mask   : (B, L) float — 1.0 at valid (non-padding) positions.
        srd_threshold    : binary mask threshold α (default = 5).

    Returns:
        (B, L) float CDM binary mask {0, 1} with padding positions zeroed.
    """
    B, L = aspect_indicator.shape
    device = aspect_indicator.device
    dtype = aspect_indicator.dtype
    positions = torch.arange(L, device=device, dtype=torch.float32)
    cdm = torch.zeros(B, L, device=device, dtype=dtype)

    for b in range(B):
        asp = torch.nonzero(aspect_indicator[b] > 0.5, as_tuple=False).squeeze(-1)
        if asp.numel() == 0:
            cdm[b] = torch.ones(L, device=device, dtype=dtype)
            continue
        a_start = asp.min().float()
        a_end   = asp.max().float()
        center  = (a_start + a_end) / 2.0
        half_aspect = torch.floor((a_end - a_start + 1.0) / 2.0)
        srd = torch.clamp(torch.abs(positions - center) - half_aspect, min=0.0)
        w = torch.where(
            srd <= float(srd_threshold),
            torch.ones_like(srd),
            torch.zeros_like(srd),  # Binary mask: 0 outside threshold
        )
        cdm[b] = w.to(dtype)

    return cdm * attention_mask.to(dtype)


class _SALayer(nn.Module):
    """Multi-head self-attention with residual connection and LayerNorm."""

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x)
        return self.norm(x + self.drop(out))


class FastLcfBertMultiTask(nn.Module):
    """Multi-task BERT model: predicts sentiment AND aspect_category jointly.

    Args:
        bert           : HuggingFace BERT model (AutoModel / BertModel).
        num_sentiment  : number of sentiment classes (default 3).
        num_aspect_cat : number of aspect-category classes.
        use_lcf        : enable Local Context Focus masking.
        use_cdm        : if True, use CDM (binary mask); if False, use CDW (gradient).
        use_tome       : enable Token Merging after BERT backbone.
        dropout        : dropout probability.
        num_heads      : heads for self-attention layers.
        tome_merge_steps: bipartite merge rounds per sample.
    """

    def __init__(
        self,
        bert,
        num_sentiment: int = 3,
        num_aspect_cat: int = 10,
        use_lcf: bool = True,
        use_cdm: bool = False,
        use_tome: bool = False,
        tome_resize: bool = True,
        tome_merge_strategy: str = "bipartite",
        dropout: float = 0.1,
        num_heads: int = 8,
        tome_merge_steps: int = 2,
        srd_threshold: int = 5,
    ) -> None:
        super().__init__()
        self.bert = bert
        self._use_lcf = use_lcf
        self._use_cdm = use_cdm
        self._use_tome = use_tome
        self._tome_resize = tome_resize
        self._srd_threshold = srd_threshold

        H = bert.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.bert_SA = _SALayer(H, num_heads, dropout)   # after LCF masking
        self.linear2 = nn.Linear(H * 2, H)              # fuse lcf + global
        self.bert_SA_ = _SALayer(H, num_heads, dropout)  # before pooling
        self.bert_pooler = BertPooler(bert.config)

        self.dense_sentiment = nn.Linear(H, num_sentiment)
        self.dense_aspect_cat = nn.Linear(H, num_aspect_cat)

        if use_tome:
            self.tome = ToMeSequenceMerger(
                num_merge_steps=tome_merge_steps,
                protect_cls=True,
                protect_sep=True,
                protect_aspect=True,
                resize=tome_resize,
                merge_strategy=tome_merge_strategy,
            )

    def forward(
        self,
        input_ids: torch.Tensor,       # (B, L)
        attention_mask: torch.Tensor,  # (B, L)
        lcf_vec: torch.Tensor,         # (B, L) float – binary aspect indicator
    ) -> dict:
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = bert_out.last_hidden_state          # (B, L, H) — global stream

        # ── ToMe runs on the UNMASKED global hidden so the global branch is
        # not corrupted by LCF.  The aspect indicator is propagated through
        # merges (max over merged positions) and reused below to build CDW/CDM.
        if self._use_tome:
            _, hidden, lcf_vec, new_attn_mask = self.tome.forward_with_trace(
                hidden, lcf_vec, attention_mask.float()
            )
            if new_attn_mask is not None:
                attention_mask = new_attn_mask.long()

        # ── Local stream: mask hidden by CDW or CDM computed from aspect indicator.
        # When use_lcf=False the local stream falls back to the plain hidden states.
        if self._use_lcf:
            if self._use_cdm:
                # Use CDM (Context Dynamic Mask): binary mask {0, 1}
                cdm = _compute_cdm_weights(
                    lcf_vec, attention_mask.to(hidden.dtype),
                    srd_threshold=self._srd_threshold,
                )                                     # (B, L) binary
                lcf_features = hidden * cdm.unsqueeze(-1)
            else:
                # Use CDW (Context Dynamic Weight): gradient mask [0, 1]
                cdw = _compute_cdw_weights(
                    lcf_vec, attention_mask.to(hidden.dtype),
                    srd_threshold=self._srd_threshold,
                )                                     # (B, L) gradient
                lcf_features = hidden * cdw.unsqueeze(-1)
        else:
            lcf_features = hidden

        lcf_features = self.bert_SA(lcf_features)    # (B, L['], H)

        # ── Fuse local + global ────────────────────────────────────────────────
        cat_features = torch.cat([lcf_features, hidden], dim=-1)  # (B, L['], 2H)
        cat_features = self.linear2(cat_features)
        cat_features = self.dropout(cat_features)
        cat_features = self.bert_SA_(cat_features)

        pooled = self.bert_pooler(cat_features)

        return {
            "sentiment_logits":  self.dense_sentiment(pooled),
            "aspect_cat_logits": self.dense_aspect_cat(pooled),
        }
