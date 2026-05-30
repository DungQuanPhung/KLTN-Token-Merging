# -*- coding: utf-8 -*-
"""Multi-task FAST_LCF_BERT: aspect_category + sentiment classification.

Supports 4 configurations via constructor flags:
    use_lcf      (bool) – apply Local Context Focus masking after BERT
    use_tome     (bool) – apply Token Merging (ToMe) after BERT backbone
    tome_resize  (bool) – how ToMe handles sequence length:
        True  (default): merged tokens are interpolated BACK to original L.
                         Output shape stays (B, L, H) — no architecture change.
                         Use for: measuring representation quality of merging.
        False:           keep the compact merged sequence of length L' ≤ L.
                         SA layers run on shorter sequence → real speed gain.
                         Use for: measuring accuracy + speed trade-off.

Architecture:
    BERT  →  [ToMe]  →  [LCF mask]  →  SA
    →  cat(lcf_feat, global)  →  Linear(2H→H)  →  Dropout
    →  SA  →  BertPooler  →  [sentiment head | aspect_cat head]
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPooler

from thesis_apc_baseline.token_merging.tome_1d import ToMeSequenceMerger


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
        use_tome: bool = False,
        tome_resize: bool = True,
        tome_merge_strategy: str = "bipartite",
        dropout: float = 0.1,
        num_heads: int = 8,
        tome_merge_steps: int = 2,
    ) -> None:
        super().__init__()
        self.bert = bert
        self._use_lcf = use_lcf
        self._use_tome = use_tome
        self._tome_resize = tome_resize

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
        lcf_vec: torch.Tensor,         # (B, L) float – 1.0 at aspect positions
    ) -> dict:
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = bert_out.last_hidden_state  # (B, L, H)

        # ── Apply LCF first when ToMe is enabled, otherwise preserve original flow.
        if self._use_lcf:
            lcf_matrix = lcf_vec.unsqueeze(-1)    # (B, L, 1)
            hidden_lcf = hidden * lcf_matrix      # (B, L, H) – zero out non-aspect tokens
        else:
            hidden_lcf = hidden

        if self._use_tome:
            # forward_with_trace returns (trace, merged_h, merged_lcf, new_attn_mask)
            #   tome_resize=True : new_attn_mask is None,  hidden shape stays (B, L, H)
            #   tome_resize=False: new_attn_mask is (B, L'), hidden shape is (B, L', H)
            _, hidden, lcf_vec, new_attn_mask = self.tome.forward_with_trace(
                hidden_lcf, lcf_vec, attention_mask.float()
            )
            # When resize=False the attention mask must follow the shorter sequence
            if new_attn_mask is not None:
                attention_mask = new_attn_mask.long()

            # If LCF was already applied before merge, merged hidden is the local-context input.
            lcf_features = hidden
        else:
            lcf_features = hidden_lcf

        lcf_features = self.bert_SA(lcf_features)  # (B, L['], H)

        # ── Fuse local + global ────────────────────────────────────────────────
        cat_features = torch.cat([lcf_features, hidden], dim=-1)  # (B, L['], 2H)
        cat_features = self.linear2(cat_features)                  # (B, L['], H)
        cat_features = self.dropout(cat_features)
        cat_features = self.bert_SA_(cat_features)                 # (B, L['], H)

        pooled = self.bert_pooler(cat_features)  # (B, H) via CLS token

        return {
            "sentiment_logits": self.dense_sentiment(pooled),
            "aspect_cat_logits": self.dense_aspect_cat(pooled),
        }
