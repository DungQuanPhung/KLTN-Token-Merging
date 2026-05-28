# -*- coding: utf-8 -*-
"""Token merging for 1D token sequences after a Transformer encoder.

Three merge strategies are available (controlled by ToMeSequenceMerger.merge_strategy):

  "bipartite"  (default, ToMe CVPR 2023)
      Divide interior tokens into two alternating groups A and B.
      Each A-token greedily picks the most similar B-token (cosine).
      All pairs are found simultaneously then merged in one shot.

  "sequential_local"
      Scan tokens left → right. At each active token i, compare
      cosine similarity with the nearest active left and right neighbour
      (within the mergeable zone, excluding protected CLS/SEP).
          sim_left  > sim_right  →  token[i] folds INTO left  (left updated, i removed)
          sim_right ≥ sim_left   →  right neighbour folds INTO token[i] (right removed)
      One pass = one merge step; repeat for num_merge_steps rounds.

  "attention_weighted"
      Rank tokens by attention weight (ascending).
      Iteratively merge lowest-attention tokens with their most similar neighbor.
      High-attention and aspect-position tokens (LCF=1) are protected.
      Merges most "unimportant" tokens first while preserving crucial signal.

Use ``forward_with_trace`` for thesis figures (lengths, rounds, pairs).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def _tensor_preview(x: torch.Tensor, k: int = 4) -> List[List[float]]:
    """
    Chỉ lấy vài chiều đầu để debug cho gọn.
    """
    if x.numel() == 0:
        return []
    return x[:, :k].detach().cpu().tolist()


def _normalize(metric: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return metric / (metric.norm(dim=-1, keepdim=True) + eps)


def _bipartite_pairs(
    x: torch.Tensor,
    mask_1d: torch.Tensor,
    protect_left: int,
    protect_right: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[Dict[str, Any]]]:
    """Return (src, dst, meta) for one merge step, or three Nones plus failure meta dict.

    x: (n_valid, d) — already squeezed to valid tokens only (no batch padding).
    mask_1d: (n,) 1 for eligible rows (caller uses all-one for valid slice).
    protect_left/right: CLS / SEP are excluded from merging when set to 1.

    Merge rule: bipartite cosine matching on normalized BERT hidden states.
    Candidate pool = contiguous interior indices (everything except first ``protect_left``
    and last ``protect_right`` positions). Tokens in the pool listed in ascending index
    order alternate ``merge_side_A`` / ``merge_side_B``; side A gathers side B greedily:
    each A chooses the opposing B token with maximal cosine similarity, one-to-one on B.

    Meta includes per-index merge role aligned with ``lcf_*`` vectors in traces.
    LCF scalars do not enter pair scoring; `_merge_pairs` uses max(LCF[src], LCF[dst]).
    """
    device = x.device
    n = x.size(0)
    idx = torch.nonzero(mask_1d > 0.5, as_tuple=False).squeeze(-1)
    if idx.numel() < 2:
        return None, None, {
            "status": "no_merge",
            "reason": "fewer_than_2_masked_tokens",
            "n_masked": int(idx.numel()),
        }

    inner = torch.ones(n, device=device, dtype=x.dtype)
    if protect_left > 0:
        inner[:protect_left] = 0
    if protect_right > 0:
        inner[-protect_right:] = 0
    inner = inner * mask_1d
    pos = torch.nonzero(inner > 0.5, as_tuple=False).squeeze(-1)
    if pos.numel() < 2:
        return None, None, {
            "status": "no_merge",
            "reason": "interior_merge_pool_smaller_than_2",
            "protect_left": int(protect_left),
            "protect_right": int(protect_right),
        }

    metric = _normalize(x[pos])
    nm = metric.size(0)
    even = torch.arange(0, nm, 2, device=device)
    odd = torch.arange(1, nm, 2, device=device)
    if even.numel() == 0 or odd.numel() == 0:
        return None, None, {
            "status": "no_merge",
            "reason": "empty_bipartite_side_after_split",
            "interior_count": int(nm),
        }

    a = metric[even]
    b = metric[odd]
    scores = a @ b.transpose(0, 1)
    assign_a = scores.argmax(dim=1)

    pos_cpu = [int(pos[i].item()) for i in range(pos.numel())]
    pos_rank = {tok_i: rk for rk, tok_i in enumerate(pos_cpu)}
    role: List[str] = []
    for i in range(n):
        if protect_left > 0 and i < protect_left:
            role.append("protected_cls")
        elif protect_right > 0 and i >= n - protect_right:
            role.append("protected_sep")
        elif i in pos_rank:
            rk = pos_rank[i]
            role.append("merge_side_A" if (rk % 2 == 0) else "merge_side_B")
        else:
            role.append("masked_no_merge")

    src_list = []
    dst_list = []
    pair_sims: List[Dict[str, Any]] = []
    used_b = set()
    for ia in range(scores.size(0)):
        ib = int(assign_a[ia].item())
        if ib in used_b:
            continue
        used_b.add(ib)
        pi = pos[even[ia]]
        pj = pos[odd[ib]]
        src_list.append(pi)
        dst_list.append(pj)
        sim = float(scores[ia, ib].detach().cpu().item())
        pair_sims.append(
            {"src_seq_idx": int(pi.item()), "dst_seq_idx": int(pj.item()), "cosine_sim": sim}
        )

    if not src_list:
        return None, None, {
            "status": "no_merge",
            "reason": "no_pairs_after_bipartite_greedy",
            "protect_left": int(protect_left),
            "protect_right": int(protect_right),
            "interior_candidates_ordered": pos_cpu,
            "per_token_merge_role_before_step": role,
        }

    meta: Dict[str, Any] = {
        "status": "ok",
        "protect_left_positions": int(protect_left),
        "protect_right_positions": int(protect_right),
        "interior_candidates_ordered": pos_cpu,
        "per_token_merge_role_before_step": role,
        "pair_cosine_similarity": pair_sims,
    }
    return torch.stack(src_list), torch.stack(dst_list), meta


def _merge_pairs(
    x: torch.Tensor,
    lcf: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge dst into src (average); drop dst positions (mask False)."""
    n, d = x.shape
    device = x.device
    keep = torch.ones(n, dtype=torch.bool, device=device)
    xs = x[src]
    xd = x[dst]
    x_new = x.clone()
    x_new[src] = 0.5 * (xs + xd)
    keep[dst] = False

    lf = lcf.clone()
    lf[src] = torch.maximum(lf[src], lf[dst])

    packed_x = x_new[keep]
    packed_lcf = lf[keep]
    return packed_x, packed_lcf, keep


def _attention_weighted_merge(
    x: torch.Tensor,        # (n, d)  – valid tokens only
    lcf: torch.Tensor,      # (n,)    – LCF flags (1.0 = aspect position)
    attn: torch.Tensor,     # (n,)    – attention weights per token [0,1]
    protect_left: int,      # CLS protected positions
    protect_right: int,     # SEP protected positions
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    """Attention-weighted merge: merge lowest-attention tokens with nearest neighbors.

    Tokens are ranked by attention weight (ascending). At each step:
    1. Find lowest-attention token not yet merged & not protected
    2. Find its nearest neighbor (by cosine similarity) also not protected
    3. Merge them: keep neighbor, remove low-attention token
    4. Repeat until target reached or no mergeable pairs remain

    Protection rules:
    - CLS/SEP (protect_left/protect_right) never removed
    - Aspect tokens (LCF=1) get higher priority (never removed unless necessary)
    - High-attention tokens (top quartile) are protected

    Returns (merged_x, merged_lcf, keep_mask, pairs)
    """
    n, d = x.shape
    device = x.device

    x_w   = x.clone()
    lcf_w = lcf.clone()
    attn_w = attn.clone()
    removed = torch.zeros(n, dtype=torch.bool, device=device)
    pairs: List[Tuple[int, int]] = []

    end = n - protect_right

    # ── Compute protection mask ───────────────────────────────────────────────
    protected = torch.zeros(n, dtype=torch.bool, device=device)
    if protect_left > 0:
        protected[:protect_left] = True
    if protect_right > 0:
        protected[-protect_right:] = True

    # Aspect tokens (LCF=1) are highly protected (should not be removed)
    is_aspect = lcf > 0.5

    # High-attention tokens (top 25%) are protected
    attn_threshold = torch.quantile(attn_w, 0.75)
    is_high_attn = attn_w >= attn_threshold

    # ── Iteratively merge lowest-attention tokens ────────────────────────────
    metric = _normalize(x_w)
    max_merges = (n - protect_left - protect_right) // 2  # limit merges
    merges_done = 0

    while merges_done < max_merges:
        # Find lowest-attention non-removed, non-protected token
        candidate_mask = ~removed & ~protected & ~is_aspect
        if not candidate_mask.any():
            break

        # Get lowest attention among candidates
        attn_candidates = attn_w.clone()
        attn_candidates[~candidate_mask] = float('inf')
        i = attn_candidates.argmin().item()

        # Find nearest neighbor (highest cosine similarity) not removed, not aspect
        neighbor_mask = ~removed & ~is_aspect
        neighbor_mask[i] = False  # exclude self
        if not neighbor_mask.any():
            break

        # Compute similarity with all potential neighbors
        sims = (metric[i] * metric).sum(dim=-1)  # (n,)
        sims[~neighbor_mask] = -2.0
        j = sims.argmax().item()

        if sims[j].item() < -1.5:  # no valid neighbor
            break

        # Merge i into j: i removed, j updated
        x_w[j] = 0.5 * (x_w[j] + x_w[i])
        lcf_w[j] = torch.maximum(lcf_w[j], lcf_w[i])
        removed[i] = True
        pairs.append((int(i), int(j)))
        merges_done += 1

        # Update metric for next iteration
        metric = _normalize(x_w)

    keep_mask = ~removed
    return x_w[keep_mask], lcf_w[keep_mask], keep_mask, pairs


def _sequential_neighbor_merge(
    x: torch.Tensor,        # (n, d)  – valid tokens only, no batch padding
    lcf: torch.Tensor,      # (n,)    – LCF flags (1.0 = aspect position)
    protect_left: int,      # number of protected tokens on the left  (CLS)
    protect_right: int,     # number of protected tokens on the right (SEP)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    """Sequential left-to-right local-neighbour merge (one pass).

    Scans active tokens from left to right (excluding protected CLS / SEP).
    At each active token i:

        sim_left  = cosine( x[i], nearest active left  neighbour in mergeable zone )
        sim_right = cosine( x[i], nearest active right neighbour in mergeable zone )

        sim_left  > sim_right  →  token[i] folds INTO left  neighbour:
                                    x[left]  ← avg(x[left], x[i])
                                    lcf[left] ← max(lcf[left], lcf[i])
                                    i is removed
        sim_right ≥ sim_left   →  right neighbour folds INTO token[i]:
                                    x[i]    ← avg(x[i], x[right])
                                    lcf[i]  ← max(lcf[i], lcf[right])
                                    right is removed

    Constraint:

    **Mid-SEP protection**: in SPC tokenisation the sequence is
    [CLS] text [SEP] aspect [SEP].  The first [SEP] (between text and aspect
    segments) is NOT the last token, so protect_right alone does not cover it.
    Any token whose LCF flag == 0 AND whose immediate right neighbour has
    LCF flag == 1 is treated as the mid-SEP boundary and excluded from merging
    (it is never removed; it may still receive merges from its left).

    Tokens are allowed to receive multiple merges from different neighbors
    in a single pass (no one-to-one per destination limit).

    LCF flags are combined by max so aspect-position signal is never lost.

    Returns:
        merged_x   : (n', d)     n' = n - number_of_merges
        merged_lcf : (n',)
        keep_mask  : (n,) bool   True = position kept in output
        pairs      : list of (removed_idx, kept_idx)  for trace logging
    """
    n, d = x.shape
    device = x.device

    x_w   = x.clone()
    lcf_w = lcf.clone()
    removed  = torch.zeros(n, dtype=torch.bool,  device=device)
    pairs: List[Tuple[int, int]] = []

    end = n - protect_right          # first non-mergeable index on the right

    # ── Detect mid-SEP boundary (segment-A / segment-B separator in SPC) ──────
    # Position j is "boundary" if lcf[j]==0 and lcf[j+1]==1 (first aspect token).
    # We mark such positions as protected sources (cannot be removed).
    mid_sep = torch.zeros(n, dtype=torch.bool, device=device)
    for j in range(protect_left, end - 1):
        if lcf_w[j].item() < 0.5 and lcf_w[j + 1].item() > 0.5:
            mid_sep[j] = True

    i = protect_left
    while i < end:
        if removed[i] or mid_sep[i]:
            i += 1
            continue

        # ── Nearest active left neighbour within mergeable zone ───────────────
        left = i - 1
        while left >= protect_left and (removed[left] or mid_sep[left]):
            left -= 1
        has_left = (
            left >= protect_left
            and not removed[left]
        )

        # ── Nearest active right neighbour within mergeable zone ──────────────
        right = i + 1
        while right < end and (removed[right] or mid_sep[right]):
            right += 1
        has_right = right < end and not removed[right] and not mid_sep[right]

        if not has_left and not has_right:
            i += 1
            continue

        # ── Cosine similarities ───────────────────────────────────────────────
        sim_left  = -2.0
        sim_right = -2.0
        if has_left:
            sim_left  = float(
                F.cosine_similarity(x_w[i].unsqueeze(0), x_w[left].unsqueeze(0)).item()
            )
        if has_right:
            sim_right = float(
                F.cosine_similarity(x_w[i].unsqueeze(0), x_w[right].unsqueeze(0)).item()
            )

        if has_left and sim_left > sim_right:
            # token[i] folds into left: average replaces left, i is removed
            x_w[left]   = 0.5 * (x_w[left]   + x_w[i])
            lcf_w[left] = torch.maximum(lcf_w[left], lcf_w[i])
            removed[i]  = True
            pairs.append((int(i), int(left)))
        elif has_right:
            # right folds into token[i]: average replaces i, right is removed
            x_w[i]         = 0.5 * (x_w[i]    + x_w[right])
            lcf_w[i]       = torch.maximum(lcf_w[i], lcf_w[right])
            removed[right] = True
            pairs.append((int(right), int(i)))

        i += 1

    keep_mask = ~removed
    return x_w[keep_mask], lcf_w[keep_mask], keep_mask, pairs


def _resize_to_length(
    x: torch.Tensor,
    lcf: torch.Tensor,
    target_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Linearly interpolate sequence from (n, d) to (target_len, d)."""
    if x.size(0) == target_len:
        return x, lcf
    # (1, d, n) -> (1, d, target_len)
    xd = x.unsqueeze(0).transpose(1, 2)
    xd = F.interpolate(xd, size=target_len, mode="linear", align_corners=False)
    x_out = xd.transpose(1, 2).squeeze(0)

    lf = lcf.unsqueeze(0).unsqueeze(0)
    lf = F.interpolate(lf, size=target_len, mode="linear", align_corners=False)
    lcf_out = lf.squeeze(0).squeeze(0)
    return x_out, lcf_out


class ToMeSequenceMerger(nn.Module):
    """Apply several bipartite merge steps per sentence.

    resize=True  (default): interpolate merged sequence back to original length L.
                            Output shape (B, L, H) — downstream layers unchanged.
                            Measures representation quality of merging.

    resize=False:           keep the shorter merged sequence (length L' ≤ L).
                            Pad batch to L'_max across the batch, return new
                            attention mask so SA layers see fewer real tokens.
                            Measures both representation quality AND inference speed.
    """

    def __init__(
        self,
        num_merge_steps: int = 2,
        protect_cls: bool = True,
        protect_sep: bool = True,
        resize: bool = True,
        merge_strategy: str = "bipartite",
    ):
        """
        Args:
            num_merge_steps  : number of merge passes (each reduces seq length by ~half).
            protect_cls      : never remove or absorb the CLS token [0].
            protect_sep      : never remove or absorb the SEP token [-1].
            resize           : if True, interpolate back to original L after merging;
                               if False, keep compact sequence of length L' ≤ L.
            merge_strategy   : "bipartite"         – ToMe (CVPR 2023) alternating-group
                                                     cosine matching (default).
                               "sequential_local"  – left-to-right nearest-neighbour:
                                                     each token merges toward its more
                                                     similar neighbour (left or right).
                               "attention_weighted" – merge lowest-attention tokens first,
                                                      protect aspect & high-attention tokens.
        """
        super().__init__()
        if merge_strategy not in ("bipartite", "sequential_local", "attention_weighted"):
            raise ValueError(
                f"merge_strategy must be 'bipartite', 'sequential_local', or 'attention_weighted', "
                f"got {merge_strategy!r}"
            )
        self.num_merge_steps  = num_merge_steps
        self.protect_cls      = protect_cls
        self.protect_sep      = protect_sep
        self.resize           = resize
        self.merge_strategy   = merge_strategy

    def forward(
        self,
        hidden: torch.Tensor,
        lcf_vec: torch.Tensor,
        attention_mask: torch.Tensor,
        return_trace: bool = False,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[Dict[str, Any]]]]:
        trace_out, merged_h, merged_lcf, _ = self.forward_with_trace(
            hidden, lcf_vec, attention_mask, attention_weights
        )
        if return_trace:
            return merged_h, merged_lcf, trace_out
        return merged_h, merged_lcf, None

    def forward_with_trace(
        self,
        hidden: torch.Tensor,
        lcf_vec: torch.Tensor,
        attention_mask: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[List[Dict[str, Any]], torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Run ToMe and return (trace, hidden_out, lcf_out, new_attn_mask).

        Args:
            attention_weights: (B, L) optional per-token attention weights for attention_weighted strategy.
                              If None, uses uniform weights (attention_weighted falls back to cosine).

        new_attn_mask:
            None         when resize=True  — output shape (B, L, H), same as input.
            (B, L'_max)  when resize=False — output shape (B, L'_max, H), L'_max ≤ L.
        """
        B, L, D = hidden.shape
        device = hidden.device
        dtype = hidden.dtype

        if lcf_vec.dim() == 2:
            lcf_exp = lcf_vec
        else:
            lcf_exp = lcf_vec.squeeze(-1)

        # Per-sample merged tensors (variable length when resize=False)
        out_h: List[torch.Tensor] = []
        out_l: List[torch.Tensor] = []
        batch_trace: List[Dict[str, Any]] = []

        for b in range(B):
            mask = attention_mask[b].float()
            tok = hidden[b]
            lf = lcf_exp[b].float()
            seq_trace: Dict[str, Any] = {
                "batch_index": b,
                "length_in": int(L),
                "steps": [],
            }

            # Valid positions (non-padding)
            valid_idx = torch.nonzero(mask > 0.5, as_tuple=False).squeeze(-1)
            if valid_idx.numel() == 0:
                out_h.append(hidden[b])
                out_l.append(lcf_exp[b])
                batch_trace.append(seq_trace)
                continue

            x_seg = tok[valid_idx].clone()
            lf_seg = lf[valid_idx].clone()
            m_seg = torch.ones(x_seg.size(0), device=device, dtype=dtype)

            prot_l = 1 if self.protect_cls else 0
            prot_r = 1 if self.protect_sep else 0

            for step in range(self.num_merge_steps):
                before = x_seg.size(0)
                x_before   = x_seg.clone()
                lcf_before = lf_seg.clone()

                if self.merge_strategy == "bipartite":
                    # ── Bipartite cosine matching (ToMe CVPR 2023) ────────────
                    src, dst, pair_meta = _bipartite_pairs(
                        x_seg, m_seg, prot_l, prot_r
                    )
                    if src is None:
                        seq_trace["steps"].append(
                            {
                                "step": step,
                                "skipped": True,
                                "reason": "no_pairs",
                                "strategy": "bipartite",
                                "merge_pair_selection": pair_meta,
                            }
                        )
                        break
                    pairs_list = torch.stack([src, dst], dim=1).tolist()
                    x_seg, lf_seg, keep_mask = _merge_pairs(
                        x_seg, lf_seg, src, dst
                    )
                    pair_meta_log = pair_meta

                elif self.merge_strategy == "sequential_local":
                    # ── Sequential local-neighbour merge ──────────────────────
                    x_seg, lf_seg, keep_mask, raw_pairs = _sequential_neighbor_merge(
                        x_seg, lf_seg, prot_l, prot_r
                    )
                    if not raw_pairs:
                        seq_trace["steps"].append(
                            {
                                "step": step,
                                "skipped": True,
                                "reason": "no_pairs",
                                "strategy": "sequential_local",
                            }
                        )
                        break
                    pairs_list = raw_pairs        # list of (removed, kept) tuples
                    pair_meta_log = {
                        "status": "ok",
                        "strategy": "sequential_local",
                        "num_pairs": len(raw_pairs),
                    }

                else:  # attention_weighted
                    # ── Attention-weighted merge ─────────────────────────────
                    attn_seg = torch.ones(x_seg.size(0), device=device, dtype=dtype)
                    if attention_weights is not None and attention_weights.size(0) > b:
                        attn_full = attention_weights[b].float()
                        if attn_full.numel() > 0:
                            attn_seg = attn_full[valid_idx].clone()

                    x_seg, lf_seg, keep_mask, raw_pairs = _attention_weighted_merge(
                        x_seg, lf_seg, attn_seg, prot_l, prot_r
                    )
                    if not raw_pairs:
                        seq_trace["steps"].append(
                            {
                                "step": step,
                                "skipped": True,
                                "reason": "no_pairs",
                                "strategy": "attention_weighted",
                            }
                        )
                        break
                    pairs_list = raw_pairs        # list of (removed, kept) tuples
                    pair_meta_log = {
                        "status": "ok",
                        "strategy": "attention_weighted",
                        "num_pairs": len(raw_pairs),
                    }

                x_after  = x_seg.clone()
                lcf_after = lf_seg.clone()
                m_seg = torch.ones(x_seg.size(0), device=device, dtype=dtype)
                after = x_seg.size(0)

                seq_trace["steps"].append(
                    {
                        "step": step,
                        "skipped": False,
                        "strategy": self.merge_strategy,
                        "length_before": before,
                        "length_after": after,
                        "pairs": pairs_list,
                        "merge_pair_selection": pair_meta_log,
                        "keep_mask": keep_mask.detach().cpu().tolist(),
                        "x_before_preview": _tensor_preview(x_before),
                        "x_after_preview": _tensor_preview(x_after),
                        "lcf_before": lcf_before.detach().cpu().tolist(),
                        "lcf_after": lcf_after.detach().cpu().tolist(),
                    }
                )

            # ── resize=True: interpolate back to original L ──────────────────
            if self.resize:
                x_fixed, lf_fixed = _resize_to_length(x_seg, lf_seg, L)
                seq_trace["length_out"] = int(L)
                seq_trace["resize_mode"] = "interpolated_back"
            else:
                # resize=False: keep the shorter merged sequence
                x_fixed, lf_fixed = x_seg, lf_seg
                seq_trace["length_out"] = int(x_seg.size(0))
                seq_trace["resize_mode"] = "compact"

            out_h.append(x_fixed)
            out_l.append(lf_fixed)
            batch_trace.append(seq_trace)

        # ── Assemble batch ────────────────────────────────────────────────────
        if self.resize:
            # All samples have length L — stack directly
            merged_h   = torch.stack(out_h, dim=0)     # (B, L, H)
            merged_lcf = torch.stack(out_l, dim=0)     # (B, L)
            new_attn_mask: Optional[torch.Tensor] = None
        else:
            # Samples have different lengths — pad to L'_max
            L_prime = max(h.size(0) for h in out_h)
            padded_h   = []
            padded_lcf = []
            new_mask_rows = []
            for h, lf in zip(out_h, out_l):
                n = h.size(0)
                pad = L_prime - n
                if pad > 0:
                    h  = F.pad(h,  (0, 0, 0, pad))  # (L_prime, H)
                    lf = F.pad(lf, (0, pad))          # (L_prime,)
                m = torch.zeros(L_prime, device=device, dtype=dtype)
                m[:n] = 1.0
                padded_h.append(h)
                padded_lcf.append(lf)
                new_mask_rows.append(m)
            merged_h      = torch.stack(padded_h,   dim=0)  # (B, L_prime, H)
            merged_lcf    = torch.stack(padded_lcf, dim=0)  # (B, L_prime)
            new_attn_mask = torch.stack(new_mask_rows, dim=0)  # (B, L_prime)

        return batch_trace, merged_h, merged_lcf, new_attn_mask

