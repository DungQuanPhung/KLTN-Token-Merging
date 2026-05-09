# -*- coding: utf-8 -*-
"""Token merging (ToMe-style) for 1D token sequences after a Transformer encoder.

Adapted from "Token Merging: Your ViT But Faster" (CVPR 2023) — bipartite matching
on two token subsets, merge by averaging, pack valid tokens left, then resize back
to fixed length so downstream APC (LCF + SA) stays shape-compatible.

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
    """Apply several bipartite merge steps per sentence, then resize to original length."""

    def __init__(
        self,
        num_merge_steps: int = 2,
        protect_cls: bool = True,
        protect_sep: bool = True,
    ):
        super().__init__()
        self.num_merge_steps = num_merge_steps
        self.protect_cls = protect_cls
        self.protect_sep = protect_sep

    def forward(
        self,
        hidden: torch.Tensor,
        lcf_vec: torch.Tensor,
        attention_mask: torch.Tensor,
        return_trace: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[Dict[str, Any]]]]:
        trace_out, merged_h, merged_lcf = self.forward_with_trace(
            hidden, lcf_vec, attention_mask
        )
        if return_trace:
            return merged_h, merged_lcf, trace_out
        return merged_h, merged_lcf, None

    def forward_with_trace(
        self,
        hidden: torch.Tensor,
        lcf_vec: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[List[Dict[str, Any]], torch.Tensor, torch.Tensor]:
        """Returns (trace, hidden_out, lcf_out) with same shape as inputs."""
        B, L, D = hidden.shape
        device = hidden.device
        dtype = hidden.dtype

        if lcf_vec.dim() == 2:
            lcf_exp = lcf_vec
        else:
            lcf_exp = lcf_vec.squeeze(-1)

        out_h = []
        out_l = []
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
                src, dst, pair_meta = _bipartite_pairs(x_seg, m_seg, prot_l, prot_r)
                if src is None:
                    seq_trace["steps"].append(
                        {
                            "step": step,
                            "skipped": True,
                            "reason": "no_pairs",
                            "merge_pair_selection": pair_meta,
                        }
                    )
                    break

                pairs = torch.stack([src, dst], dim=1).tolist()

                # ===== DEBUG BEFORE =====
                x_before = x_seg.clone()
                lcf_before = lf_seg.clone()

                # merge
                x_seg, lf_seg, keep_mask = _merge_pairs(
                    x_seg,
                    lf_seg,
                    src,
                    dst,
                )

                # ===== DEBUG AFTER =====
                x_after = x_seg.clone()
                lcf_after = lf_seg.clone()

                m_seg = torch.ones(x_seg.size(0), device=device, dtype=dtype)
                after = x_seg.size(0)

                seq_trace["steps"].append(
                    {
                        "step": step,
                        "skipped": False,

                        # shape info
                        "length_before": before,
                        "length_after": after,

                        # merge pairs
                        "pairs": pairs,
                        "merge_pair_selection": pair_meta,

                        # removed positions
                        "keep_mask": keep_mask.detach().cpu().tolist(),

                        # embedding preview
                        "x_before_preview": _tensor_preview(x_before),
                        "x_after_preview": _tensor_preview(x_after),

                        # lcf preview
                        "lcf_before": lcf_before.detach().cpu().tolist(),
                        "lcf_after": lcf_after.detach().cpu().tolist(),
                    }
                )

            x_fixed, lf_fixed = _resize_to_length(x_seg, lf_seg, L)
            row_h = x_fixed
            row_l = lf_fixed
            out_h.append(row_h)
            out_l.append(row_l)
            seq_trace["length_after_resize"] = int(L)
            batch_trace.append(seq_trace)

        merged_h = torch.stack(out_h, dim=0)
        merged_lcf = torch.stack(out_l, dim=0)
        return batch_trace, merged_h, merged_lcf

