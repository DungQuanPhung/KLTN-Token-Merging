# -*- coding: utf-8 -*-
"""
Load a trained APC checkpoint and predict polarity for **one** train-style sample:

  - ``sentence``: raw sentence with exactly one ``$T$`` (same convention as ``*.apc`` line 1)
  - ``aspect``: aspect span text (same as ``*.apc`` line 2)

JSON output includes ``thesis_visualization`` with ``token_texts`` (valid subwords).
If the checkpoint uses ToMe (``use_tome``), the same block also includes
``tome_token_evolution``: per-step tokens + LCF, ``merge_pair_selection``
(cosine, roles CLS/SEP/side_A/B), and Vietnamese summaries of merge conditions.
With ``--show-steps``, a single preceding JSON document (``raw_data_through_pipeline_steps``)
summarizes the same text-level progression without embedding/LCF dumps.
Optional ``--output-text PATH`` saves the same final payloads as pretty JSON in a UTF-8 text file.

Run from repo root (parent of ``thesis_apc_baseline``), e.g. ``kltn``:

  python thesis_apc_baseline/experiments/infer_one.py ^
    --checkpoint thesis_apc_baseline/checkpoints/baseline_fast_lcf_bert/<run_folder> ^
    --sentence "The $T$ was cold ." ^
    --aspect "pizza"

Or pass one JSON object:

  python thesis_apc_baseline/experiments/infer_one.py ^
    --checkpoint <path> ^
    --json-item "{\"sentence\": \"The $T$ was cold .\", \"aspect\": \"pizza\"}"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thesis_apc_baseline.experiments.apc_inference import (
    infer_train_style_item,
    load_apc_sentiment_classifier,
    normalize_train_style_item,
    parse_item_json,
    thesis_visualization_for_train_style_item,
    train_style_to_pyabsa_text,
    to_json_serializable,
)


def _attach_tome_trace_hook(classifier: Any) -> Dict[str, Any]:
    """Hook ToMe forward_with_trace to capture real merge pairs during inference."""
    cache: Dict[str, Any] = {"trace": None, "hooked": False}
    try:
        model_ens = getattr(classifier, "model", None)
        models = getattr(model_ens, "models", None)
        base = models[0] if models and len(models) > 0 else None
        tome = getattr(base, "tome", None)
        if tome is None or not hasattr(tome, "forward_with_trace"):
            return cache

        original = tome.forward_with_trace

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            trace, merged_h, merged_lcf = original(*args, **kwargs)
            cache["trace"] = trace
            return trace, merged_h, merged_lcf

        tome.forward_with_trace = _wrapped  # type: ignore[assignment]
        cache["hooked"] = True
    except Exception:
        return cache
    return cache


def _merge_token_display(a: str, b: str) -> str:
    """Readable label after ToMe averaging (embedding space), for thesis diagrams."""
    return f"({a}∥{b})"


def _tome_merge_explanation_vi() -> Dict[str, str]:
    """Static reference: what can merge, how pairs are chosen, how LCF updates."""
    return {
        "pool_dieu_kien_vi": (
            "Token đầu chuỗi hợp lệ (CLS) và token cuối (SEP) được bảo vệ, không tham gia merge. "
            "Các vị trí còn lại trong phạm vi hợp lệ tạo thành 'pool nội bộ'; cần ít nhất 2 vị trí "
            "trong pool thì mới có thể merge."
        ),
        "chon_cap_dieu_kien_vi": (
            "Việc ghép cặp dựa trên hidden BERT (ToMe bipartite), không dùng giá trị LCF để chọn cặp. "
            "Sắp xếp các chỉ số trong pool theo thứ tự tăng dần; xen kẽ theo thứ tự đó: "
            "hạng 0,2,4,… thuộc nhánh A, hạng 1,3,5,… thuộc nhánh B. "
            "Chuẩn hóa L2 vector hidden theo từng token; mỗi token A chọn token B sao cho "
            "tích vô hướng (cosine) lớn nhất; mỗi B tối đa một lần ghép (duyệt A theo thứ tự)."
        ),
        "lcf_sau_merge_vi": (
            "LCF (trọng số local-context theo aspect) không quyết định cặp merge. "
            "Sau khi gộp, vị trí src giữ hidden = trung bình hai token; "
            "LCF tại src = max(LCF[src], LCF[dst]), vị trí dst bị loại bỏ."
        ),
    }


def _cosine_by_pair(
    pair_meta: Any, si: int, di: int
) -> Optional[float]:
    if not isinstance(pair_meta, dict):
        return None
    for row in pair_meta.get("pair_cosine_similarity") or []:
        if not isinstance(row, dict):
            continue
        if int(row.get("src_seq_idx", -1)) == si and int(row.get("dst_seq_idx", -1)) == di:
            return float(row["cosine_sim"])
    return None


def _rows_token_lcf_role(
    token_texts: List[str],
    lcf_vec: Any,
    roles: Any,
) -> List[Dict[str, Any]]:
    lb = lcf_vec if isinstance(lcf_vec, list) else []
    rl = roles if isinstance(roles, list) else []
    out: List[Dict[str, Any]] = []
    for i, tok in enumerate(token_texts):
        out.append(
            {
                "idx": i,
                "token": tok,
                "lcf_scalar": lb[i] if i < len(lb) else None,
                "merge_role_before_this_step": rl[i] if i < len(rl) else None,
            }
        )
    return out


def _enrich_tome_trace_with_token_texts(
    trace: Any,
    token_texts: List[str],
) -> Optional[List[Dict[str, Any]]]:
    """Map merge indices → subword strings; per step: before / after sequences and pair captions."""
    if not isinstance(trace, list) or not trace or not token_texts:
        return None

    out_batches: List[Dict[str, Any]] = []

    for seq in trace:
        bidx = int(seq.get("batch_index", 0))
        steps_in = seq.get("steps") or []
        labels = list(token_texts)
        enriched_steps: List[Dict[str, Any]] = []

        for s in steps_in:
            if s.get("skipped"):
                msel_skip = s.get("merge_pair_selection") or {}
                enriched_steps.append(
                    {
                        "step": s.get("step"),
                        "skipped": True,
                        "reason": s.get("reason"),
                        "merge_pair_selection": msel_skip,
                        "token_texts_state": list(labels),
                    }
                )
                continue

            want_before = int(s["length_before"])
            if len(labels) != want_before:
                if len(labels) > want_before:
                    labels = labels[:want_before]
                else:
                    labels = labels + ["<?>"] * (want_before - len(labels))

            token_texts_before = list(labels)
            pairs = s.get("pairs") or []
            msel = s.get("merge_pair_selection") or {}
            roles_before = msel.get("per_token_merge_role_before_step")
            lb = s.get("lcf_before") or []
            la = s.get("lcf_after") or []

            pairs_text: List[Dict[str, Any]] = []
            for p in pairs:
                if len(p) < 2:
                    continue
                si, di = int(p[0]), int(p[1])
                cs = _cosine_by_pair(msel, si, di)
                pairs_text.append(
                    {
                        "src_idx": si,
                        "dst_idx": di,
                        "src": token_texts_before[si] if 0 <= si < len(token_texts_before) else "?",
                        "dst": token_texts_before[di] if 0 <= di < len(token_texts_before) else "?",
                        "pair_cosine_sim": cs,
                        "merged_display": _merge_token_display(
                            token_texts_before[si] if 0 <= si < len(token_texts_before) else "?",
                            token_texts_before[di] if 0 <= di < len(token_texts_before) else "?",
                        ),
                    }
                )

            keep = s.get("keep_mask") or []
            labels_merged = list(token_texts_before)
            for p in pairs:
                if len(p) < 2:
                    continue
                si, di = int(p[0]), int(p[1])
                if 0 <= si < len(labels_merged) and 0 <= di < len(labels_merged):
                    labels_merged[si] = _merge_token_display(labels_merged[si], labels_merged[di])

            token_texts_after = [
                labels_merged[i]
                for i in range(len(labels_merged))
                if i < len(keep) and keep[i]
            ]
            labels = token_texts_after

            enriched_steps.append(
                {
                    "step": s.get("step"),
                    "skipped": False,
                    "length_before": s.get("length_before"),
                    "length_after": s.get("length_after"),
                    "merge_pair_selection": msel,
                    "tokens_lcf_merge_roles_before_merge": _rows_token_lcf_role(
                        token_texts_before, lb, roles_before
                    ),
                    "tokens_lcf_after_merge_packed": [
                        {"idx": i, "token": t, "lcf_scalar_after_step": la[i] if i < len(la) else None}
                        for i, t in enumerate(token_texts_after)
                    ],
                    "token_texts_before": token_texts_before,
                    "token_texts_after": token_texts_after,
                    "pairs": pairs,
                    "pairs_text": pairs_text,
                }
            )

        out_batches.append(
            {
                "batch_index": bidx,
                "length_in": seq.get("length_in"),
                "length_after_resize": seq.get("length_after_resize"),
                "token_texts_initial": list(token_texts),
                "merge_rules_reference_vi": _tome_merge_explanation_vi(),
                "steps": enriched_steps,
                "token_texts_after_merges_pre_resize": list(labels),
            }
        )

    return out_batches


def _estimate_tome_lengths(valid_tokens: int, merge_steps: int, protect_tokens: int = 2) -> List[Dict[str, int]]:
    """Estimate token count shrink per ToMe step (before resize back to fixed length)."""
    rows: List[Dict[str, int]] = []
    cur = max(0, int(valid_tokens))
    for step in range(int(merge_steps)):
        interior = max(0, cur - protect_tokens)
        merged_pairs = interior // 2
        nxt = cur - merged_pairs
        rows.append(
            {
                "step": step,
                "length_before": cur,
                "merged_pairs": merged_pairs,
                "length_after": nxt,
            }
        )
        if merged_pairs == 0:
            break
        cur = nxt
    return rows


def _pipeline_raw_steps_report(
    raw_payload: Dict[str, Any],
    item: Dict[str, str],
    classifier: Any,
    token_texts: List[str],
    trace_hooked: bool,
    tome_evolution: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Ordered stages: CLI/JSON payload → normalization → APC string → subwords → ToMe (text view)."""
    sentence_with_t, aspect = normalize_train_style_item(item)
    pyabsa_text = train_style_to_pyabsa_text(sentence_with_t, aspect)
    sentence_reconstructed = sentence_with_t.replace("$T$", aspect)

    use_tome = bool(getattr(classifier.config, "use_tome", False))
    merge_steps = int(getattr(classifier.config, "tome_merge_steps", 0) or 0)
    merge_sim: Optional[List[Dict[str, int]]] = None
    if use_tome and token_texts:
        merge_sim = _estimate_tome_lengths(len(token_texts), merge_steps, protect_tokens=2)

    note_resize = (
        f"After merging, the model resizes the shortened sequence back to length={len(token_texts)} "
        "before later layers."
        if use_tome and token_texts
        else None
    )

    evolution_payload: Optional[Any] = tome_evolution
    evolution_note = None
    if use_tome and not evolution_payload:
        if not trace_hooked:
            evolution_note = "ToMe trace hook did not attach (model structure?)."
        else:
            evolution_note = "Forward produced no usable trace."

    stages: List[Dict[str, Any]] = [
        {"stage": "raw_input_cli_or_json", "data": dict(raw_payload)},
        {
            "stage": "normalized_train_style_fields",
            "sentence_with_$T$_placeholder": sentence_with_t,
            "aspect": aspect,
        },
        {"stage": "reconstructed_plain_sentence", "sentence": sentence_reconstructed},
        {"stage": "pyabsa_predict_line", "text": pyabsa_text},
        {
            "stage": "tokenizer_valid_subwords_ordered",
            "token_texts": list(token_texts),
            "n_valid_tokens": len(token_texts),
        },
    ]

    tome_stage: Dict[str, Any] = {
        "stage": "tome_configuration",
        "use_tome": use_tome,
        "config_tome_merge_steps": merge_steps,
        "estimated_seq_lengths_before_resize": merge_sim,
        "resize_note": note_resize,
    }
    if use_tome:
        tome_stage["merge_pair_rules_reference_vi"] = _tome_merge_explanation_vi()
    if evolution_payload:
        tome_stage["live_token_evolution_per_batch"] = evolution_payload
    elif evolution_note:
        tome_stage["live_token_evolution_per_batch"] = None
        tome_stage["live_trace_note"] = evolution_note
    stages.append(tome_stage)

    return {"title": "raw_data_through_pipeline_steps", "stages": stages}


def _write_infer_result_text(out_path: Path, sections: List[Dict[str, Any]], *, encoding: str = "utf-8") -> None:
    """Save pretty JSON sections to one UTF-8 text file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    for blk in sections:
        title = str(blk.get("title") or "Section")
        body = blk.get("data")
        blob = json.dumps(to_json_serializable(body), indent=2, ensure_ascii=False)
        parts.append(f"== {title} ==\n{blob}")
    out_path.write_text("\n\n".join(parts).rstrip() + "\n", encoding=encoding)


def main() -> None:
    parser = argparse.ArgumentParser(description="APC inference: one train-style sentence + aspect.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Folder that contains PyABSA .state_dict / .config / .tokenizer (inner run directory).",
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--json-item",
        type=str,
        default=None,
        help='One JSON object, e.g. {"sentence": "The $T$ was cold.", "aspect": "pizza"}',
    )
    g.add_argument("--sentence", type=str, default=None, help="Sentence with a single $T$ placeholder.")
    parser.add_argument(
        "--aspect",
        type=str,
        default=None,
        help="Aspect text (required with --sentence).",
    )
    parser.add_argument(
        "--show-steps",
        action="store_true",
        help="Print one JSON block: raw input → normalized strings → tokenizer → ToMe evolution (text).",
    )
    parser.add_argument(
        "--output-text",
        type=str,
        default=None,
        metavar="PATH",
        help="Write final payloads as pretty UTF-8 text (same structured JSON as printed to stdout). "
        "If --show-steps was used, the file also starts with the pipeline JSON block.",
    )
    args = parser.parse_args()

    if args.sentence is not None:
        if not args.aspect:
            parser.error("--aspect is required when using --sentence")
        raw_payload: Dict[str, Any] = {"sentence": args.sentence, "aspect": args.aspect}
        item = dict(raw_payload)
    else:
        item = parse_item_json(args.json_item)
        raw_payload = dict(item) if isinstance(item, dict) else {"json_item": args.json_item}

    sentence_with_t, aspect = normalize_train_style_item(item)
    item = {"sentence": sentence_with_t, "aspect": aspect}

    clf = load_apc_sentiment_classifier(args.checkpoint, verbose=False)

    trace_cache: Dict[str, Any] = {}
    use_tome = bool(getattr(clf.config, "use_tome", False))
    if args.show_steps or use_tome:
        trace_cache = _attach_tome_trace_hook(clf)

    out = infer_train_style_item(clf, item, print_result=False)

    viz = thesis_visualization_for_train_style_item(clf, item)
    raw_trace = trace_cache.get("trace")
    enriched = _enrich_tome_trace_with_token_texts(raw_trace, viz.get("token_texts") or [])
    if enriched is not None:
        viz = {**viz, "tome_token_evolution": enriched}

    pipe_report: Optional[Dict[str, Any]] = None
    if args.show_steps:
        pipe_report = _pipeline_raw_steps_report(
            raw_payload,
            item,
            clf,
            viz.get("token_texts") or [],
            trace_hooked=bool(trace_cache.get("hooked")),
            tome_evolution=enriched,
        )
        print(json.dumps(to_json_serializable(pipe_report), indent=2, ensure_ascii=False))
    ser = to_json_serializable(out)
    if isinstance(ser, dict):
        ser_with_viz = {**ser, "thesis_visualization": viz}
    else:
        ser_with_viz = {"result": ser, "thesis_visualization": viz}

    resolved = to_json_serializable(ser_with_viz)

    print(json.dumps(resolved, indent=2, ensure_ascii=False))

    if args.output_text:
        text_sections: List[Dict[str, Any]] = []
        if pipe_report is not None:
            text_sections.append(
                {"title": "Tiến trình raw → tokenizer → ToMe (--show-steps)", "data": pipe_report}
            )
        text_sections.append({"title": "Kết quả inference (sentiment + thesis_visualization)", "data": resolved})
        out_txt = Path(args.output_text).expanduser()
        try:
            _write_infer_result_text(out_txt, text_sections)
        except OSError as e:
            print(f"[infer_one] Lỗi ghi file text: {e}", file=sys.stderr)
            raise SystemExit(2) from e
        print(f"[infer_one] Đã ghi file text: {out_txt.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
