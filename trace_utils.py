from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class TraceWriteResult:
    path: Path


def _safe_str(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return repr(x)


def _now_stamp() -> str:
    # Filesystem-friendly timestamp
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def default_trace_path(*, root: Optional[Path] = None, prefix: str = "raw_state") -> Path:
    """Return `logs/<prefix>_<timestamp>.txt` under repo root by default."""
    base = Path(root) if root is not None else Path.cwd()
    logs = base / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    return logs / f"{prefix}_{_now_stamp()}.txt"


def write_token_merging_trace_txt(
    path: Path,
    *,
    sentence: Optional[str] = None,
    aspect: Optional[str] = None,
    tokens: Optional[Sequence[str]] = None,
    input_ids: Any = None,
    attention_mask: Any = None,
    hidden_shape: Optional[Iterable[int]] = None,
    lcf_vec: Any = None,
    trace: Optional[List[Dict[str, Any]]] = None,
    merged_hidden_shape: Optional[Iterable[int]] = None,
    merged_lcf_shape: Optional[Iterable[int]] = None,
) -> TraceWriteResult:
    """Write a human-readable `.txt` snapshot of raw states across ToMe merging steps."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def w(line: str = "") -> None:
        f.write(line + "\n")

    with path.open("w", encoding="utf-8") as f:
        w("===== RAW STATE TRACE (TOKEN MERGING) =====")
        if sentence is not None:
            w(f"Sentence: {sentence}")
        if aspect is not None:
            w(f"Aspect  : {aspect}")
        w("")

        if tokens is not None:
            w("== TOKENS ==")
            for i, tok in enumerate(tokens):
                w(f"{i:3d}: {tok}")
            w("")

        if input_ids is not None:
            w("== INPUT_IDS ==")
            w(_safe_str(input_ids))
            w("")
        if attention_mask is not None:
            w("== ATTENTION_MASK ==")
            w(_safe_str(attention_mask))
            w("")
        if hidden_shape is not None:
            w("== BERT HIDDEN ==")
            w(f"shape: {tuple(hidden_shape)}")
            w("")
        if lcf_vec is not None:
            w("== LCF VECTOR ==")
            w(_safe_str(lcf_vec))
            w("")

        if trace is not None:
            w("== TOME TRACE ==")
            for seq in trace:
                w(f"- batch_index: {seq.get('batch_index')}")
                w(f"  length_in : {seq.get('length_in')}")
                steps = seq.get("steps") or []
                for st in steps:
                    if st.get("skipped"):
                        w(f"  step {st.get('step')}: skipped ({st.get('reason')})")
                        continue
                    w(
                        f"  step {st.get('step')}: "
                        f"len {st.get('length_before')} -> {st.get('length_after')}"
                    )
                    pairs = st.get("pairs") or []
                    for src, dst in pairs:
                        if tokens is not None and src < len(tokens) and dst < len(tokens):
                            w(f"    pair: {src}({tokens[src]}) <- {dst}({tokens[dst]})")
                        else:
                            w(f"    pair: {src} <- {dst}")
                w(f"  length_after_resize: {seq.get('length_after_resize')}")
            w("")

        if merged_hidden_shape is not None or merged_lcf_shape is not None:
            w("== AFTER TOME ==")
            if merged_hidden_shape is not None:
                w(f"hidden shape: {tuple(merged_hidden_shape)}")
            if merged_lcf_shape is not None:
                w(f"lcf shape   : {tuple(merged_lcf_shape)}")
            w("")

    return TraceWriteResult(path=path)

