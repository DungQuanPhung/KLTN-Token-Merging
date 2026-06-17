# -*- coding: utf-8 -*-
"""Wrapper chạy eval_joint_triplet.py với model từ folder Bert/ hoặc T5/.

Usage (from thesis_apc_baseline/):
    # Eval tất cả config T5 (mặc định)
    python experiments/eval_joint_triplet_run.py

    # Eval tất cả config Bert
    python experiments/eval_joint_triplet_run.py --model-type bert

    # Chỉ định thư mục chứa model tuỳ ý
    python experiments/eval_joint_triplet_run.py --model-type t5 --runs-dir path/to/dir

Cách hoạt động:
    Script này override 3 biến trong eval_joint_triplet trước khi chạy:
        MODEL_TYPE  → "bert" hoặc "t5"
        PRETRAINED  → "bert-base-uncased" hoặc "t5-base"
        RUNS_DIR    → ROOT/Bert hoặc ROOT/T5 (hoặc --runs-dir nếu cung cấp)
    Sau đó gọi hàm main() của eval_joint_triplet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Eval joint triplet với Bert/ hoặc T5/")
parser.add_argument(
    "--model-type", choices=["bert", "t5"], default="t5",
    help="Encoder: 'bert' → dùng Bert/, 't5' → dùng T5/ (default: t5)",
)
parser.add_argument(
    "--runs-dir", type=str, default=None,
    help="Override thư mục chứa các run (mặc định: ROOT/Bert hoặc ROOT/T5)",
)
parser.add_argument(
    "--ate-csv", type=str, default=None,
    help="Đường dẫn file ATE predictions CSV (mặc định: runs_ate/test_ate_predictions.csv)",
)
args = parser.parse_args()

MODEL_TYPE = args.model_type.lower()
_MODEL_CONFIGS = {
    "bert": "bert-base-uncased",
    "t5":   "t5-base",
}
PRETRAINED = _MODEL_CONFIGS[MODEL_TYPE]

if args.runs_dir:
    RUNS_DIR = Path(args.runs_dir)
else:
    RUNS_DIR = ROOT / ("Bert" if MODEL_TYPE == "bert" else "T5")

if not RUNS_DIR.is_dir():
    print(f"[ERROR] Không tìm thấy thư mục: {RUNS_DIR}")
    sys.exit(1)

ATE_CSV = Path(args.ate_csv) if args.ate_csv else ROOT / "runs_ate" / "test_ate_predictions.csv"

# ─── Align câu ATE CSV theo thứ tự dòng với test.apc ─────────────────────────
# Dòng i trong ATE CSV tương ứng với entry i trong test.apc.
# Thay sentence text trong CSV bằng sentence text chuẩn từ test.apc theo index.

import csv as _csv
import tempfile as _tmp
import os as _os
from dataset_utils import parse_apc_file as _parse_apc

_gold_entries = _parse_apc(str(ROOT / "dataset" / "test.apc"))
_gold_sents   = [e["text"] for e in _gold_entries]  # 1 entry per row in test.apc

_tmp_path = None
with open(ATE_CSV, newline="", encoding="utf-8") as _fin:
    _rows = list(_csv.DictReader(_fin))
    _fieldnames = list(_rows[0].keys()) if _rows else []

if len(_rows) != len(_gold_sents):
    print(f"[WARN] ATE CSV có {len(_rows)} dòng, test.apc có {len(_gold_sents)} entries — không align được theo index")
else:
    _aligned = [dict(r, sentence=_gold_sents[i]) for i, r in enumerate(_rows)]
    _changed  = sum(1 for o, n in zip(_rows, _aligned) if o["sentence"] != n["sentence"])
    _fd, _tmp_path = _tmp.mkstemp(suffix=".csv", prefix="ate_align_")
    _os.close(_fd)
    with open(_tmp_path, "w", newline="", encoding="utf-8") as _fout:
        _w = _csv.DictWriter(_fout, fieldnames=_fieldnames)
        _w.writeheader()
        _w.writerows(_aligned)
    ATE_CSV = Path(_tmp_path)
    print(f"[align] {_changed} câu đã được thay bằng sentence từ test.apc theo thứ tự dòng")

# ─── Patch eval_joint_triplet trước khi import main() ────────────────────────

import importlib.util as _ilu
import time as _time

_spec = _ilu.spec_from_file_location(
    "eval_joint_triplet",
    Path(__file__).parent / "eval_joint_triplet.py",
)
_eval_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_eval_mod)

_eval_mod.MODEL_TYPE = MODEL_TYPE
_eval_mod.PRETRAINED  = PRETRAINED
_eval_mod.RUNS_DIR    = RUNS_DIR
_eval_mod.ATE_CSV     = ATE_CSV
_eval_mod.OUT_CSV     = ROOT / "runs_ate" / f"eval_joint_triplet_{MODEL_TYPE.upper()}.csv"

# ─── Wrap predict_triplets để đo thời gian inference ─────────────────────────

_infer_times: dict = {}
_orig_predict = _eval_mod.predict_triplets

def _timed_predict(model, tokenizer, ate_preds, cat_map, cat_labels):
    _t0 = _time.perf_counter()
    result = _orig_predict(model, tokenizer, ate_preds, cat_map, cat_labels)
    _elapsed = _time.perf_counter() - _t0
    num_terms = sum(len(v) for v in ate_preds.values())
    num_sents = len(ate_preds)
    _infer_times[id(model)] = {
        "elapsed_sec": _elapsed,
        "num_sentences": num_sents,
        "num_terms": num_terms,
        "ms_per_sent": _elapsed / num_sents * 1000 if num_sents else 0,
        "ms_per_term": _elapsed / num_terms * 1000 if num_terms else 0,
    }
    print(f"  Inference time : {_elapsed:.3f}s  "
          f"({_elapsed/num_sents*1000:.1f} ms/sent, "
          f"{_elapsed/num_terms*1000:.1f} ms/term)  "
          f"[{num_sents} sents, {num_terms} terms]")
    return result

_eval_mod.predict_triplets = _timed_predict

# ─── Run ──────────────────────────────────────────────────────────────────────

print(f"Model type : {MODEL_TYPE.upper()}")
print(f"Pretrained : {PRETRAINED}")
print(f"Runs dir   : {RUNS_DIR}")
print(f"ATE CSV    : {ATE_CSV}")
print(f"Output CSV : {_eval_mod.OUT_CSV}")
print()

try:
    _eval_mod.main()
finally:
    if _tmp_path and _os.path.exists(_tmp_path):
        _os.remove(_tmp_path)
