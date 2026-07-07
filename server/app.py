from __future__ import annotations

import os
import io
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from docx import Document

from common.pipeline_inference import PipelineInference


def read_txt_lines(file_bytes: bytes) -> List[str]:
    text = file_bytes.decode("utf-8", errors="ignore")
    return [line.strip() for line in text.splitlines() if line.strip()]


def read_docx_lines(file_bytes: bytes) -> List[str]:
    bio = io.BytesIO(file_bytes)
    doc = Document(bio)
    lines: List[str] = []
    for p in doc.paragraphs:
        text = p.text.strip()
        if text:
            for line in text.splitlines():
                s = line.strip()
                if s:
                    lines.append(s)
    return lines


class TextRequest(BaseModel):
    text: str


class TextsRequest(BaseModel):
    texts: List[str]


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load config from env
ATE_CHECKPOINT = os.environ.get("ATE_CHECKPOINT", "checkpoints/gas_t5_ate/best")
APC_CHECKPOINT_DIR = os.environ.get("APC_CHECKPOINT_DIR", "runs_joint/lcf_bip_resize")
BERT_NAME = os.environ.get("BERT_NAME", "bert-base-uncased")
def _get_env_str(name: str, default: str) -> str:
    """Read env var, strip surrounding quotes/whitespace and drop inline comments after '#'."""
    val = os.environ.get(name, default)
    if val is None:
        return default
    # drop inline comment and trim
    val = str(val).split("#", 1)[0].strip()
    # drop surrounding quotes if any
    if len(val) >= 2 and ((val[0] == val[-1] == '"') or (val[0] == val[-1] == "'")):
        val = val[1:-1]
    return val


def _get_env_int(name: str, default: int) -> int:
    s = _get_env_str(name, str(default))
    try:
        return int(s)
    except Exception:
        return default


ATE_CHECKPOINT = _get_env_str("ATE_CHECKPOINT", "checkpoints/gas_t5_ate/best")
APC_CHECKPOINT_DIR = _get_env_str("APC_CHECKPOINT_DIR", "runs_joint/lcf_scm_cdm_resize")
BERT_NAME = _get_env_str("BERT_NAME", "bert-base-uncased")
CLAUSE_SPLIT_MODE = _get_env_str("CLAUSE_SPLIT_MODE", "none")  # "none" | "rulebase" | "uos"


print(f"[server] Loading pipeline: ATE={ATE_CHECKPOINT} APC={APC_CHECKPOINT_DIR} clause_split_mode={CLAUSE_SPLIT_MODE!r}")
PIPELINE = PipelineInference.load(
    ate_checkpoint=ATE_CHECKPOINT,
    apc_checkpoint_dir=APC_CHECKPOINT_DIR,
    bert_name=BERT_NAME,
    clause_split_mode=CLAUSE_SPLIT_MODE,
)


@app.post("/predict")
async def predict(req: TextRequest):
    text = req.text
    if not text:
        return JSONResponse({"success": False, "error": "No text provided", "data": None}, status_code=400)
    try:
        results = await run_in_threadpool(PIPELINE.predict, text)
        return JSONResponse({"success": True, "data": results, "error": None})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e), "data": None}, status_code=500)


@app.post("/batch_predict")
async def batch_predict(texts_req: Optional[TextsRequest] = None, file: Optional[UploadFile] = File(None)):
    # If JSON body provided
    if texts_req is not None:
        texts = texts_req.texts
        if not isinstance(texts, list):
            return JSONResponse({"success": False, "error": "`texts` must be a list", "data": None}, status_code=400)
    else:
        if file is None:
            return JSONResponse({"success": False, "error": "No file provided", "data": None}, status_code=400)
        filename = file.filename.lower()
        content = await file.read()
        if filename.endswith(".txt"):
            texts = read_txt_lines(content)
        elif filename.endswith(".docx"):
            texts = read_docx_lines(content)
        else:
            return JSONResponse({"success": False, "error": "Unsupported file type. Use .txt or .docx", "data": None}, status_code=400)
    try:
        all_results = await run_in_threadpool(PIPELINE.predict_batch, texts)
        data = [
            {"text": text, "aspects": results}
            for text, results in zip(texts, all_results)
        ]
        return JSONResponse({"success": True, "data": data, "error": None})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e), "data": None}, status_code=500)


@app.get("/aspect-stats")
async def get_aspect_stats():
    """Get frequency of aspect terms from dataset"""
    try:
        aspect_freq = {}
        
        # Read from train.apc file
        dataset_file = "dataset/train.apc"
        if not os.path.exists(dataset_file):
            return JSONResponse({"success": False, "error": "Dataset file not found", "data": None}, status_code=404)
        
        with open(dataset_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Format: sentence, then aspect, category, sentiment
            # Each aspect group is 4 lines
            if i + 3 < len(lines):
                sentence_line = lines[i].strip()
                aspect_line = lines[i + 1].strip()
                category_line = lines[i + 2].strip()
                sentiment_line = lines[i + 3].strip()
                
                # Check if this looks like a valid aspect group
                if (sentence_line and aspect_line and 
                    category_line in ["SERVICE", "FOOD", "AMBIANCE", "PRICE", "QUALITY", "GENERAL"] and
                    sentiment_line in ["Positive", "Negative", "Neutral"]):
                    # Count the aspect term
                    aspect_term = aspect_line.lower()
                    aspect_freq[aspect_term] = aspect_freq.get(aspect_term, 0) + 1
                    i += 4
                else:
                    i += 1
            else:
                i += 1
        
        # Sort by frequency and return top 50
        sorted_aspects = sorted(aspect_freq.items(), key=lambda x: x[1], reverse=True)[:50]
        data = [{"term": term, "count": count} for term, count in sorted_aspects]
        
        return JSONResponse({"success": True, "data": data, "error": None})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e), "data": None}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
