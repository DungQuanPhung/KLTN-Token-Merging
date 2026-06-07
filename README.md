# Thesis

Dự án này chứa hai mô-đun chính:

1. **ATE** (Aspect Term Extraction) dùng mô hình T5 để trích xuất term khía cạnh từ câu.
2. **APC Joint** (Aspect Polarity Classification) dùng BERT multitask để dự đoán sentiment + category cho mỗi khía cạnh.

---

## Yêu cầu

- Python 3.8+ (hoặc tương đương môi trường conda/venv)
- Thư viện Python trong `requirements.txt`

Cài đặt:

```bash
python -m pip install -r requirements.txt
```

---

## Cấu trúc chính của repo

- `src/` : mã nguồn chung
  - `src/train.py` : CLI train T5 ATE
  - `src/dataset.py` : chuẩn bị dataloader cho ATE
  - `src/model.py` : định nghĩa mô hình T5AspectExtractor
  - `src/trainer.py` : training loop ATE
  - `src/inference.py` : hàm dự đoán ATE
- `experiments/` : joint APC multitask và các script thí nghiệm
  - `experiments/run_joint_experiments.py` : training APC multitask
- `pipeline_inference.py` : pipeline 2 bước ATE -> APC
- `dataset/` : dữ liệu train/dev/test và supplement
- `checkpoints/` : checkpoint ATE và các mô hình khác
- `runs_joint/` : output training APC joint

---

## 1. Train ATE (T5 Aspect Term Extraction)

Để train ATE, chạy từ thư mục gốc repo:

```bash
python src/train.py \
  --data-dir dataset \
  --output-dir checkpoints/gas_t5_ate \
  --model-name t5-base \
  --batch-size 16 \
  --learning-rate 3e-4 \
  --epochs 20 \
  --seed 42
```

`ATETrainer` sẽ tự động lưu checkpoint tốt nhất vào:

```text
checkpoints/gas_t5_ate/best/
```

Ngoài ra mô hình cuối cùng cũng được lưu vào:

```text
checkpoints/gas_t5_ate/last/
```

### Tham số chính

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--data-dir` | `dataset/` | Thư mục chứa `train.apc`, `dev.apc`, `test.apc` |
| `--output-dir` | `checkpoints/gas_t5_ate` | Nơi lưu checkpoint |
| `--model-name` | `t5-base` | Base model; có thể dùng `t5-small`, `t5-large`, ... |
| `--batch-size` | `16` | Batch size cho training |
| `--learning-rate` | `3e-4` | Learning rate |
| `--epochs` | `20` | Số epoch tối đa |
| `--max-input-length` | `128` | Độ dài input tối đa |
| `--max-target-length` | `64` | Độ dài target tối đa |
| `--seed` | `42` | Seed cho tái lập kết quả |
| `--num-workers` | `0` | Số worker cho DataLoader |

---

## 2. Train Joint APC Multitask (sentiment + category)

Chạy training joint từ thư mục gốc repo:

```bash
python experiments/run_joint_experiments.py
```

`run_joint_experiments.py` không dùng CLI args. Cấu hình được hardcode trong biến `CONFIGS` tại file:

```python
# experiments/run_joint_experiments.py — dòng 109–128
CONFIGS = [
    (True, True, True, True, "bipartite", "LCF+Bip (resize)", "lcf_bip_resize"),  # ← đang active
    # (True, True, True, False, "bipartite", "LCF+Bip (compact)", "lcf_bip_compact"),
    # (False, False, True, False, "bipartite", "Bip (compact)", "bip_compact"),
    # (True, True, False, True, "bipartite", "LCF only", "lcf_only"),
    # ...
]
```

Để train config khác, mở `experiments/run_joint_experiments.py` và bỏ comment dòng tương ứng trong `CONFIGS`.

### Output

Output của mỗi config sẽ được lưu trong thư mục con `runs_joint/<short_id>/`.

- `runs_joint/<short_id>/best_model.pt` : checkpoint tốt nhất
- `runs_joint/<short_id>/meta.json` : metadata về label và config
- `runs_joint/experiment_results_joint.txt` : bảng kết quả tổng hợp
- `runs_joint/experiment_results_joint.csv` : kết quả chi tiết từng config

### Các hyperparameter chính trong file

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `PRETRAINED_BERT` | `bert-base-uncased` | BERT backbone |
| `NUM_EPOCHS` | `15` | Số epoch tối đa |
| `PATIENCE` | `4` | Early stopping |
| `BATCH_SIZE` | `16` | Batch size |
| `LR` | `2e-5` | Learning rate |
| `MAX_SEQ_LEN` | `128` | Độ dài tối đa cho BERT |

---

## 3. Inference Pipeline (ATE → APC)

Sử dụng pipeline 2 bước để dự đoán aspect + sentiment + category.

### Dự đoán 1 câu

```bash
python pipeline_inference.py 
  --ate-checkpoint checkpoints/gas_t5_ate/best 
  --apc-checkpoint-dir runs_joint/lcf_bip_resize 
  --bert-name bert-base-uncased 
  --sentence "The food was amazing but the service was slow" --clause-split
```

### Chế độ interactive (nhập nhiều câu)

```bash
python pipeline_inference.py 
  --ate-checkpoint checkpoints/gas_t5_ate/best 
  --apc-checkpoint-dir runs_joint/lcf_bip_resize 
  --bert-name bert-base-uncased --clause-split
```

Sau đó nhập nhiều câu, nhấn Enter dòng trống để thoát.

### Ví dụ kết quả

```text
Input: The food was amazing but the service was slow
  aspect='food'     sentiment=positive   category=EXPERIENCE
  aspect='service'  sentiment=negative   category=SERVICE
```

---

## 4. Inference trực tiếp: Sentiment + Category cho một aspect term

Sử dụng script `infer_aspect_term.py` để dự đoán sentiment và category cho một câu + một aspect term cụ thể (không cần ATE).

### Cú pháp

```bash
python infer_aspect_term.py \
  --apc-checkpoint-dir runs_joint/lcf_bip_resize \
  --bert-name bert-base-uncased \
  --sentence "SENTENCE_CONTAINING_ASPECT" \
  --aspect "ASPECT_TERM"
```

### Ví dụ

```bash
python infer_aspect_term.py \
  --apc-checkpoint-dir runs_joint/lcf_bip_resize \
  --bert-name bert-base-uncased \
  --sentence "food is rich, a bit salty. The chef is not as polite as the restaurant service" \
  --aspect "food"
```

### Kết quả

```text
Input sentence: food is rich, a bit salty. The chef is not as polite as the restaurant service
Aspect term: food
Sentiment: positive
Category: FOOD
```

### Tham số

| Tham số | Bắt buộc | Mặc định | Ý nghĩa |
|---|---|---|---|
| `--apc-checkpoint-dir` | Có | — | Thư mục chứa `best_model.pt` + `meta.json` |
| `--bert-name` | Không | `bert-base-uncased` | HuggingFace BERT variant |
| `--sentence` | Có | — | Câu đầu vào chứa aspect term |
| `--aspect` | Có | — | Aspect term cần phân loại |
| `--max-seq-len` | Không | `128` | Độ dài tối đa cho BERT tokenization |

---

## Dataset

Thư mục `dataset/` chứa:

- `train.apc`
- `dev.apc`
- `test.apc`
- `train.xml.seg`, `dev.xml.seg`, `test.xml.seg`
- `supplement/negative.tsv`
- `supplement/neutral.tsv`

`src/dataset.py` và `dataset_utils.py` xử lý dữ liệu `.apc` và supplement cho ATE/APC.

---

## Ghi chú

- Mô hình ATE lưu tốt nhất vào `checkpoints/gas_t5_ate/best/` và checkpoint cuối cùng vào `checkpoints/gas_t5_ate/last/`.
- Khi train joint APC, `runs_joint/` chứa model và `meta.json` để `pipeline_inference.py` sử dụng lại.
- Nếu dùng GPU, các script sẽ tự động kích hoạt CUDA khi có sẵn.

---

## Web UI & Full-pipeline inference (ATE → APC)


Thư mục `server/` chứa một FastAPI API wrapper để gọi `pipeline_inference.PipelineInference` và trả về kết quả JSON (mỗi aspect gồm 3 label: `aspect`, `sentiment`, `category`). Thư mục `frontend/` là một React app tối thiểu để test nhanh: nhập một câu hoặc upload một file (`.txt` hoặc `.docx`, mỗi dòng một câu).

1) Cài đặt dependencies (từ root project; file `requirements.txt` đã bao gồm `torch`/`transformers`):

```bash
python -m pip install -r requirements.txt
pip install fastapi uvicorn python-docx
```

2) Cấu hình biến môi trường cho backend (hoặc sửa trực tiếp trong `server/app.py`) và chạy bằng `uvicorn`:

> Lưu ý: khi copy vào terminal **không** kèm phần chú thích trên cùng dòng. Ví dụ: `set CLAUSE_SPLIT=1  # comment` sẽ lưu cả phần `# comment` vào biến và gây lỗi.

```bash
# Windows (cmd)
set ATE_CHECKPOINT=checkpoints/gas_t5_ate/best && set APC_CHECKPOINT_DIR=runs_joint/lcf_bip_resize && set BERT_NAME=bert-base-uncased && set CLAUSE_SPLIT=1 && uvicorn server.app:app --host 0.0.0.0 --port 5000

# PowerShell
uvicorn server.app:app --host 0.0.0.0 --port 5000
```

3) Chạy frontend (mở terminal trong `frontend/`):

```bash
cd frontend
npm install
npm run dev
```

4) Sử dụng UI

- Chọn `Single sentence` để nhập trực tiếp một câu rồi nhấn `Predict`.
- Chọn `Upload file` để gửi `.txt` hoặc `.docx` (mỗi dòng 1 câu). Kết quả trả về là một mảng cho mỗi dòng; mỗi phần tử chứa danh sách các aspect với 3 label: `aspect`, `sentiment`, `category`.

5) Curl ví dụ (JSON single sentence):

```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"text": "The food was amazing but the service was slow"}'
```

6) Curl ví dụ (upload file):

```bash
curl -X POST http://localhost:5000/batch_predict -F file=@sentences.txt
```

Ghi chú: backend sẽ load mô hình khi khởi động — việc này có thể mất vài phút nếu lần đầu tải trọng số lớn.

---
Ghi chú: backend sẽ load mô hình khi khởi động — việc này có thể mất vài phút nếu lần đầu tải trọng số lớn.

---

---

