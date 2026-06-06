# Thesis APC Baseline

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
python pipeline_inference.py \
  --ate-checkpoint checkpoints/gas_t5_ate/best \
  --apc-checkpoint-dir runs_joint/lcf_bip_resize \
  --bert-name bert-base-uncased \
  --sentence "The food was amazing but the service was slow"
```

### Chế độ interactive (nhập nhiều câu)

```bash
python pipeline_inference.py \
  --ate-checkpoint checkpoints/gas_t5_ate/best \
  --apc-checkpoint-dir runs_joint/lcf_bip_resize \
  --bert-name bert-base-uncased
```

Sau đó nhập nhiều câu, nhấn Enter dòng trống để thoát.

### Ví dụ kết quả

```text
Input: The food was amazing but the service was slow
  aspect='food'     sentiment=positive   category=EXPERIENCE
  aspect='service'  sentiment=negative   category=SERVICE
```

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
