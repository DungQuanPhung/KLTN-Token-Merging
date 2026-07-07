# Hướng dẫn biên dịch luận văn LaTeX (Overleaf)

Toàn bộ nội dung luận văn nằm trong thư mục `thesis/` này, tách biệt khỏi
phần code ở thư mục gốc của repo.

## Cấu trúc file

| File | Nội dung |
|------|----------|
| `main.tex` | File chính — upload làm main document (include tất cả chương bên dưới) |
| `mo_dau.tex` | Bìa, tóm tắt, abstract, chương 1 (giới thiệu, câu hỏi nghiên cứu) |
| `co_so_ly_thuyet.tex` | Chương 2 — Cơ sở lý thuyết |
| `phuong_phap_thuc_nghiem.tex` | Chương 3 — Phương pháp & thực nghiệm (include `uos_splitting.tex`) |
| `ket_qua_thuc_nghiem.tex` | Chương 4 — Kết quả thực nghiệm |
| `ket_luan.tex` | Chương 5 — Kết luận |
| `references.bib` | Tài liệu tham khảo |
| `figures/` | Hình ảnh sinh bởi `scripts/generate_thesis_figures.py` |

## Overleaf

1. Tạo project mới, upload toàn bộ file `.tex`/`.bib` trong thư mục này + thư mục `figures/`.
2. Compiler: **pdfLaTeX** + **Biber**.
3. Main document: `main.tex`.

## Tạo hình ảnh

Chạy từ thư mục gốc của repo (không phải từ `thesis/`):

```bash
pip install matplotlib pandas seaborn
python scripts/generate_thesis_figures.py
```

Script ghi trực tiếp vào `thesis/figures/`. Upload lại thư mục này lên Overleaf sau khi chạy.

Sơ đồ kiến trúc (TikZ) biên dịch trực tiếp trong `co_so_ly_thuyet.tex` và `phuong_phap_thuc_nghiem.tex`.

## Bổ sung kết quả (Bước 3–5)

- **Bước 3:** Điền số liệu GAS/paper vào `tab:step3_paper` trong `ket_qua_thuc_nghiem.tex`.
- **Bước 4:** Chạy `USE_SUPPLEMENT=True` trong `run_joint_experiments.py`, điền `tab:step4_supplement`.
- **Bước 5:** Chạy eval với `CLAUSE_SPLIT_MODE=uos`, điền `tab:step5_split`.

Thay `Nhận xét tôi sẽ bổ sung sau` bằng nhận xét của bạn.

## Số trang

Với `\onehalfspacing`, font 12pt, margins khóa luận (~80+ trang khi biên dịch đủ hình + phụ lục).
