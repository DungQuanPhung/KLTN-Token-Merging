# Hướng dẫn biên dịch luận văn LaTeX (Overleaf)

## Cấu trúc file

| File | Nội dung |
|------|----------|
| `main.tex` | File chính — upload làm main document |
| `mo_dau.tex` | Bìa, tóm tắt, abstract |
| `chuong_1_gioi_thieu.tex` | Chương 1 |
| `co_so_ly_thuyet.tex` | Chương 2 |
| `phuong_phap_thuc_nghiem.tex` | Chương 3 |
| `ket_qua_thuc_nghiem.tex` | Chương 4 (5 bước Evaluation) |
| `ket_luan.tex` | Chương 5 |
| `phu_luc.tex` | Phụ lục |
| `references.bib` | Tài liệu tham khảo |

## Overleaf

1. Tạo project mới, upload toàn bộ các file trên + thư mục `figures/`.
2. Compiler: **pdfLaTeX** + **Biber**.
3. Main document: `main.tex`.

## Tạo hình ảnh

```bash
pip install matplotlib pandas seaborn
python scripts/generate_thesis_figures.py
```

Upload thư mục `figures/` lên Overleaf sau khi chạy script.

Sơ đồ kiến trúc (TikZ) biên dịch trực tiếp trong `co_so_ly_thuyet.tex` và `phuong_phap_thuc_nghiem.tex`.

## Bổ sung kết quả (Bước 3–5)

- **Bước 3:** Điền số liệu GAS/paper vào `tab:step3_paper` trong `ket_qua_thuc_nghiem.tex`.
- **Bước 4:** Chạy `USE_SUPPLEMENT=True` trong `run_joint_experiments.py`, điền `tab:step4_supplement`.
- **Bước 5:** Chạy eval với `CLAUSE_SPLIT_MODE=uos`, điền `tab:step5_split`.

Thay `Nhận xét tôi sẽ bổ sung sau` bằng nhận xét của bạn.

## Số trang

Với `\onehalfspacing`, font 12pt, margins khóa luận (~80+ trang khi biên dịch đủ hình + phụ lục).
