# -*- coding: utf-8 -*-
"""Visualize Sequential Cosine Merging (SCM) algorithm step by step.

Hành vi thực tế của chiến lược sequential_cosine (trước gọi là attention_weighted):
  - Mỗi bước: chọn token TRÁI NHẤT chưa bị xóa và không bị bảo vệ
  - Tìm token có cosine similarity CAO NHẤT với nó trong toàn chuỗi
  - Merge: token trái nhất bị xóa, token đích được cập nhật (trung bình)
  - Lặp tối đa max_merges = floor((n - protect_left - protect_right) / 2) lần

Chạy:
    python scripts/visualize_scm.py
Output:
    thesis/figures/scm_steps.pdf  và  thesis/figures/scm_steps.png
"""

import math
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False

# ── Cấu hình ví dụ ────────────────────────────────────────────────────────────
# Câu: [CLS] "chất" "lượng" "pin" "rất" "tốt" [SEP] "pin" [SEP]
# Dạng SPC: segment A = text, segment B = aspect term
TOKENS   = ["[CLS]", "chất", "lượng", "pin", "rất", "tốt", "[SEP]", "pin", "[SEP]"]
IS_ASPECT = [False,   False,   False,  False,  False,  False,  False,  True,  False]
PROTECT_L = 1   # CLS
PROTECT_R = 1   # SEP cuối

N = len(TOKENS)

# ── Cosine similarity giả lập (để minh hoạ thuật toán) ───────────────────────
# Ma trận similarity: giá trị được chọn sao cho mỗi bước merge hợp lý về ngữ nghĩa
#   Bước 1: token 1 ("chất") → giống nhất với token 2 ("lượng")  sim=0.82
#   Bước 2: token 3 ("pin")  → giống nhất với token 7 ("pin")    sim=0.95
#   Bước 3: token 4 ("rất")  → giống nhất với token 5 ("tốt")    sim=0.71
np.random.seed(0)
SIM = np.full((N, N), 0.3)
np.fill_diagonal(SIM, 1.0)
# Đặt similarity cao cho các cặp sẽ merge
SIM[1, 2] = SIM[2, 1] = 0.82   # chất ↔ lượng
SIM[3, 7] = SIM[7, 3] = 0.95   # pin  ↔ pin(aspect) — nhưng aspect bị protect
SIM[3, 4] = SIM[4, 3] = 0.68
SIM[4, 5] = SIM[5, 4] = 0.71
SIM[3, 5] = SIM[5, 3] = 0.60

# ── Simulate thuật toán SCM ───────────────────────────────────────────────────
def simulate_scm(tokens, is_aspect, protect_l, protect_r, sim_matrix):
    n = len(tokens)
    removed  = [False] * n
    protected = [False] * n
    for i in range(protect_l):
        protected[i] = True
    for i in range(n - protect_r, n):
        protected[i] = True

    max_merges = (n - protect_l - protect_r) // 2
    steps = []   # list of (state_before, i, j, sim_val)

    for _ in range(max_merges):
        # Tìm token trái nhất không bị xóa, không bị protect, không phải aspect
        i = None
        for t in range(n):
            if not removed[t] and not protected[t] and not is_aspect[t]:
                i = t
                break
        if i is None:
            break

        # Tìm neighbor có sim cao nhất (không bị xóa, không phải aspect, khác i)
        best_j, best_sim = -1, -2.0
        for t in range(n):
            if t == i or removed[t] or is_aspect[t]:
                continue
            if sim_matrix[i, t] > best_sim:
                best_sim = sim_matrix[i, t]
                best_j = t
        if best_j == -1 or best_sim < -1.5:
            break

        # Lưu trạng thái trước khi merge
        state = {
            "removed":   removed[:],
            "protected": protected[:],
            "aspect":    is_aspect[:],
        }
        steps.append((state, i, best_j, best_sim))
        removed[i] = True

    # Trạng thái cuối
    final_state = {
        "removed":   removed[:],
        "protected": protected[:],
        "aspect":    is_aspect[:],
    }
    return steps, final_state

steps, final_state = simulate_scm(TOKENS, IS_ASPECT, PROTECT_L, PROTECT_R, SIM)

# ── Màu sắc ───────────────────────────────────────────────────────────────────
CLR_PROTECT = "#B0BEC5"   # xám — CLS/SEP
CLR_ASPECT  = "#FFB74D"   # cam  — aspect token
CLR_NORMAL  = "#90CAF9"   # xanh dương nhạt — token thường
CLR_SRC     = "#EF5350"   # đỏ   — token bị xóa (source)
CLR_DST     = "#66BB6A"   # xanh lá — token nhận merge (destination)
CLR_REMOVED = "#ECEFF1"   # rất nhạt — đã bị xóa

EDGE_SRC    = "#B71C1C"
EDGE_DST    = "#1B5E20"
EDGE_NORMAL = "#546E7A"

# ── Vẽ ───────────────────────────────────────────────────────────────────────
N_ROWS = len(steps) + 2   # initial + N steps + final
FIG_W  = max(12, N * 1.3)
FIG_H  = N_ROWS * 1.8

fig, axes = plt.subplots(N_ROWS, 1, figsize=(FIG_W, FIG_H))
if N_ROWS == 1:
    axes = [axes]

BOX_W  = 0.8
BOX_H  = 0.55
Y_MID  = 0.5

def token_color(t, state, src=None, dst=None):
    if state["removed"][t]:
        return CLR_REMOVED, EDGE_NORMAL, 0.3
    if t == src:
        return CLR_SRC, EDGE_SRC, 1.0
    if t == dst:
        return CLR_DST, EDGE_DST, 1.0
    if state["protected"][t]:
        return CLR_PROTECT, EDGE_NORMAL, 1.0
    if state["aspect"][t]:
        return CLR_ASPECT, EDGE_NORMAL, 1.0
    return CLR_NORMAL, EDGE_NORMAL, 1.0


def draw_state(ax, tokens, state, src=None, dst=None,
               title="", sim_val=None, step_idx=None):
    ax.set_xlim(-0.5, len(tokens) - 0.5)
    ax.set_ylim(0, 1.1)
    ax.axis("off")

    for t, tok in enumerate(tokens):
        fc, ec, alpha = token_color(t, state, src, dst)
        lw = 2.5 if t in (src, dst) else 1.2

        rect = mpatches.FancyBboxPatch(
            (t - BOX_W / 2, Y_MID - BOX_H / 2),
            BOX_W, BOX_H,
            boxstyle="round,pad=0.04",
            facecolor=fc, edgecolor=ec,
            linewidth=lw, alpha=alpha,
            zorder=3,
        )
        ax.add_patch(rect)

        # Token label
        label_style = {}
        if state["removed"][t]:
            label_style = {"color": "#90A4AE", "fontstyle": "italic"}
        elif t == src:
            label_style = {"color": "white", "fontweight": "bold"}
        elif t == dst:
            label_style = {"color": "white", "fontweight": "bold"}
        ax.text(t, Y_MID, tok, ha="center", va="center",
                fontsize=9, zorder=4, **label_style)

        # Index nhỏ bên dưới box
        ax.text(t, Y_MID - BOX_H / 2 - 0.08, f"{t}",
                ha="center", va="top", fontsize=7, color="#607D8B", zorder=4)

    # Vẽ mũi tên merge: src → dst
    if src is not None and dst is not None:
        x0, x1 = src, dst
        y0 = y1 = Y_MID + BOX_H / 2 + 0.03
        ax.annotate(
            "",
            xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="->,head_width=0.25,head_length=0.15",
                color="#D32F2F", lw=2.0,
                connectionstyle="arc3,rad=-0.35" if abs(x1 - x0) > 2 else "arc3,rad=-0.2",
            ),
            zorder=5,
        )
        if sim_val is not None:
            mx = (x0 + x1) / 2
            my = y0 + 0.18
            ax.text(mx, my, f"sim={sim_val:.2f}",
                    ha="center", va="bottom", fontsize=8,
                    color="#D32F2F", fontweight="bold", zorder=6)

    # Title bên trái
    ax.text(-0.45, Y_MID, title, ha="left", va="center",
            fontsize=9, fontweight="bold", color="#263238")


# ── Row 0: trạng thái ban đầu ─────────────────────────────────────────────────
init_state = {
    "removed":   [False] * N,
    "protected": [False] * N,
    "aspect":    IS_ASPECT[:],
}
init_state["protected"][:PROTECT_L] = [True] * PROTECT_L
init_state["protected"][N - PROTECT_R:] = [True] * PROTECT_R

draw_state(axes[0], TOKENS, init_state,
           title="Khởi đầu\n(max_merges=3)")

# ── Rows 1..k: từng bước merge ───────────────────────────────────────────────
for idx, (state, src, dst, sim_val) in enumerate(steps):
    draw_state(
        axes[idx + 1], TOKENS, state,
        src=src, dst=dst, sim_val=sim_val,
        title=f"Bước {idx + 1}\nmerge [{src}]→[{dst}]",
    )

# ── Row cuối: kết quả ─────────────────────────────────────────────────────────
draw_state(axes[-1], TOKENS, final_state,
           title="Kết quả\n(sau 3 bước)")

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=CLR_PROTECT, edgecolor=EDGE_NORMAL, label="CLS / SEP (bảo vệ)"),
    mpatches.Patch(facecolor=CLR_ASPECT,  edgecolor=EDGE_NORMAL, label="Token khía cạnh (bảo vệ)"),
    mpatches.Patch(facecolor=CLR_NORMAL,  edgecolor=EDGE_NORMAL, label="Token thường"),
    mpatches.Patch(facecolor=CLR_SRC,     edgecolor=EDGE_SRC,    label="Token bị xóa (trái nhất)"),
    mpatches.Patch(facecolor=CLR_DST,     edgecolor=EDGE_DST,    label="Token nhận merge (sim cao nhất)"),
    mpatches.Patch(facecolor=CLR_REMOVED, edgecolor=EDGE_NORMAL,
                   alpha=0.5, label="Đã bị xóa"),
]
fig.legend(handles=legend_items, loc="lower center", ncol=3,
           fontsize=8.5, framealpha=0.9,
           bbox_to_anchor=(0.5, -0.01))

fig.suptitle(
    "Sequential Cosine Merging (SCM)\n"
    "Mỗi bước: chọn token TRÁI NHẤT không bị bảo vệ → merge với token có cosine similarity CAO NHẤT",
    fontsize=11, fontweight="bold", y=1.01,
)

plt.tight_layout(rect=[0, 0.06, 1, 1])

# ── Lưu file ──────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "thesis", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

pdf_path = os.path.join(OUT_DIR, "scm_steps.pdf")
png_path = os.path.join(OUT_DIR, "scm_steps.png")

fig.savefig(pdf_path, bbox_inches="tight", dpi=150)
fig.savefig(png_path, bbox_inches="tight", dpi=150)

print(f"Saved:\n  {pdf_path}\n  {png_path}")
plt.show()
