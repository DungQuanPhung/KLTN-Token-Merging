# -*- coding: utf-8 -*-
"""Visualize Sequential Cosine Merging (SCM) với ví dụ:
   "food was tasty but service was poor" — aspect: service

SPC format: [CLS] food was tasty but service was poor [SEP] service [SEP]
Index:         0    1    2    3     4     5       6    7    8      9      10

Chạy:   python scripts/visualize_scm_example.py
Output: thesis/figures/scm_example.pdf  và  thesis/figures/scm_example.png
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False

# ─── Chuỗi đầu vào ────────────────────────────────────────────────────────────
TOKENS = ["[CLS]", "food", "was", "tasty", "but",
          "service", "was", "poor", "[SEP]", "service", "[SEP]"]
IS_ASPECT = [False, False, False, False, False,
             False, False, False, False, True, False]
PROTECT_L = 1   # [CLS]
PROTECT_R = 1   # [SEP] cuối

N = len(TOKENS)   # = 11

# ─── Ma trận cosine similarity giả lập (minh hoạ ngữ nghĩa) ──────────────────
SIM = np.full((N, N), 0.25)
np.fill_diagonal(SIM, 1.0)

# food(1) gần tasty(3) nhất — cùng mô tả chất lượng món ăn
SIM[1, 3] = SIM[3, 1] = 0.72
SIM[1, 5] = SIM[5, 1] = 0.50   # food ↔ service (đều là danh từ chủ đề)
SIM[1, 2] = SIM[2, 1] = 0.38

# was(2) gần was(6) nhất — cùng từ
SIM[2, 6] = SIM[6, 2] = 0.95
SIM[2, 4] = SIM[4, 2] = 0.32

# but(4) gần poor(7) nhất — liên từ đối lập dẫn vào nhận xét tiêu cực
SIM[4, 7] = SIM[7, 4] = 0.58
SIM[4, 6] = SIM[6, 4] = 0.40

# tasty(3) ↔ poor(7): đối nghĩa nhưng cùng nhóm tính từ đánh giá
SIM[3, 7] = SIM[7, 3] = 0.30

# ─── Simulate SCM ─────────────────────────────────────────────────────────────
def simulate_scm(tokens, is_aspect, protect_l, protect_r, sim_matrix):
    n = len(tokens)
    removed   = [False] * n
    protected = [False] * n
    for i in range(protect_l):
        protected[i] = True
    for i in range(n - protect_r, n):
        protected[i] = True

    max_merges = (n - protect_l - protect_r) // 2
    steps = []   # (snapshot_before, src, dst, sim_val)
    merged_into = list(range(n))   # merged_into[t] = label token t đã hợp nhất

    for _ in range(max_merges):
        # Tìm token nguồn: trái nhất không bị xóa, không protected, không aspect
        src = None
        for t in range(n):
            if not removed[t] and not protected[t] and not is_aspect[t]:
                src = t
                break
        if src is None:
            break

        # Tìm token đích: cosine cao nhất, không bị xóa, không aspect, khác src
        best_dst, best_sim = -1, -2.0
        for t in range(n):
            if t == src or removed[t] or is_aspect[t]:
                continue
            if sim_matrix[src, t] > best_sim:
                best_sim = sim_matrix[src, t]
                best_dst = t
        if best_dst == -1 or best_sim < -1.5:
            break

        steps.append({
            "removed":   removed[:],
            "protected": protected[:],
            "src": src,
            "dst": best_dst,
            "sim": best_sim,
            "merged_labels": merged_into[:],
        })
        removed[src] = True
        # Cập nhật merged_into: dst nhận thêm label từ src
        merged_into[best_dst] = best_dst  # giữ dst, chỉ đánh dấu src đã xóa

    final_snap = {"removed": removed[:], "protected": protected[:],
                  "src": None, "dst": None, "sim": None,
                  "merged_labels": merged_into[:]}
    return steps, final_snap, max_merges

steps, final_snap, max_merges = simulate_scm(
    TOKENS, IS_ASPECT, PROTECT_L, PROTECT_R, SIM
)

# ─── Nhãn hiển thị (theo dõi token nào hợp nhất vào đâu) ─────────────────────
# display_label[t] = chuỗi hiện thị trên box tại bước cuối
step_labels = []
cur_labels = list(TOKENS)
step_labels.append(cur_labels[:])

for step in steps:
    src, dst = step["src"], step["dst"]
    new_labels = cur_labels[:]
    # Gộp label: dst nhận thêm tên của src (viết tắt nếu dài)
    src_name = cur_labels[src].replace("[CLS]", "CLS").replace("[SEP]", "SEP")
    dst_name = cur_labels[dst].replace("[CLS]", "CLS").replace("[SEP]", "SEP")
    combined = f"{dst_name}+{src_name}"
    if len(combined) > 14:
        combined = dst_name + "+"  # rút gọn nếu quá dài
    new_labels[dst] = combined
    new_labels[src] = "×"
    cur_labels = new_labels
    step_labels.append(cur_labels[:])

# ─── Màu sắc ──────────────────────────────────────────────────────────────────
C_PROTECT = "#B0BEC5"   # xám     — CLS / SEP
C_ASPECT  = "#FFB74D"   # cam     — token khía cạnh
C_NORMAL  = "#90CAF9"   # xanh    — token thường
C_SRC     = "#EF5350"   # đỏ      — token nguồn (bị xóa bước này)
C_DST     = "#66BB6A"   # xanh lá — token đích (nhận hợp nhất)
C_DEAD    = "#F5F5F5"   # trắng   — đã bị xóa trước đó

E_SRC  = "#B71C1C"
E_DST  = "#1B5E20"
E_STD  = "#607D8B"

# ─── Vẽ ───────────────────────────────────────────────────────────────────────
N_ROWS = len(steps) + 2   # initial + steps + final
FIG_W  = max(13, N * 1.25)
FIG_H  = N_ROWS * 1.85

fig, axes = plt.subplots(N_ROWS, 1, figsize=(FIG_W, FIG_H))
if N_ROWS == 1:
    axes = [axes]

BOX_W = 0.82
BOX_H = 0.50
Y_MID = 0.52


def draw_row(ax, snap, labels, row_title, step_idx=None):
    removed   = snap["removed"]
    protected = snap["protected"]
    src = snap.get("src")
    dst = snap.get("dst")
    sim = snap.get("sim")

    ax.set_xlim(-0.7, N - 0.3)
    ax.set_ylim(0.0, 1.15)
    ax.axis("off")

    for t in range(N):
        # Chọn màu
        if removed[t] and t != src:           # đã xóa trước đó
            fc, ec, lw, alpha = C_DEAD, E_STD, 1.0, 0.5
        elif t == src:                         # đang bị xóa bước này
            fc, ec, lw, alpha = C_SRC, E_SRC, 2.5, 1.0
        elif t == dst:                         # đang nhận hợp nhất
            fc, ec, lw, alpha = C_DST, E_DST, 2.5, 1.0
        elif IS_ASPECT[t]:                     # token khía cạnh
            fc, ec, lw, alpha = C_ASPECT, E_STD, 1.5, 1.0
        elif protected[t]:                     # CLS / SEP
            fc, ec, lw, alpha = C_PROTECT, E_STD, 1.2, 1.0
        else:                                  # token thường
            fc, ec, lw, alpha = C_NORMAL, E_STD, 1.2, 1.0

        rect = mpatches.FancyBboxPatch(
            (t - BOX_W / 2, Y_MID - BOX_H / 2), BOX_W, BOX_H,
            boxstyle="round,pad=0.05",
            facecolor=fc, edgecolor=ec,
            linewidth=lw, alpha=alpha, zorder=3,
        )
        ax.add_patch(rect)

        # Nhãn token
        txt_kw = {"ha": "center", "va": "center", "fontsize": 8, "zorder": 4}
        if removed[t] and t != src:
            txt_kw.update(color="#BDBDBD", fontstyle="italic")
        elif t in (src, dst):
            txt_kw.update(color="white", fontweight="bold")

        lbl = labels[t]
        ax.text(t, Y_MID, lbl, **txt_kw)

        # Chỉ số vị trí nhỏ bên dưới
        ax.text(t, Y_MID - BOX_H / 2 - 0.07, str(t),
                ha="center", va="top", fontsize=6.5,
                color="#90A4AE", zorder=4)

    # Mũi tên từ src → dst (cung cong)
    if src is not None and dst is not None:
        rad = -0.35 if abs(dst - src) > 2 else -0.22
        ax.annotate(
            "",
            xy=(dst, Y_MID + BOX_H / 2 + 0.04),
            xytext=(src, Y_MID + BOX_H / 2 + 0.04),
            arrowprops=dict(
                arrowstyle="->,head_width=0.22,head_length=0.12",
                color=E_SRC, lw=2.0,
                connectionstyle=f"arc3,rad={rad}",
            ),
            zorder=5,
        )
        if sim is not None:
            mx = (src + dst) / 2
            my = Y_MID + BOX_H / 2 + (0.30 if abs(dst - src) > 2 else 0.22)
            ax.text(mx, my, f"sim = {sim:.2f}",
                    ha="center", va="bottom", fontsize=7.5,
                    color=E_SRC, fontweight="bold", zorder=6)

    # Tiêu đề hàng (bên trái)
    ax.text(-0.65, Y_MID, row_title,
            ha="left", va="center", fontsize=8.5,
            fontweight="bold", color="#263238")


# Hàng 0: trạng thái ban đầu
init_snap = {
    "removed": [False] * N, "protected": [False] * N,
    "src": None, "dst": None, "sim": None,
}
init_snap["protected"][:PROTECT_L] = [True] * PROTECT_L
init_snap["protected"][N - PROTECT_R:] = [True] * PROTECT_R

draw_row(axes[0], init_snap, step_labels[0],
         f"Ban đầu\n$T_{{\\max}}={max_merges}$")

# Hàng 1..k: từng bước merge
for idx, step in enumerate(steps):
    snap = {
        "removed":   step["removed"],
        "protected": init_snap["protected"],
        "src": step["src"],
        "dst": step["dst"],
        "sim": step["sim"],
    }
    draw_row(axes[idx + 1], snap, step_labels[idx],
             f"Bước {idx + 1}\n$i={step['src']}\\!\\to\\! j={step['dst']}$")

# Hàng cuối: kết quả
draw_row(axes[-1],
         {"removed": final_snap["removed"], "protected": init_snap["protected"],
          "src": None, "dst": None, "sim": None},
         step_labels[-1],
         "Kết quả")

# ─── Legend ───────────────────────────────────────────────────────────────────
legend_elems = [
    mpatches.Patch(fc=C_PROTECT, ec=E_STD,  label="CLS / SEP (bảo vệ)"),
    mpatches.Patch(fc=C_ASPECT,  ec=E_STD,  label="Token khía cạnh (bảo vệ)"),
    mpatches.Patch(fc=C_NORMAL,  ec=E_STD,  label="Token thường"),
    mpatches.Patch(fc=C_SRC,     ec=E_SRC,  label="Token nguồn $i$ (trái nhất, bị xóa)"),
    mpatches.Patch(fc=C_DST,     ec=E_DST,  label="Token đích $j$ (cosine cao nhất)"),
    mpatches.Patch(fc=C_DEAD,    ec=E_STD,  alpha=0.6, label="Đã xóa ở bước trước"),
]
fig.legend(handles=legend_elems, loc="lower center", ncol=3,
           fontsize=8, framealpha=0.95,
           bbox_to_anchor=(0.5, -0.005))

fig.suptitle(
    "Sequential Cosine Merging (SCM)\n"
    r"Câu: \textit{food was tasty but service was poor} — aspect: \textbf{service}",
    fontsize=11, fontweight="bold", y=1.01,
    usetex=False,
)

plt.tight_layout(rect=[0, 0.07, 1, 1])

# ─── Lưu ──────────────────────────────────────────────────────────────────────
OUT = os.path.join(os.path.dirname(__file__), "..", "thesis", "figures")
os.makedirs(OUT, exist_ok=True)

for ext in ("pdf", "png"):
    path = os.path.join(OUT, f"scm_example.{ext}")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"Saved: {path}")

plt.show()
