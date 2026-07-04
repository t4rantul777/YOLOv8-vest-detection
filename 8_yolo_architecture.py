import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np

# ── палитра ──────────────────────────────────────────────────────────────────
BG       = "#0D1117"
C_INPUT  = "#1F6FEB"
C_CONV   = "#238636"
C_C2F    = "#1F6FEB"
C_SPPF   = "#9E3B96"
C_UPSAMP = "#E3B341"
C_CONCAT = "#DA3633"
C_HEAD   = "#F78166"
C_OUT    = "#3FB950"
WHITE    = "#E6EDF3"
GRAY     = "#8B949E"
DARKGRAY = "#21262D"

fig, ax = plt.subplots(figsize=(22, 13), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 22)
ax.set_ylim(0, 13)
ax.axis("off")

fig.suptitle("YOLOv8n Architecture  —  Construction PPE Lite",
             color=WHITE, fontsize=16, fontweight="bold", y=0.98)

# ── вспомогательные функции ──────────────────────────────────────────────────

def box(x, y, w, h, color, label, sublabel="", alpha=0.9, radius=0.18):
    """Закруглённый прямоугольник с текстом."""
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle=f"round,pad=0.06,rounding_size={radius}",
                           linewidth=1.4, edgecolor=color,
                           facecolor=color + "33",  # hex прозрачность ~20%
                           zorder=3)
    ax.add_patch(rect)
    ax.text(x, y + (0.12 if sublabel else 0), label,
            ha="center", va="center", color=color,
            fontsize=8.5, fontweight="bold", zorder=4)
    if sublabel:
        ax.text(x, y - 0.22, sublabel,
                ha="center", va="center", color=GRAY,
                fontsize=6.8, zorder=4)

def arrow(x0, y0, x1, y1, color=GRAY, style="->", lw=1.3):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"),
                zorder=2)

def curve_arrow(x0, y0, x1, y1, color=C_CONCAT, rad=0.35, lw=1.3):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=lw, connectionstyle=f"arc3,rad={rad}"),
                zorder=2)

def section_label(x, y, text, color):
    ax.text(x, y, text, ha="center", va="center",
            color=color, fontsize=9, fontweight="bold",
            bbox=dict(facecolor=color+"22", edgecolor=color+"55",
                      boxstyle="round,pad=0.3", linewidth=1))

def divider(x, y0, y1):
    ax.plot([x, x], [y0, y1], color="#30363D", lw=1, ls="--", zorder=1)

# ════════════════════════════════════════════════════════════════════════════
# КОЛОНКИ:  input | backbone | neck | head
# X-позиции центров блоков
# ════════════════════════════════════════════════════════════════════════════
X_IN  = 1.2
X_BB  = 4.5   # backbone
X_NK  = 10.5  # neck (два sub-столбца)
X_NK2 = 13.8
X_HD  = 18.5  # detection head
X_OUT = 21.0

BW = 2.0   # ширина блока
BH = 0.72  # высота блока

# ── секционные метки ─────────────────────────────────────────────────────────
section_label(X_IN,  12.5, "INPUT",    C_INPUT)
section_label(X_BB,  12.5, "BACKBONE", C_CONV)
section_label(11.8,  12.5, "NECK  (PAN + FPN)", C_UPSAMP)
section_label(X_HD,  12.5, "HEAD",     C_HEAD)

divider(2.5,  0.3, 12.3)
divider(15.8, 0.3, 12.3)

# ── INPUT ────────────────────────────────────────────────────────────────────
box(X_IN, 11.3, 1.6, BH, C_INPUT,  "Image",        "640 × 640 × 3")

# ── BACKBONE ─────────────────────────────────────────────────────────────────
# P1  320×320
box(X_BB, 10.4, BW, BH, C_CONV, "Conv", "320×320 × 16")
# P2  160×160
box(X_BB,  9.2, BW, BH, C_CONV, "Conv", "160×160 × 32")
box(X_BB,  8.1, BW, BH, C_C2F,  "C2f",  "160×160 × 32")
# P3  80×80
box(X_BB,  6.9, BW, BH, C_CONV, "Conv", "80×80 × 64")
box(X_BB,  5.8, BW, BH, C_C2F,  "C2f",  "80×80 × 64")   # <- P3 skip
# P4  40×40
box(X_BB,  4.6, BW, BH, C_CONV, "Conv", "40×40 × 128")
box(X_BB,  3.5, BW, BH, C_C2F,  "C2f",  "40×40 × 128")  # <- P4 skip
# P5  20×20
box(X_BB,  2.3, BW, BH, C_CONV, "Conv", "20×20 × 256")
box(X_BB,  1.2, BW, BH, C_C2F,  "C2f",  "20×20 × 256")
box(X_BB,  0.3, BW, BH, C_SPPF, "SPPF", "20×20 × 256")  # <- P5

# стрелки вниз по backbone
for y0, y1 in [(11.3-BH/2, 10.4+BH/2), (10.4-BH/2, 9.2+BH/2),
               (9.2-BH/2, 8.1+BH/2),   (8.1-BH/2, 6.9+BH/2),
               (6.9-BH/2, 5.8+BH/2),   (5.8-BH/2, 4.6+BH/2),
               (4.6-BH/2, 3.5+BH/2),   (3.5-BH/2, 2.3+BH/2),
               (2.3-BH/2, 1.2+BH/2),   (1.2-BH/2, 0.3+BH/2)]:
    arrow(X_BB, y0, X_BB, y1, GRAY)
arrow(X_IN, 11.3 - BH/2, X_BB, 10.4 + BH/2, C_INPUT)

# ── NECK: верхний путь (FPN: top-down upsample) ──────────────────────────────
# P5 -> Upsample -> Concat(P4) -> C2f -> P4'
# P4' -> Upsample -> Concat(P3) -> C2f -> P3'

NX1 = 8.2   # upsample колонка
NX2 = 10.5  # concat+c2f колонка

# Upsample P5
box(NX1, 0.3, BW, BH, C_UPSAMP, "Upsample", "40×40 × 256")
arrow(X_BB + BW/2, 0.3, NX1 - BW/2, 0.3, C_UPSAMP)

# Concat P4 + Up(P5)  -> C2f
box(NX2, 0.3, BW, BH, C_CONCAT, "Concat", "40×40 × 384")
arrow(NX1 + BW/2, 0.3, NX2 - BW/2, 0.3, C_UPSAMP)
# skip от P4 (backbone 3.5)
curve_arrow(X_BB + BW/2, 3.5, NX2, 0.3 + BH/2, C_CONCAT, rad=-0.22)

box(NX2, 1.4, BW, BH, C_C2F, "C2f", "40×40 × 128")    # N3
arrow(NX2, 0.3 + BH/2, NX2, 1.4 - BH/2, GRAY)

# Upsample N3
box(NX1, 1.4, BW, BH, C_UPSAMP, "Upsample", "80×80 × 128")
arrow(NX2 - BW/2, 1.4, NX1 + BW/2, 1.4, C_UPSAMP)

# Concat P3 + Up(N3) -> C2f
box(NX2, 2.5, BW, BH, C_CONCAT, "Concat", "80×80 × 192")
arrow(NX1 - BW/2, 1.4, NX1 - BW/2 - 0.01, 2.5, GRAY)  # dummy, use curve
curve_arrow(NX1, 1.4 + BH/2, NX2, 2.5 - BH/2, C_UPSAMP, rad=0.22)
# skip от P3 (backbone 5.8)
curve_arrow(X_BB + BW/2, 5.8, NX2, 2.5 + BH/2, C_CONCAT, rad=-0.18)

box(NX2, 3.6, BW, BH, C_C2F, "C2f  P3'", "80×80 × 64")   # N4 -> small obj
arrow(NX2, 2.5 + BH/2, NX2, 3.6 - BH/2, GRAY)

# ── NECK: нижний путь (PAN: bottom-up downsample) ────────────────────────────
NX3 = 13.0

# Conv(stride2) P3' -> Concat(N3) -> C2f  P4'
box(NX3, 3.6, BW, BH, C_CONV, "Conv (s=2)", "40×40 × 64")
arrow(NX2 + BW/2, 3.6, NX3 - BW/2, 3.6, C_CONV)

box(NX3, 2.5, BW, BH, C_CONCAT, "Concat", "40×40 × 192")
arrow(NX3, 3.6 - BH/2, NX3, 2.5 + BH/2, GRAY)
# skip от N3 C2f(1.4)
curve_arrow(NX2 + BW/2, 1.4, NX3, 2.5 - BH/2, C_CONCAT, rad=0.22)

box(NX3, 1.4, BW, BH, C_C2F, "C2f  P4'", "40×40 × 128")  # medium obj
arrow(NX3, 2.5 - BH/2, NX3, 1.4 + BH/2, GRAY)

# Conv(stride2) P4' -> Concat(P5) -> C2f  P5'
NX4 = 13.0
box(NX4, 0.3, BW, BH, C_CONV, "Conv (s=2)", "20×20 × 128")
arrow(NX3, 1.4 - BH/2, NX4, 0.3 + BH/2, GRAY)

box(15.2, 0.3, BW, BH, C_CONCAT, "Concat", "20×20 × 384")
arrow(NX4 + BW/2, 0.3, 15.2 - BW/2, 0.3, GRAY)
# skip от SPPF
curve_arrow(X_BB + BW/2, 0.3, 15.2 - BW/2, 0.3 - 0.01, C_SPPF, rad=0.0)

box(15.2, 1.4, BW, BH, C_C2F, "C2f  P5'", "20×20 × 256")  # large obj
arrow(15.2, 0.3 + BH/2, 15.2, 1.4 - BH/2, GRAY)

# ── HEAD: три ветки детекции ──────────────────────────────────────────────────
# P3' -> Detect small   (80×80 -> stride 8)
# P4' -> Detect medium  (40×40 -> stride 16)
# P5' -> Detect large   (20×20 -> stride 32)

DX = 17.4
box(DX, 3.6, BW, BH, C_HEAD, "Detect", "stride 8\n80×80")
arrow(NX2 + BW/2, 3.6, DX - BW/2, 3.6, C_HEAD)

box(DX, 1.4, BW, BH, C_HEAD, "Detect", "stride 16\n40×40")
arrow(NX3 + BW/2, 1.4, DX - BW/2, 1.4, C_HEAD)

box(DX, 0.3, BW, BH, C_HEAD, "Detect", "stride 32\n20×20")
arrow(15.2 + BW/2, 0.3, DX - BW/2, 0.3, C_HEAD)

# ── OUTPUT ────────────────────────────────────────────────────────────────────
OX = 20.2
box(OX, 2.0, 1.8, 2.4, C_OUT, "Output\nboxes +\nscores\n4 classes",
    "", alpha=0.9, radius=0.2)
arrow(DX + BW/2, 3.6, OX - 0.9, 2.6, C_OUT)
arrow(DX + BW/2, 1.4, OX - 0.9, 2.0, C_OUT)
arrow(DX + BW/2, 0.3, OX - 0.9, 1.4, C_OUT)

# ── C2F detail inset ──────────────────────────────────────────────────────────
ins_x, ins_y, ins_w, ins_h = 5.8, 5.0, 3.2, 2.6
inset = FancyBboxPatch((ins_x, ins_y), ins_w, ins_h,
                        boxstyle="round,pad=0.1", linewidth=1,
                        edgecolor="#30363D", facecolor="#161B22", zorder=5)
ax.add_patch(inset)
ax.text(ins_x + ins_w/2, ins_y + ins_h - 0.22, "C2f block (detail)",
        ha="center", va="center", color=GRAY, fontsize=7.5,
        fontweight="bold", zorder=6)

def mini_box(x, y, w, h, c, txt):
    r = FancyBboxPatch((x - w/2, y - h/2), w, h,
                        boxstyle="round,pad=0.05", linewidth=1,
                        edgecolor=c, facecolor=c+"33", zorder=6)
    ax.add_patch(r)
    ax.text(x, y, txt, ha="center", va="center",
            color=c, fontsize=6.5, fontweight="bold", zorder=7)

def mini_arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.9),
                zorder=7)

mx = ins_x + 0.6;  my_top = ins_y + ins_h - 0.6
mini_box(mx, my_top,         1.0, 0.35, C_CONV,  "Conv 1×1")
mini_box(mx, my_top - 0.65,  1.0, 0.35, C_C2F,   "split →")
# branches
mini_box(mx - 0.65, my_top - 1.3, 0.9, 0.32, C_C2F, "Bottleneck")
mini_box(mx + 0.65, my_top - 1.3, 0.9, 0.32, C_C2F, "Bottleneck")
mini_box(mx, my_top - 1.95,  1.0, 0.35, C_CONCAT,"Concat")
mini_box(mx, my_top - 2.55,  1.0, 0.35, C_CONV,  "Conv 1×1")

for y0, y1 in [(my_top-0.175, my_top-0.475),
               (my_top-0.825, my_top-1.14)]:
    mini_arrow(mx, y0, mx, y1)
mini_arrow(mx, my_top - 1.14, mx - 0.65, my_top - 1.14)
mini_arrow(mx, my_top - 1.14, mx + 0.65, my_top - 1.14)
mini_arrow(mx - 0.65, my_top - 1.46, mx, my_top - 1.775)
mini_arrow(mx + 0.65, my_top - 1.46, mx, my_top - 1.775)
mini_arrow(mx, my_top - 2.125, mx, my_top - 2.375)

# пунктирная линия к C2f блоку backbone
ax.annotate("", xy=(ins_x + ins_w, ins_y + ins_h/2),
            xytext=(X_BB + BW/2, 5.8),
            arrowprops=dict(arrowstyle="-", color="#30363D",
                            lw=1, linestyle="dashed"),
            zorder=1)

# ── легенда ───────────────────────────────────────────────────────────────────
legend_items = [
    (C_INPUT,  "Input / reshape"),
    (C_CONV,   "Conv (BN + SiLU)"),
    (C_C2F,    "C2f (cross-stage partial)"),
    (C_SPPF,   "SPPF (spatial pyramid)"),
    (C_UPSAMP, "Upsample (nearest ×2)"),
    (C_CONCAT, "Concat (channel-wise)"),
    (C_HEAD,   "Detect head (DFL)"),
    (C_OUT,    "Output predictions"),
]
patches = [mpatches.Patch(facecolor=c+"44", edgecolor=c,
                           label=lbl, linewidth=1.2)
           for c, lbl in legend_items]
ax.legend(handles=patches, loc="upper left",
          bbox_to_anchor=(0.0, 0.97),
          facecolor="#161B22", edgecolor="#30363D",
          labelcolor=WHITE, fontsize=7.5,
          title="Блоки", title_fontsize=8,
          framealpha=0.95, ncol=1)

# ── сохранение ────────────────────────────────────────────────────────────────
out = "yolo_architecture.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out}")
plt.show()
