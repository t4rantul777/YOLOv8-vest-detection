import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

# ── загрузка данных ──────────────────────────────────────────────────────────
LABEL_DIRS = [
    r"datasets\construction-ppe\labels\train",
    r"datasets\construction-ppe\labels\val",
    r"datasets\construction-ppe\labels\test",
]

CLASS_NAMES = {
    0: "helmet", 1: "gloves", 2: "vest", 3: "boots", 4: "goggles",
    5: "none", 6: "Person", 7: "no_helmet", 8: "no_goggle",
    9: "no_gloves", 10: "no_boots"
}

# bbox per class: list of (w, h, area) — все в нормализованных координатах [0..1]
bbox_data = defaultdict(lambda: {"w": [], "h": [], "area": []})
# objects per image per class
objs_per_img = defaultdict(list)  # cls -> [count per image]

for ldir in LABEL_DIRS:
    for fname in os.listdir(ldir):
        if not fname.endswith(".txt") or fname == "classes.txt":
            continue
        img_cls_count = defaultdict(int)
        with open(os.path.join(ldir, fname)) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                w, h = float(parts[3]), float(parts[4])
                bbox_data[cls]["w"].append(w)
                bbox_data[cls]["h"].append(h)
                bbox_data[cls]["area"].append(w * h)
                img_cls_count[cls] += 1
        for cls, cnt in img_cls_count.items():
            objs_per_img[cls].append(cnt)

cls_ids = sorted(bbox_data.keys())
labels  = [CLASS_NAMES.get(i, str(i)) for i in cls_ids]
n_cls   = len(cls_ids)

COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2",
          "#CCB974", "#64B5CD", "#E07B54", "#6DCCDA",
          "#A8D0E6", "#F8BE26", "#D64045"]
colors = {cid: COLORS[i % len(COLORS)] for i, cid in enumerate(cls_ids)}

BG      = "#1C1C2E"
PANEL   = "#2C2C3E"
WHITE   = "white"
GRID    = "#444"

# ── фигура ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 18), facecolor=BG)
fig.suptitle("Анализ bounding box — Construction PPE Dataset",
             color=WHITE, fontsize=17, fontweight="bold", y=0.995)

gs = gridspec.GridSpec(3, 2, figure=fig,
                       hspace=0.52, wspace=0.32,
                       left=0.07, right=0.97, top=0.96, bottom=0.05)

def style_ax(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=WHITE, labelsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.25, color=GRID)
    ax.set_title(title, color=WHITE, fontsize=11, pad=8)
    if xlabel: ax.set_xlabel(xlabel, color=WHITE, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=WHITE, fontsize=9)

# ── 1. Box-plot ширины bbox по классам ──────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, "Ширина bbox (норм.) по классам",
         ylabel="Нормализованная ширина [0..1]")

bp1 = ax1.boxplot(
    [bbox_data[c]["w"] for c in cls_ids],
    patch_artist=True, notch=False,
    medianprops=dict(color="white", linewidth=2),
    whiskerprops=dict(color=GRID), capprops=dict(color=GRID),
    flierprops=dict(marker="o", markerfacecolor="#888", markersize=2, alpha=0.4),
)
for patch, cid in zip(bp1["boxes"], cls_ids):
    patch.set_facecolor(colors[cid])
    patch.set_alpha(0.8)

ax1.set_xticks(range(1, n_cls + 1))
ax1.set_xticklabels(labels, rotation=30, ha="right", color=WHITE, fontsize=8)

# ── 2. Box-plot высоты bbox по классам ──────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, "Высота bbox (норм.) по классам",
         ylabel="Нормализованная высота [0..1]")

bp2 = ax2.boxplot(
    [bbox_data[c]["h"] for c in cls_ids],
    patch_artist=True, notch=False,
    medianprops=dict(color="white", linewidth=2),
    whiskerprops=dict(color=GRID), capprops=dict(color=GRID),
    flierprops=dict(marker="o", markerfacecolor="#888", markersize=2, alpha=0.4),
)
for patch, cid in zip(bp2["boxes"], cls_ids):
    patch.set_facecolor(colors[cid])
    patch.set_alpha(0.8)

ax2.set_xticks(range(1, n_cls + 1))
ax2.set_xticklabels(labels, rotation=30, ha="right", color=WHITE, fontsize=8)

# ── 3. Гистограммы площади bbox (overlapping) ────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
style_ax(ax3, "Гистограмма площади bbox (w*h, норм.)",
         xlabel="Площадь [0..1]", ylabel="Кол-во объектов")

bins = np.linspace(0, 0.5, 60)
for cid in cls_ids:
    areas = np.clip(bbox_data[cid]["area"], 0, 0.5)
    ax3.hist(areas, bins=bins, alpha=0.55, color=colors[cid],
             label=CLASS_NAMES.get(cid, str(cid)), edgecolor="none")

ax3.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=WHITE,
           fontsize=7.5, ncol=2)
ax3.set_xlim(0, 0.5)

# ── 4. Scatter W vs H (2-D распределение) ────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
style_ax(ax4, "Scatter: ширина vs высота bbox",
         xlabel="Ширина (норм.)", ylabel="Высота (норм.)")

for cid in cls_ids:
    w = np.array(bbox_data[cid]["w"])
    h = np.array(bbox_data[cid]["h"])
    # прореживаем для читаемости
    step = max(1, len(w) // 600)
    ax4.scatter(w[::step], h[::step], s=10, alpha=0.45,
                color=colors[cid], label=CLASS_NAMES.get(cid, str(cid)))

ax4.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=WHITE,
           fontsize=7.5, ncol=2)
ax4.set_xlim(0, 1); ax4.set_ylim(0, 1)

# ── 5. Violin: кол-во объектов на изображение ────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
style_ax(ax5, "Кол-во объектов на изображение (violin)",
         ylabel="Объектов / кадр")

vdata = [objs_per_img[c] for c in cls_ids if objs_per_img[c]]
vids  = [c for c in cls_ids if objs_per_img[c]]

vp = ax5.violinplot(vdata, positions=range(1, len(vids) + 1),
                    showmedians=True, showextrema=True)
for i, (body, cid) in enumerate(zip(vp["bodies"], vids)):
    body.set_facecolor(colors[cid])
    body.set_alpha(0.7)
vp["cmedians"].set_color("white")
vp["cmedians"].set_linewidth(2)
for part in ["cbars", "cmins", "cmaxes"]:
    vp[part].set_color(GRID)

ax5.set_xticks(range(1, len(vids) + 1))
ax5.set_xticklabels([CLASS_NAMES.get(c, str(c)) for c in vids],
                    rotation=30, ha="right", color=WHITE, fontsize=8)

# ── 6. Summary таблица-бар: медиана площади bbox ─────────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
style_ax(ax6, "Медиана площади bbox по классам",
         ylabel="Медиана площади (норм.)")

medians = [float(np.median(bbox_data[c]["area"])) for c in cls_ids]
x = np.arange(n_cls)
bars = ax6.bar(x, medians, color=[colors[c] for c in cls_ids],
               edgecolor="#1C1C2E", linewidth=0.8, zorder=3)

for bar, val in zip(bars, medians):
    ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
             f"{val:.4f}", ha="center", va="bottom",
             color=WHITE, fontsize=7.5, fontweight="bold")

ax6.set_xticks(x)
ax6.set_xticklabels(labels, rotation=30, ha="right", color=WHITE, fontsize=8)
ax6.set_xlim(-0.6, n_cls - 0.4)

# ── сохранение ───────────────────────────────────────────────────────────────
out = "bbox_analysis.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out}")
plt.show()
