import os
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── данные ──────────────────────────────────────────────────────────────────
splits = {
    "train": r"datasets\construction-ppe\labels\train",
    "val":   r"datasets\construction-ppe\labels\val",
    "test":  r"datasets\construction-ppe\labels\test",
}

class_names = {
    0: "helmet", 1: "gloves", 2: "vest", 3: "boots", 4: "goggles",
    5: "none", 6: "Person", 7: "no_helmet", 8: "no_goggle",
    9: "no_gloves", 10: "no_boots"
}

total = Counter()
per_split = {"train": Counter(), "val": Counter(), "test": Counter()}

for split, path in splits.items():
    for fname in os.listdir(path):
        if fname.endswith(".txt") and fname != "classes.txt":
            with open(os.path.join(path, fname)) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls = int(parts[0])
                        per_split[split][cls] += 1
                        total[cls] += 1

# ── подготовка ──────────────────────────────────────────────────────────────
cls_ids   = sorted(total.keys())
labels    = [class_names.get(i, str(i)) for i in cls_ids]
counts    = [total[i] for i in cls_ids]
n_total   = sum(counts)

COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2",
          "#CCB974", "#64B5CD", "#E07B54", "#6DCCDA",
          "#A8D0E6", "#F8BE26", "#D64045"]
colors = COLORS[:len(cls_ids)]

split_colors = {"train": "#4C72B0", "val": "#55A868", "test": "#C44E52"}

# ── рисунок ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12), facecolor="#1C1C2E")
fig.suptitle("Распределение классов — Construction PPE Dataset",
             color="white", fontsize=16, fontweight="bold", y=0.98)

# ── 1. Pie chart (total) ─────────────────────────────────────────────────────
ax1 = fig.add_axes([0.02, 0.38, 0.38, 0.55])
ax1.set_facecolor("#1C1C2E")

explode = [0.04] * len(cls_ids)
wedges, texts, autotexts = ax1.pie(
    counts, labels=None, colors=colors,
    autopct=lambda p: f"{p:.1f}%" if p > 2 else "",
    startangle=140, explode=explode,
    wedgeprops=dict(edgecolor="#1C1C2E", linewidth=1.5),
    pctdistance=0.78,
)
for at in autotexts:
    at.set_color("white")
    at.set_fontsize(9)

legend_labels = [f"{lbl}  ({cnt:,})" for lbl, cnt in zip(labels, counts)]
ax1.legend(wedges, legend_labels, title="Класс  (кол-во)",
           loc="lower center", bbox_to_anchor=(0.5, -0.18),
           fontsize=8.5, title_fontsize=9,
           facecolor="#2C2C3E", edgecolor="#555", labelcolor="white",
           ncol=2)
ax1.set_title(f"Pie — все разбивки\n(всего {n_total:,} bbox)",
              color="white", fontsize=11, pad=10)

# ── 2. Bar chart total ───────────────────────────────────────────────────────
ax2 = fig.add_axes([0.43, 0.38, 0.55, 0.55])
ax2.set_facecolor("#2C2C3E")
x = np.arange(len(cls_ids))
bars = ax2.bar(x, counts, color=colors, edgecolor="#1C1C2E", linewidth=0.8, zorder=3)

for bar, cnt in zip(bars, counts):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
             f"{cnt:,}", ha="center", va="bottom", color="white",
             fontsize=8.5, fontweight="bold")

ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=30, ha="right", color="white", fontsize=9)
ax2.set_ylabel("Количество аннотаций", color="white", fontsize=10)
ax2.tick_params(colors="white")
ax2.spines[["top", "right"]].set_visible(False)
ax2.spines[["left", "bottom"]].set_color("#555")
ax2.yaxis.grid(True, linestyle="--", alpha=0.3, color="#888", zorder=0)
ax2.set_title("Bar — итого по всем разбивкам", color="white", fontsize=11, pad=10)

# ── 3. Stacked bar по split-ам ───────────────────────────────────────────────
ax3 = fig.add_axes([0.02, 0.04, 0.96, 0.28])
ax3.set_facecolor("#2C2C3E")

bar_w = 0.26
offsets = {"train": -bar_w, "val": 0, "test": bar_w}

for split, offset in offsets.items():
    sc = per_split[split]
    vals = [sc.get(i, 0) for i in cls_ids]
    ax3.bar(x + offset, vals, width=bar_w,
            color=split_colors[split], edgecolor="#1C1C2E",
            linewidth=0.6, label=split, zorder=3)

ax3.set_xticks(x)
ax3.set_xticklabels(labels, rotation=30, ha="right", color="white", fontsize=9)
ax3.set_ylabel("Кол-во аннотаций", color="white", fontsize=10)
ax3.tick_params(colors="white")
ax3.spines[["top", "right"]].set_visible(False)
ax3.spines[["left", "bottom"]].set_color("#555")
ax3.yaxis.grid(True, linestyle="--", alpha=0.3, color="#888", zorder=0)
ax3.legend(facecolor="#2C2C3E", edgecolor="#555", labelcolor="white",
           fontsize=9, title="Разбивка", title_fontsize=9)
ax3.set_title("Grouped bar — train / val / test", color="white", fontsize=11, pad=8)

# ── сохранение ───────────────────────────────────────────────────────────────
out = "class_distribution.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out}")
plt.show()
