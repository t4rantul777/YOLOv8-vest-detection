"""
Generates all report visualizations for Construction PPE project.
Output: report_visuals/ folder with PNG files.
"""

import os, random
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from ultralytics import YOLO

BASE = Path(__file__).parent
LABELS_TRAIN = BASE / "datasets/construction-ppe/labels/train"
LABELS_TEST  = BASE / "datasets/construction-ppe/labels/test"
IMAGES_TEST  = BASE / "datasets/construction-ppe/images/test"
MODEL_PATH   = BASE / "runs/detect/runs/train/ppe_lite/weights/best.pt"
OUT          = BASE / "report_visuals"
OUT.mkdir(exist_ok=True)

CLASS_NAMES  = {0: "helmet", 2: "vest", 6: "person", 7: "no_helmet"}
CLASS_COLORS = {0: "#2ecc71", 2: "#f39c12", 6: "#3498db", 7: "#e74c3c"}


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_labels(label_dir):
    classes, widths, heights = [], [], []
    for f in Path(label_dir).glob("*.txt"):
        for line in f.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                classes.append(int(parts[0]))
                widths.append(float(parts[3]))
                heights.append(float(parts[4]))
    return classes, widths, heights


# ── 1. collect all annotations ───────────────────────────────────────────────

all_cls, all_w, all_h = [], [], []
for d in [LABELS_TRAIN, LABELS_TEST]:
    c, w, h = parse_labels(d)
    all_cls += c; all_w += w; all_h += h

counts   = Counter(all_cls)
cls_keys = sorted(counts.keys())
labels   = [CLASS_NAMES.get(k, str(k)) for k in cls_keys]
values   = [counts[k] for k in cls_keys]
colors   = [CLASS_COLORS.get(k, "#95a5a6") for k in cls_keys]


# ── 2. class distribution (bar + pie) ────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Распределение классов в датасете Construction PPE",
             fontsize=14, fontweight="bold")

bars = ax1.bar(labels, values, color=colors, edgecolor="white", linewidth=1.5)
ax1.set_title("Bar chart: количество аннотаций")
ax1.set_ylabel("Количество объектов")
ax1.grid(axis="y", alpha=0.3)
for bar, val in zip(bars, values):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
             str(val), ha="center", fontsize=11, fontweight="bold")

_, _, autotexts = ax2.pie(values, labels=labels, colors=colors,
                           autopct="%1.1f%%", startangle=140, pctdistance=0.82)
for at in autotexts:
    at.set_fontsize(10)
ax2.set_title("Pie chart: доля каждого класса")

plt.tight_layout()
plt.savefig(OUT / "1_class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ 1_class_distribution.png")


# ── 3. bbox size histograms ───────────────────────────────────────────────────

areas = [w * h for w, h in zip(all_w, all_h)]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Гистограммы размеров Bounding Boxes", fontsize=14, fontweight="bold")

def hist_ax(ax, data, title, xlabel, color):
    ax.hist(data, bins=60, color=color, edgecolor="white", alpha=0.85)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("Количество")
    med = np.median(data)
    ax.axvline(med, color="black", linestyle="--", linewidth=1.5,
               label=f"Медиана: {med:.3f}")
    ax.legend(); ax.grid(alpha=0.3)

hist_ax(axes[0, 0], all_w,  "Ширина bbox",  "Ширина (0–1)",  "#3498db")
hist_ax(axes[0, 1], all_h,  "Высота bbox",  "Высота (0–1)",  "#e74c3c")
hist_ax(axes[1, 0], areas,  "Площадь bbox", "Площадь (0–1)", "#2ecc71")

axes[1, 1].scatter(all_w[:3000], all_h[:3000],
                   alpha=0.25, c="#9b59b6", s=6)
axes[1, 1].set_title("Ширина vs Высота (scatter)")
axes[1, 1].set_xlabel("Ширина"); axes[1, 1].set_ylabel("Высота")
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "2_bbox_histograms.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ 2_bbox_histograms.png")


# ── 4. system architecture diagram ───────────────────────────────────────────

fig, ax = plt.subplots(figsize=(18, 7))
ax.set_xlim(0, 18); ax.set_ylim(0, 7)
ax.axis("off")
fig.patch.set_facecolor("#f0f4f8")
ax.set_facecolor("#f0f4f8")
ax.set_title("Схема системы детекции СИЗ на строительных объектах",
             fontsize=15, fontweight="bold", pad=18)

BOXES = [
    # x,    y,   w,   h,     fill,     label_lines,                 sublabel
    (0.4,  2.8, 2.4, 1.4, "#2980b9", "Источник\n(камера / фото)",  ""),
    (3.4,  2.8, 2.4, 1.4, "#16a085", "Предобработка\nresize 640×640", "OpenCV"),
    (6.8,  1.8, 2.8, 3.4, "#c0392b", "YOLOv8 Nano\n─────────\nBackbone\nNeck\nHead", "PyTorch / CUDA"),
    (10.4, 2.8, 2.4, 1.4, "#d35400", "Post-processing\nNMS + threshold", "conf 0.12"),
    (13.4, 4.8, 2.0, 1.0, "#8e44ad", "Bboxes + Labels\n+ Confidence",  ""),
    (13.4, 3.2, 2.0, 1.0, "#27ae60", "Safety Analyzer\n(IoU matching)",  ""),
    (13.4, 1.5, 2.0, 1.0, "#e67e22", "Streamlit UI\n(web interface)",  ""),
]

for (x, y, w, h, col, txt, sub) in BOXES:
    rect = mpatches.FancyBboxPatch((x, y), w, h,
                                   boxstyle="round,pad=0.12",
                                   facecolor=col, edgecolor="white", linewidth=2)
    ax.add_patch(rect)
    yoff = 0.15 if sub else 0
    ax.text(x + w/2, y + h/2 + yoff, txt, ha="center", va="center",
            fontsize=8.5, color="white", fontweight="bold", linespacing=1.35)
    if sub:
        ax.text(x + w/2, y + 0.22, sub, ha="center", va="center",
                fontsize=7, color="#dfe6e9", style="italic")

ARROWS = [
    (2.8, 3.5, 3.4, 3.5),
    (5.8, 3.5, 6.8, 3.5),
    (9.6, 3.5, 10.4, 3.5),
    (12.8, 3.5, 13.4, 5.3),
    (12.8, 3.5, 13.4, 3.7),
    (12.8, 3.5, 13.4, 2.0),
]
for (x1, y1, x2, y2) in ARROWS:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=2.2))

ax.text(9, 0.6,
        "Датасет: 1 270 фото  |  4 класса: helmet, vest, person, no_helmet  |  mAP@0.5 = 0.754",
        ha="center", fontsize=10.5, color="#2c3e50", style="italic")

plt.tight_layout()
plt.savefig(OUT / "3_system_architecture.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ 3_system_architecture.png")


# ── 5. model inference on 2 test images ──────────────────────────────────────

model = YOLO(str(MODEL_PATH))
test_imgs = list(IMAGES_TEST.glob("*.jpg")) + list(IMAGES_TEST.glob("*.png"))
if not test_imgs:
    print("⚠  Тестовые изображения не найдены, пропускаю инференс")
else:
    random.seed(42)
    sample = random.sample(test_imgs, min(2, len(test_imgs)))
    for i, img_path in enumerate(sample, 1):
        res = model(str(img_path), conf=0.25, iou=0.45, verbose=False)[0]
        annotated = res.plot()
        out_path = OUT / f"4_inference_example{i}.jpg"
        cv2.imwrite(str(out_path), annotated)
        n = len(res.boxes)
        print(f"✓ 4_inference_example{i}.jpg  ({n} объект{'а' if n in (2,3,4) else 'ов'}, src: {img_path.name})")


# ── 6. summary ────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"Все файлы сохранены в: {OUT}")
print(f"{'='*60}")
EXISTING = {
    "losses.png":                    "Лосс-кривые обучения (train/val)",
    "runs/detect/runs/train/ppe_lite/confusion_matrix.png": "Confusion Matrix",
    "runs/detect/runs/train/ppe_lite/BoxPR_curve.png":      "PR кривая",
    "runs/detect/runs/train/ppe_lite/BoxF1_curve.png":      "F1 кривая",
    "runs/detect/runs/train/ppe_lite/results.png":          "Все метрики YOLO",
    "runs/detect/runs/train/ppe_lite/val_batch0_pred.jpg":  "Примеры с разметкой (val)",
    "runs/detect/runs/train/ppe_lite/train_batch0.jpg":     "Примеры датасета с bbox",
    "runs/detect/runs/train/ppe_lite/labels.jpg":           "Распределение меток YOLO",
}
print("\nУже готовые файлы (созданы при обучении):")
for p, desc in EXISTING.items():
    full = BASE / p
    exists = "✓" if full.exists() else "✗"
    print(f"  {exists}  {desc}")
    print(f"     {full}")
print()
for f in sorted(OUT.iterdir()):
    print(f"  ✓  {f.name}  ({f.stat().st_size // 1024} KB)")
