"""
Inference on test images → composite collage for the report.
"""
import os, random
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ultralytics import YOLO

BASE       = Path(__file__).parent
MODEL_PATH = BASE / "runs/detect/runs/train/ppe_lite/weights/best.pt"
IMAGES_DIR = BASE / "datasets/construction-ppe/images/test"
OUT        = BASE / "inference_examples.png"

# цвета классов (BGR для OpenCV, hex для легенды)
CLS_COLOR_BGR = {
    "helmet":    (50,  205, 50),
    "gloves":    (255, 165, 0),
    "vest":      (30,  144, 255),
    "boots":     (147, 112, 219),
    "no_helmet": (0,   0,   220),
}
CLS_COLOR_HEX = {
    "helmet":    "#32CD32",
    "gloves":    "#FFA500",
    "vest":      "#1E90FF",
    "boots":     "#9370DB",
    "no_helmet": "#DC0000",
}

model = YOLO(str(MODEL_PATH))

# собираем тестовые изображения
all_imgs = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png"))
random.seed(7)
random.shuffle(all_imgs)

# ищем 6 разнообразных кадров (с разными наборами детекций)
selected = []
seen_cls_sets = []
for img_path in all_imgs:
    if len(selected) >= 6:
        break
    res = model(str(img_path), conf=0.25, iou=0.45, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        continue
    cls_set = frozenset(int(b.cls[0]) for b in res.boxes)
    # берём только если набор классов новый
    if cls_set not in seen_cls_sets:
        seen_cls_sets.append(cls_set)
        selected.append((img_path, res))

print(f"Selected {len(selected)} images")

# ── рисуем bbox на каждом кадре ──────────────────────────────────────────────
def annotate(img_bgr, result):
    img = img_bgr.copy()
    names = result.names
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_name = names[int(box.cls[0])]
        conf     = float(box.conf[0])
        color    = CLS_COLOR_BGR.get(cls_name, (200, 200, 200))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        # фон метки
        y_bg = max(y1 - 1, th + 4)
        cv2.rectangle(img, (x1, y_bg - th - 4), (x1 + tw + 4, y_bg), color, -1)
        cv2.putText(img, label, (x1 + 2, y_bg - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


THUMB = 480   # размер миниатюры (квадрат)

def square_thumb(img_bgr, size):
    h, w = img_bgr.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img_bgr, (nw, nh))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas


thumbs = []
captions = []
for img_path, res in selected:
    img_bgr = cv2.imread(str(img_path))
    ann     = annotate(img_bgr, res)
    thumb   = square_thumb(ann, THUMB)
    thumbs.append(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))

    names = res.names
    cls_counts = {}
    for b in res.boxes:
        n = names[int(b.cls[0])]
        cls_counts[n] = cls_counts.get(n, 0) + 1
    cap = "  ".join(f"{n}×{c}" for n, c in sorted(cls_counts.items()))
    captions.append(cap)

# ── matplotlib collage ────────────────────────────────────────────────────────
COLS = 3
ROWS = 2
BG = "#0D1117"
WHITE = "#E6EDF3"
GRAY  = "#8B949E"

fig, axes = plt.subplots(ROWS, COLS, figsize=(18, 13), facecolor=BG)
fig.suptitle("Примеры применения модели на тестовых изображениях\nConstruction PPE — YOLOv8n",
             color=WHITE, fontsize=15, fontweight="bold", y=0.98)

for idx, ax in enumerate(axes.flat):
    ax.set_facecolor(BG)
    if idx < len(thumbs):
        ax.imshow(thumbs[idx])
        ax.set_title(captions[idx], color=GRAY, fontsize=9, pad=5)
    ax.axis("off")

# легенда
patches = [mpatches.Patch(facecolor=h, label=n, edgecolor="#333")
           for n, h in CLS_COLOR_HEX.items()]
fig.legend(handles=patches, loc="lower center", ncol=len(patches),
           facecolor="#161B22", edgecolor="#30363D", labelcolor=WHITE,
           fontsize=10, title="Классы детекции", title_fontsize=10,
           bbox_to_anchor=(0.5, 0.01), framealpha=0.95)

plt.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.07,
                    hspace=0.12, wspace=0.05)
plt.savefig(str(OUT), dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved: {OUT}")
plt.show()
