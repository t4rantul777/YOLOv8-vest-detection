"""
Презентация для защиты — Construction PPE Lite
Навигация: → / Space / Click = следующий слайд
            ← / Backspace    = предыдущий слайд
            Q / Esc          = выход
            цифры 1-9        = прыжок к слайду
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from matplotlib.widgets import Button
import sys

BASE = Path(__file__).parent
RUNS = BASE / "runs/detect/runs/train/ppe_lite"

# ── слайды: (путь, заголовок, описание) ─────────────────────────────────────
SLIDES = [
    # 0. Титул
    (None,
     "Construction PPE Lite",
     "Система мониторинга средств индивидуальной защиты\n"
     "на строительных объектах\n\n"
     "YOLOv8n  |  4 класса  |  mAP@0.5 = 0.756"),

    # 1. Архитектура
    (BASE / "yolo_architecture.png",
     "Архитектура YOLOv8n",
     "Backbone → Neck (FPN+PAN) → Head\n"
     "три масштаба детекции: 80×80 / 40×40 / 20×20"),

    # 2. Распределение классов
    (BASE / "class_distribution.png",
     "Распределение классов в датасете",
     "6 132 аннотации  |  дисбаланс: boots в 4.5× реже vest"),

    # 3. Анализ bbox
    (BASE / "bbox_analysis.png",
     "Анализ размеров Bounding Box",
     "vest — крупнее всех (медиана площади ~10%)  |  "
     "большинство объектов < 5% кадра"),

    # 4. labels.jpg (YOLO auto-generated)
    (RUNS / "labels.jpg",
     "Статистика обучающей разметки (YOLO)",
     "Распределение, центры и размеры bbox в train-split"),

    # 5. train_batch0
    (RUNS / "train_batch0.jpg",
     "Примеры обучающих батчей с разметкой",
     "Аугментированные изображения — mosaic, flip, HSV"),

    # 6. Кривые обучения (results.png)
    (RUNS / "results.png",
     "Кривые обучения — 50 эпох",
     "train/val loss ↓  |  Precision ↑  |  Recall ↑  |  mAP50 ↑\n"
     "Синхронное снижение train и val — переобучения нет"),

    # 7. Confusion matrix
    (RUNS / "confusion_matrix_normalized.png",
     "Матрица ошибок (нормализованная)",
     "person 0.90  |  vest 0.84  |  helmet 0.81  |  no_helmet 0.29\n"
     "Слабость no_helmet — малая выборка (400 экз.) + визуальная схожесть с фоном"),

    # 8. PR-кривая
    (RUNS / "BoxPR_curve.png",
     "Precision-Recall кривая",
     "mAP@0.5 = 0.756  |  person 0.895  |  vest 0.851  |  helmet 0.836  |  no_helmet 0.442"),

    # 9. F1-кривая
    (RUNS / "BoxF1_curve.png",
     "F1-Confidence кривая",
     "Оптимальный порог confidence = 0.238  (F1 max = 0.72)\n"
     "В системе используется conf = 0.12 — занижен намеренно, "
     "чтобы не пропускать нарушения"),

    # 10. Val predictions
    (RUNS / "val_batch0_pred.jpg",
     "Предсказания на валидационных данных",
     "Автоматически сохранено YOLOv8 после обучения"),

    # 11. Инференс — практические примеры
    (BASE / "inference_examples.png",
     "Примеры работы на тестовых изображениях",
     "Зелёный = helmet  |  Синий = vest  |  Красный = no_helmet\n"
     "Confidence 0.7–0.9 на чётко видимых объектах"),

    # 12. Интерфейс Streamlit
    (BASE / "streamlit_screenshot.png",
     "Веб-интерфейс — Streamlit",
     "Три режима: Image / Video / Graphs\n"
     "Готов к использованию в браузере без установки кода"),
]

# ── состояние ────────────────────────────────────────────────────────────────
state = {"idx": 0}
N = len(SLIDES)

BG    = "#0D1117"
PANEL = "#161B22"
WHITE = "#E6EDF3"
GRAY  = "#8B949E"
BLUE  = "#1F6FEB"

# ── фигура ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10), facecolor=BG)
fig.canvas.manager.set_window_title("PPE Lite — Презентация для защиты")

ax_img  = fig.add_axes([0.0,  0.10, 1.0, 0.85])   # область изображения/текста
ax_info = fig.add_axes([0.0,  0.0,  1.0, 0.10])   # нижняя полоса

ax_prev = fig.add_axes([0.01, 0.01, 0.07, 0.07])
ax_next = fig.add_axes([0.92, 0.01, 0.07, 0.07])

btn_prev = Button(ax_prev, "◀  Назад", color=PANEL, hovercolor="#21262D")
btn_next = Button(ax_next, "Вперёд  ▶", color=PANEL, hovercolor="#21262D")
for btn in (btn_prev, btn_next):
    btn.label.set_color(WHITE)
    btn.label.set_fontsize(10)

for a in (ax_img, ax_info):
    a.set_facecolor(BG)
    a.axis("off")


def render(idx):
    ax_img.cla();  ax_img.axis("off");  ax_img.set_facecolor(BG)
    ax_info.cla(); ax_info.axis("off"); ax_info.set_facecolor(BG)

    path, title, desc = SLIDES[idx]

    # ── прогресс-бар ─────────────────────────────────────────────────────────
    ax_info.add_patch(mpatches.Rectangle(
        (0, 0.7), 1.0, 0.08, transform=ax_info.transAxes,
        color="#21262D", clip_on=False))
    ax_info.add_patch(mpatches.Rectangle(
        (0, 0.7), (idx + 1) / N, 0.08, transform=ax_info.transAxes,
        color=BLUE, clip_on=False))

    ax_info.text(0.5, 0.3, desc,
                 ha="center", va="center", color=GRAY,
                 fontsize=9.5, transform=ax_info.transAxes,
                 wrap=True, linespacing=1.5)
    ax_info.text(0.5, 0.85,
                 f"Слайд {idx + 1} / {N}",
                 ha="center", va="center", color="#555",
                 fontsize=8, transform=ax_info.transAxes)

    # ── титульный слайд ───────────────────────────────────────────────────────
    if path is None:
        ax_img.set_facecolor(BG)
        # большой заголовок
        ax_img.text(0.5, 0.62, title,
                    ha="center", va="center", color=WHITE,
                    fontsize=32, fontweight="bold",
                    transform=ax_img.transAxes)
        ax_img.text(0.5, 0.42, desc,
                    ha="center", va="center", color=GRAY,
                    fontsize=14, transform=ax_img.transAxes,
                    linespacing=1.8)
        # декоративная линия
        ax_img.axhline(0.55, color=BLUE, linewidth=2, xmin=0.2, xmax=0.8)
        # подсказка
        ax_img.text(0.5, 0.18,
                    "→ / Space / клик «Вперёд» — следующий слайд\n"
                    "← / Backspace — предыдущий\n"
                    "Q / Esc — выход   |   цифры 1–9 — прыжок к слайду",
                    ha="center", va="center", color="#444",
                    fontsize=10, transform=ax_img.transAxes, linespacing=1.7)
    else:
        # ── изображение ───────────────────────────────────────────────────────
        if not Path(path).exists():
            ax_img.text(0.5, 0.5,
                        f"Файл не найден:\n{path}",
                        ha="center", va="center", color="#C44E52",
                        fontsize=12, transform=ax_img.transAxes)
        else:
            img = mpimg.imread(str(path))
            ax_img.imshow(img, aspect="equal")

        ax_img.set_title(title, color=WHITE, fontsize=14,
                         fontweight="bold", pad=10,
                         bbox=dict(facecolor=PANEL, edgecolor="none",
                                   boxstyle="round,pad=0.4"))

    fig.canvas.draw_idle()


def go(delta):
    state["idx"] = (state["idx"] + delta) % N
    render(state["idx"])

def on_key(event):
    if event.key in ("right", " ", "enter"):
        go(+1)
    elif event.key in ("left", "backspace"):
        go(-1)
    elif event.key in ("q", "escape"):
        plt.close("all"); sys.exit(0)
    elif event.key and event.key.isdigit():
        n = int(event.key)
        if 1 <= n <= N:
            state["idx"] = n - 1
            render(state["idx"])

btn_prev.on_clicked(lambda _: go(-1))
btn_next.on_clicked(lambda _: go(+1))
fig.canvas.mpl_connect("key_press_event", on_key)

render(0)
plt.show()
