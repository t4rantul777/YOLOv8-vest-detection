"""
Простые графики обучения
Запуск: python 3_plot_metrics.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Проверяем наличие файла
csv_path = Path('runs/train/ppe_lite/results.csv')
if not csv_path.exists():
    print("❌ Сначала обучите модель: python 2_train_model.py")
    exit()

# Загружаем данные
df = pd.read_csv(csv_path)
print(f"✅ Загружено {len(df)} эпох")

# ========== 1. ГРАФИКИ ПОТЕРЬ ==========
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(df['epoch'], df['train/box_loss'], 'b-', linewidth=2)
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.title('Box Loss (потери локализации)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(df['epoch'], df['train/cls_loss'], 'r-', linewidth=2)
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.title('Cls Loss (потери классификации)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(df['epoch'], df['train/dfl_loss'], 'g-', linewidth=2)
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.title('DFL Loss')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('losses.png', dpi=150)
plt.show()
print("✅ saved: losses.png")

# ========== 2. ГРАФИКИ МЕТРИК ==========
plt.figure(figsize=(15, 10))

# Precision
plt.subplot(2, 2, 1)
plt.plot(df['epoch'], df['metrics/precision(B)'], 'green', linewidth=2)
plt.xlabel('Эпоха')
plt.ylabel('Score')
plt.title('Precision (Точность)')
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)

# Recall
plt.subplot(2, 2, 2)
plt.plot(df['epoch'], df['metrics/recall(B)'], 'blue', linewidth=2)
plt.xlabel('Эпоха')
plt.ylabel('Score')
plt.title('Recall (Полнота)')
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)

# mAP50
plt.subplot(2, 2, 3)
plt.plot(df['epoch'], df['metrics/mAP50(B)'], 'purple', linewidth=2)
plt.xlabel('Эпоха')
plt.ylabel('Score')
plt.title('mAP50 (при IoU=0.5)')
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)

# mAP50-95
plt.subplot(2, 2, 4)
plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], 'orange', linewidth=2)
plt.xlabel('Эпоха')
plt.ylabel('Score')
plt.title('mAP50-95 (среднее)')
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig('metrics.png', dpi=150)
plt.show()
print("✅ saved: metrics.png")

# ========== 3. ВСЕ МЕТРИКИ ВМЕСТЕ ==========
plt.figure(figsize=(12, 5))
plt.plot(df['epoch'], df['metrics/mAP50(B)'], 'b-', linewidth=2, label='mAP50')
plt.plot(df['epoch'], df['metrics/precision(B)'], 'g-', linewidth=2, label='Precision')
plt.plot(df['epoch'], df['metrics/recall(B)'], 'r-', linewidth=2, label='Recall')
plt.xlabel('Эпоха')
plt.ylabel('Score')
plt.title('Все метрики вместе')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('all_metrics.png', dpi=150)
plt.show()
print("✅ saved: all_metrics.png")

# ========== 4. ЛУЧШАЯ МОДЕЛЬ ==========
best_idx = df['metrics/mAP50(B)'].idxmax()
print("\n" + "="*50)
print("🏆 ЛУЧШАЯ МОДЕЛЬ:")
print("="*50)
print(f"  Эпоха:        {int(df.loc[best_idx, 'epoch'])}")
print(f"  mAP50:        {df.loc[best_idx, 'metrics/mAP50(B)']:.4f}")
print(f"  mAP50-95:     {df.loc[best_idx, 'metrics/mAP50-95(B)']:.4f}")
print(f"  Precision:    {df.loc[best_idx, 'metrics/precision(B)']:.4f}")
print(f"  Recall:       {df.loc[best_idx, 'metrics/recall(B)']:.4f}")

print("\n✅ Готово! Графики сохранены.")