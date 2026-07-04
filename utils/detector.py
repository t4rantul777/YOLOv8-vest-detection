from ultralytics import YOLO
import torch
from pathlib import Path

class PPEDetector:
    def __init__(self, model_path=None, conf_threshold=0.15):
        if model_path is None:
            model_path = 'runs/detect/runs/train/ppe_lite/weights/best.pt'
        
        if not Path(model_path).exists():
            print(f"❌ Модель не найдена: {model_path}")
            alt_paths = ['runs/detect/runs/train/ppe_lite/weights/best.pt', 'runs/train/ppe_lite/weights/best.pt']
            for alt in alt_paths:
                if Path(alt).exists():
                    model_path = alt
                    break

        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.conf_threshold = conf_threshold
        self.class_names = self.model.names
        print(f"🔍 Классы модели: {self.class_names}")
        print(f"✅ Модель загружена")
    
    def detect(self, image):
        print(f"🔍 Детекция (агрессивный режим) shape={image.shape}")
        
        # Максимально чувствительный режим
        results = self.model(image, 
                           conf=0.45,           
                           iou=0.40,
                           device=self.device, 
                           verbose=False,
                           augment=True,
                           imgsz=640,
                           max_det=30)[0]

        detections = []
        print(f"✅ Найдено объектов всего: {len(results.boxes) if results.boxes is not None else 0}")
        
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = self.class_names.get(cls, f'unknown_{cls}')
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class_id': cls,
                    'class': class_name
                })
                
                print(f"   → {class_name} ({conf:.3f}) bbox=[{x1},{y1},{x2},{y2}]")
        
        return detections