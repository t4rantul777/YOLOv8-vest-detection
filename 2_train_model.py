from ultralytics import YOLO
import yaml
import torch
from pathlib import Path
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

def train_model():
    # Load config
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    train_cfg = config['training']
    dataset_yaml = Path('datasets') / config['dataset']['name'] / 'ppe.yaml'
    
    device = train_cfg['device'] if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Training on device: {device}")
    print(f"📊 Model: {train_cfg['model']}")
    print(f"📈 Epochs: {train_cfg['epochs']}")
    
    # Load model
    model = YOLO(train_cfg['model'])
    
    # Train (YOLO automatically saves results.png with all graphs!)
    results = model.train(
        data=str(dataset_yaml),
        epochs=train_cfg['epochs'],
        batch=train_cfg['batch_size'],
        imgsz=train_cfg['img_size'],
        patience=train_cfg['patience'],
        device=device,
        project='runs/train',
        name='ppe_lite',
        exist_ok=True,
        pretrained=True,
        lr0=0.01,
        verbose=True
    )
    
    print("\n✅ Training completed!")
    
    # Show built-in graphs
    results_img = Path('runs/train/ppe_lite/results.png')
    if results_img.exists():
        print(f"📊 Graphs saved to: {results_img}")
        img = Image.open(results_img)
        plt.figure(figsize=(16, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.title('YOLO Training Results (mAP50, Losses, Precision, Recall)', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    # Save metrics to CSV
    metrics = {
        'mAP50': results.results_dict.get('metrics/mAP50(B)', 0),
        'mAP50-95': results.results_dict.get('metrics/mAP50-95(B)', 0),
        'precision': results.results_dict.get('metrics/precision(B)', 0),
        'recall': results.results_dict.get('metrics/recall(B)', 0),
    }
    pd.DataFrame([metrics]).to_csv('training_metrics.csv', index=False)
    
    print("\n📈 Final metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print(f"\n🏆 Best model saved to: runs/train/ppe_lite/weights/best.pt")
    
    return results

def view_training_data():
    """Simple function to view training metrics"""
    csv_path = Path('runs/train/ppe_lite/results.csv')
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print("\n" + "="*60)
        print("📊 LAST 5 EPOCHS")
        print("="*60)
        cols = ['epoch', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 
                'metrics/precision(B)', 'metrics/recall(B)']
        print(df[cols].tail().to_string(index=False))
        
        best_row = df.loc[df['metrics/mAP50(B)'].idxmax()]
        print(f"\n🏆 Best epoch: {int(best_row['epoch'])}")
        print(f"   mAP50: {best_row['metrics/mAP50(B)']:.4f}")

if __name__ == "__main__":
    train_model()
    view_training_data()