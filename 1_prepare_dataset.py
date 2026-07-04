import os
import zipfile
import urllib.request
import shutil
from pathlib import Path
import yaml
from tqdm import tqdm

def download_and_filter():
    # Load config
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    dataset_name = config['dataset']['name']
    url = config['dataset']['url']
    keep_classes = {int(k): v for k, v in config['dataset']['keep_classes'].items()}
    
    data_dir = Path('datasets') / dataset_name
    data_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = data_dir / f"{dataset_name}.zip"
    
    # Download with progress bar
    print("📥 Downloading dataset...")
    urllib.request.urlretrieve(url, zip_path)
    
    print("📦 Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    # Move contents up
    extracted = data_dir / dataset_name
    if extracted.exists():
        for item in extracted.iterdir():
            shutil.move(str(item), str(data_dir))
        extracted.rmdir()
    
    zip_path.unlink()
    
    # Filter annotations
    print("🔄 Filtering annotations (keeping only helmet, vest, person, no_helmet)...")
    for split in ['train', 'val', 'test']:
        labels_dir = data_dir / 'labels' / split
        if not labels_dir.exists():
            continue
        
        for label_file in labels_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                old_class = int(parts[0])
                if old_class in keep_classes:
                    new_class = list(keep_classes.keys()).index(old_class)
                    new_lines.append(f"{new_class} " + " ".join(parts[1:]))
            
            if new_lines:
                with open(label_file, 'w') as f:
                    f.write("\n".join(new_lines))
            else:
                # Remove empty annotation and corresponding image
                label_file.unlink()
                img_file = label_file.parent.parent.parent / 'images' / split / (label_file.stem + '.jpg')
                if img_file.exists():
                    img_file.unlink()
    
    # Create YAML for YOLO
    yaml_content = f"""
path: {data_dir.absolute()}
train: images/train
val: images/val
test: images/test

names:
  0: helmet
  1: vest
  2: person
  3: no_helmet
"""
    with open(data_dir / 'ppe.yaml', 'w') as f:
        f.write(yaml_content)
    
    print("✅ Dataset ready! Classes:")
    for new_id, old_name in keep_classes.items():
        print(f"  {new_id}: {old_name}")

if __name__ == "__main__":
    download_and_filter()