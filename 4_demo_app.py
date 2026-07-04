import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from pathlib import Path
import yaml

from utils import PPEDetector, SafetyAnalyzer, draw_results

st.set_page_config(page_title="PPE Safety Monitor", layout="wide")

with open('config.yaml') as f:
    config = yaml.safe_load(f)

@st.cache_resource
def load_models():
    detector = PPEDetector(
        model_path=config['inference'].get('model_path'),
        conf_threshold=config['inference'].get('conf_threshold', 0.15)
    )
    
    analyzer = SafetyAnalyzer(
        iou_on_body=config['inference'].get('iou_on_body', 0.12),
        iou_in_hands=config['inference'].get('iou_in_hands', 0.05)
    )
    return detector, analyzer

def group_detections(detections, analyzer):
    persons = [d for d in detections if d['class'] == 'person']
    ppe_items = [d for d in detections if d['class'] != 'person']
    
    groups = []
    for p in persons:
        assigned = [item for item in ppe_items 
                   if analyzer.is_ppe_on_person(p['bbox'], item['bbox'])]
        groups.append({
            'person_bbox': p['bbox'],
            'detections': assigned,
            'status': None
        })
    return groups

st.title("⛑️ Construction PPE Safety Monitor")
st.markdown("### Helmet & Vest Detection")

with st.sidebar:
    st.header("⚙️ Информация")
    model_path = config['inference'].get('model_path', 'не указан')
    st.write(f"**Модель:** `{Path(model_path).name if model_path else '—'}`")

tab1, tab2, tab3 = st.tabs(["📸 Image", "🎥 Video", "📊 Graphs"])

with tab1:
    uploaded = st.file_uploader("Upload image", type=['jpg', 'png', 'jpeg'])
    if uploaded:
        image = Image.open(uploaded)
        col1, col2 = st.columns(2)
        col1.image(image, use_container_width=True)
        
        if st.button("Analyze"):
            with st.spinner("Analyzing..."):
                detector, analyzer = load_models()
                
                img_array = np.array(image)
                
                detections = detector.detect(img_array)
                
                st.write(f"**Найдено объектов:** {len(detections)}")
                if detections:
                    for d in detections:
                        st.write(f"- {d['class']}: {d['confidence']:.2f}")
                
                groups = group_detections(detections, analyzer)
                
                for g in groups:
                    g['status'] = analyzer.analyze_person(g['person_bbox'], g['detections'])
                
                result = draw_results(img_array, groups)
                col2.image(result, use_container_width=True)
                
                safe = sum(1 for g in groups if g.get('status', {}).get('status') == 'safe')
                st.metric("Safe / Total", f"{safe}/{len(groups)}")