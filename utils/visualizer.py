import cv2
import numpy as np

def draw_results(image, person_groups):
    """Draw detection results on image"""
    img = image.copy() if isinstance(image, np.ndarray) else np.array(image)
    
    for group in person_groups:
        if not group.get('status'):
            continue
            
        x1, y1, x2, y2 = group['person_bbox']
        status = group['status']
        color = status['color']
        
        # Bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        # Label
        label = status['message']
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, y1 - h - 15), (x1 + w + 10, y1), color, -1)
        cv2.putText(img, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        # PPE icons
        y_offset = y1 + 35
        for det in group['detections']:
            if det['class'] == 'helmet':
                cv2.putText(img, "[+] HELMET", (x2 + 15, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                y_offset += 35
            elif det['class'] == 'vest':
                cv2.putText(img, "[+] VEST", (x2 + 15, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                y_offset += 35
            elif det['class'] == 'no_helmet':
                cv2.putText(img, "[X] NO HELMET", (x2 + 15, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                y_offset += 35
    
    return img