import numpy as np
from collections import defaultdict

class SafetyAnalyzer:
    def __init__(self, iou_on_body=0.3, distance_threshold=50, iou_in_hands=0.05, grace_frames=30):
        self.iou_on_body = iou_on_body
        self.distance_threshold = distance_threshold  # ← НОВЫЙ ПАРАМЕТР
        self.iou_in_hands = iou_in_hands
        self.grace_frames = grace_frames
        self.person_buffer = defaultdict(lambda: {'frames_holding': 0, 'violation_issued': False})
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0
    
    def calculate_center_distance(self, box1, box2):
        """Расстояние между центрами bounding boxes"""
        center1_x = (box1[0] + box1[2]) / 2
        center1_y = (box1[1] + box1[3]) / 2
        center2_x = (box2[0] + box2[2]) / 2
        center2_y = (box2[1] + box2[3]) / 2
        return np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
    
    def is_ppe_on_person(self, person_bbox, ppe_bbox):
        """
        Определяет, принадлежит ли предмет СИЗ человеку
        Использует КОМБИНАЦИЮ методов:
        1. IoU (если есть пересечение)
        2. Расстояние между центрами (если близко)
        """
        iou = self.calculate_iou(person_bbox, ppe_bbox)
        
        # Если есть пересечение - точно принадлежит
        if iou > self.iou_on_body:
            return True
        
        # Если нет пересечения, но близко - тоже принадлежит
        distance = self.calculate_center_distance(person_bbox, ppe_bbox)
        person_height = person_bbox[3] - person_bbox[1]
        
        # Если центр предмета в пределах половины высоты человека
        if distance < person_height * 0.5:
            return True
        
        return False
    
    def analyze_person(self, person_bbox, detections, person_id=None):
        """Analyze PPE status for a single person"""
        has_helmet = False
        vest_state = 'not_found'
        
        for det in detections:
            # Используем улучшенную проверку принадлежности
            is_on_person = self.is_ppe_on_person(person_bbox, det['bbox'])
            
            if det['class'] == 'helmet' and is_on_person:
                has_helmet = True
            elif det['class'] == 'no_helmet' and is_on_person:
                has_helmet = False
            elif det['class'] == 'vest':
                iou = self.calculate_iou(person_bbox, det['bbox'])
                if is_on_person:
                    vest_state = 'worn'
                elif iou > self.iou_in_hands:
                    vest_state = 'in_hands'
        
        # Определяем статус
        if has_helmet and vest_state == 'worn':
            return {
                'status': 'safe',
                'message': '[OK] Safe: helmet + vest',
                'color': (0, 255, 0),
                'severity': 'none'
            }
        elif has_helmet and vest_state != 'worn':
            return {
                'status': 'no_vest',
                'message': '[!] NO VEST!',
                'color': (0, 255, 255),
                'severity': 'medium'
            }
        elif not has_helmet and vest_state == 'worn':
            return {
                'status': 'no_helmet',
                'message': '[!] NO HELMET!',
                'color': (0, 165, 255),
                'severity': 'high'
            }
        else:
            return {
                'status': 'no_ppe',
                'message': '[!!] NO HELMET + NO VEST!',
                'color': (0, 0, 255),
                'severity': 'critical'
            }