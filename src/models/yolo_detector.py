"""YOLO Detector for SafeFusion."""

import torch
import cv2
import numpy as np
from ultralytics import YOLO

class YOLODetector:
    """YOLOv8 object detector for vehicle detection."""
    
    def __init__(self, model_path='weights/yolov8n.pt', conf_threshold=0.25, iou_threshold=0.45, device='cuda'):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
    def detect(self, frame):
        """Detect objects in a frame."""
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, device=self.device)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class': cls
                })
        
        return detections
