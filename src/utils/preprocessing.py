"""Preprocessing utilities for SafeFusion."""

import cv2
import numpy as np

def preprocess_frame(frame, target_size=(640, 640)):
    """Preprocess frame for model input."""
    resized = cv2.resize(frame, target_size)
    normalized = resized / 255.0
    return normalized

def extract_features(detections, frame_shape):
    """Extract features from detections."""
    features = []
    for det in detections:
        bbox = det['bbox']
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        area = width * height
        aspect_ratio = width / height if height > 0 else 0
        features.append([center_x, center_y, width, height, area, aspect_ratio])
    return np.array(features) if features else np.array([])
