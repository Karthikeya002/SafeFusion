"""DeepSORT Tracker for SafeFusion."""

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

class DeepSORTTracker:
    """DeepSORT tracker for multi-object tracking."""
    
    def __init__(self, max_age=70, n_init=3, max_iou_distance=0.7):
        self.tracker = DeepSort(max_age=max_age, n_init=n_init, max_iou_distance=max_iou_distance)
        
    def update(self, detections, frame):
        """Update tracker with new detections."""
        tracks = self.tracker.update_tracks(detections, frame=frame)
        tracked_objects = []
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            tracked_objects.append({
                'track_id': track_id,
                'bbox': ltrb,
                'class': track.get_det_class()
            })
        
        return tracked_objects
