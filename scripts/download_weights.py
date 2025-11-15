#!/usr/bin/env python3
"""Download pre-trained weights for SafeFusion."""

import os
import urllib.request
from tqdm import tqdm

WEIGHTS_DIR = 'weights'
YOLOV8_URLS = {
    'yolov8n.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
    'yolov8s.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt',
    'yolov8m.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt',
}

def download_file(url, filename):
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    filepath = os.path.join(WEIGHTS_DIR, filename)
    if os.path.exists(filepath):
        print(f'{filename} already exists')
        return
    print(f'Downloading {filename}...')
    urllib.request.urlretrieve(url, filepath)
    print(f'Downloaded {filename}')

if __name__ == '__main__':
    for name, url in YOLOV8_URLS.items():
        download_file(url, name)
