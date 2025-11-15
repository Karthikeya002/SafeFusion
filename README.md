# SafeFusion: Intelligent Accident Surveillance System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)

A YOLO-Transformer Hybrid Model for Intelligent Accident Surveillance - Real-time traffic accident detection and prediction system combining state-of-the-art object detection with spatiotemporal analysis.

## 📋 Overview

SafeFusion is a hybrid deep learning framework that combines:
- **YOLOv8** for real-time object detection (vehicles, pedestrians, cyclists)
- **DeepSORT** for robust multi-object tracking
- **Transformer networks** for temporal analysis and accident prediction

### Key Features

- ✅ Real-time accident detection with 92% mAP
- ✅ Low latency: 23.8 ms per frame (~42 FPS)
- ✅ Predictive near-miss warnings
- ✅ Automated emergency alert system
- ✅ Multi-object tracking in complex traffic scenarios
- ✅ Robust performance across varying weather and lighting conditions

## 🏗️ System Architecture

```
┌─────────────────┐
│  Video Input    │
│  (HD Cameras)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Preprocessing   │
│ (Noise, Norm)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    YOLOv8       │
│ Object Detection│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   DeepSORT      │
│ Multi-Tracking  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Transformer    │
│ Temporal Model  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Alert Generation│
│ & Notification  │
└─────────────────┘
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA 11.0+ (for GPU support)
- 8GB+ RAM
- NVIDIA GPU with 4GB+ VRAM (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Karthikeya002/SafeFusion.git
cd SafeFusion
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download pre-trained weights:
```bash
python scripts/download_weights.py
```

### Quick Start

```python
from safefusion import SafeFusion

# Initialize the model
model = SafeFusion(
    yolo_weights='weights/yolov8n.pt',
    transformer_config='configs/transformer.yaml'
)

# Process video
results = model.predict_video('path/to/video.mp4')

# Or use webcam
model.predict_realtime(source=0)
```

## 📁 Project Structure

```
SafeFusion/
├── src/
│   ├── models/
│   │   ├── yolo_detector.py
│   │   ├── deepsort_tracker.py
│   │   └── transformer_temporal.py
│   ├── utils/
│   │   ├── preprocessing.py
│   │   ├── visualization.py
│   │   └── alert_system.py
│   └── safefusion.py
├── configs/
│   ├── model_config.yaml
│   ├── transformer.yaml
│   └── training_config.yaml
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── download_weights.py
│   └── demo.py
├── notebooks/
│   ├── exploration.ipynb
│   └── visualization.ipynb
├── tests/
│   └── test_models.py
├── data/
│   └── README.md
├── weights/
│   └── README.md
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

## 🔧 Configuration

Edit `configs/model_config.yaml` to customize:

```yaml
model:
  yolo:
    model_size: 'yolov8n'  # Options: n, s, m, l, x
    conf_threshold: 0.25
    iou_threshold: 0.45
  
  deepsort:
    max_age: 30
    min_hits: 3
    iou_threshold: 0.3
  
  transformer:
    num_layers: 6
    num_heads: 8
    hidden_dim: 512
    dropout: 0.1
```

## 📊 Performance

### Object Detection Performance

| Object Category | Precision | Recall | mAP@0.5 |
|----------------|-----------|--------|----------|
| Vehicles       | 95.2%     | 90.1%  | 93.0%    |
| Pedestrians    | 92.3%     | 88.4%  | 90.2%    |
| Cyclists       | 89.1%     | 86.7%  | 88.0%    |
| **Overall**    | **92.2%** | **88.4%**| **90.4%** |

### Accident Detection Performance

| Method        | Accuracy | Precision | Recall | Latency (ms) |
|--------------|----------|-----------|--------|-------------|
| YOLO-only    | 75.2%    | 72.8%     | 74.1%  | 10.2        |
| CNN-LSTM     | 80.1%    | 78.9%     | 79.3%  | 180.5       |
| **SafeFusion**| **92.0%**| **91.5%** | **92.3%**| **23.8**   |

### Computational Efficiency

- **YOLOv8 Detection**: 12.1 ms
- **DeepSORT Tracking**: 3.4 ms
- **Transformer Analysis**: 8.3 ms
- **Total Pipeline**: ~42 FPS

## 🎯 Usage Examples

### Basic Video Processing

```python
from safefusion import SafeFusion

model = SafeFusion()
results = model.predict_video(
    source='traffic_video.mp4',
    save_results=True,
    output_path='output/'
)
```

### Real-time Webcam Detection

```python
model = SafeFusion()
model.predict_realtime(
    source=0,  # Webcam index
    show_video=True,
    alert_callback=send_alert
)
```

### Custom Alert System

```python
def custom_alert_handler(alert_data):
    print(f"Accident detected at {alert_data['timestamp']}")
    print(f"Location: {alert_data['location']}")
    print(f"Confidence: {alert_data['confidence']:.2%}")
    # Send notification, log to database, etc.

model = SafeFusion()
model.set_alert_callback(custom_alert_handler)
model.predict_video('input.mp4')
```

## 🔬 Training

### Prepare Dataset

1. Organize your dataset:
```
data/
├── train/
│   ├── videos/
│   └── annotations/
└── val/
    ├── videos/
    └── annotations/
```

2. Train the model:
```bash
python scripts/train.py --config configs/training_config.yaml
```

### Fine-tuning

```bash
python scripts/train.py \
  --weights weights/safefusion_base.pt \
  --epochs 50 \
  --batch-size 32 \
  --data data/custom_dataset.yaml
```

## 📈 Evaluation

```bash
python scripts/evaluate.py \
  --weights weights/best.pt \
  --data data/test_dataset.yaml \
  --task test
```

## 🌟 Key Components

### 1. YOLOv8 Object Detector
- Anchor-free detection
- Cross-Stage Partial connections
- Path Aggregation Network
- Multi-class detection (vehicles, pedestrians, cyclists)

### 2. DeepSORT Tracker
- Kalman filtering for motion prediction
- Deep appearance descriptors
- Hungarian algorithm for data association
- Handles occlusions and identity switches

### 3. Transformer Temporal Model
- Multi-head self-attention mechanism
- Positional encoding for temporal sequences
- Long-range dependency modeling
- Accident prediction and near-miss detection

## 🎓 Citation

If you use SafeFusion in your research, please cite:

```bibtex
@article{safefusion2024,
  title={SafeFusion: A YOLO-Transformer Hybrid Model for Intelligent Accident Surveillance},
  author={Kalaichelvi, T. and Alekhya, D. and Karthikeya, K. and Ramakrishna, V. S.},
  journal={Conference Proceedings},
  year={2024}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- DeepSORT implementation
- Transformer architecture from "Attention is All You Need"
- Traffic accident datasets and benchmarks

## 📧 Contact

- **Project Maintainer**: Karthikeya
- **Email**: vtu22893@veltech.edu.in
- **Institution**: Vel Tech Rangarajan Dr.Sagunthala R&D Institute of Science and Technology, Chennai

## 🔗 Links

- [Research Paper](docs/SafeFusion_Paper.pdf)
- [Documentation](https://github.com/Karthikeya002/SafeFusion/wiki)
- [Demo Videos](https://github.com/Karthikeya002/SafeFusion/tree/main/demo)

## ⚠️ Disclaimer

This system is designed as a research prototype for traffic safety applications. Always follow local regulations and guidelines when deploying surveillance systems. The authors are not responsible for misuse of this technology.

---

**Made with ❤️ for safer roads**
