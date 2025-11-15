# Data Directory

This directory contains datasets for training and evaluation of SafeFusion.

## Dataset Structure

Organize your traffic surveillance datasets as follows:

```
data/
├── train/
│   ├── videos/
│   │   ├── video001.mp4
│   │   ├── video002.mp4
│   │   └── ...
│   └── annotations/
│       ├── video001.json
│       ├── video002.json
│       └── ...
├── val/
│   ├── videos/
│   └── annotations/
└── test/
    ├── videos/
    └── annotations/
```

## Annotation Format

Annotations should be in JSON format:

```json
{
  "video": "video001.mp4",
  "frames": [
    {
      "frame_id": 1,
      "objects": [
        {
          "bbox": [x1, y1, x2, y2],
          "class": "car",
          "track_id": 1
        }
      ],
      "accident": false
    }
  ]
}
```

## Recommended Datasets

- **CCTV Traffic Dataset**: Urban intersection surveillance footage
- **UA-DETRAC**: Vehicle detection and tracking dataset
- **Custom Dataset**: Record from your own traffic cameras

## Data Preprocessing

Before training, preprocess your data:

```bash
python scripts/preprocess_data.py --input data/raw --output data/processed
```

## Download Sample Data

```bash
python scripts/download_sample_data.py
```

This will download a small sample dataset for testing.
