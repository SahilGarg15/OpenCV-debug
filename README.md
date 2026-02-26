# Module 3: Computer Vision and Image Processing
**B-Tech AI Specialization | Chitkara University | February 2026**

---

## Completed Labs

### Day 1 — Debugging and Foundations
- **Lab 1.3: Debug Challenge** — Identified and fixed bugs in computer vision code covering image loading, preprocessing, and basic OpenCV operations.

### Day 2 — Deep Learning for Computer Vision
- **Lab 2.1: CNN Feature Visualisation (Interactive)** — Explored how convolutional neural networks learn visual features by visualising feature maps across layers using pneumonia X-ray images.
- **Lab 2.2: Transfer Learning with ResNet50** — Fine-tuned a pre-trained ResNet50 on a flower classification dataset using two-phase transfer learning (feature extraction → fine-tuning). Achieved ≥80% validation accuracy.

### Day 3 — Object Detection with YOLO
- **Lab 3.1: Pre-trained Object Detection with YOLO** — Used YOLOv8n to detect and count people in real-time from a webcam feed. Achieved 12.1 FPS (target: ≥10 FPS). Implemented bounding box annotation, live people count, and FPS overlay.
  - Key scripts: `006_live_camera_feed.py`, `009_object_counting.py`, `010_people_with_trail.py`, `lab3_1_people_detection.py`

---

## Setup

```bash
# Activate virtual environment
.venv\Scripts\activate       # Windows

# Run Day 3 people detection
cd "Day 3"
python lab3_1_people_detection.py           # video file
python lab3_1_people_detection.py --webcam  # webcam
```

## Dependencies
- Python 3.14
- `opencv-python`, `ultralytics`, `tensorflow`, `numpy`, `matplotlib`
