"""
Lab 3.1: Pre-trained Object Detection with YOLO
------------------------------------------------
Retail people counter — detects and counts people in video frames
using YOLOv8n, draws bounding boxes, displays live count and FPS.

Usage:
    python lab3_1_people_detection.py              # uses default video file
    python lab3_1_people_detection.py --webcam     # uses webcam instead
"""

import cv2
import time
import argparse
from ultralytics import YOLO

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--webcam', action='store_true',
                    help='Use webcam instead of video file')
parser.add_argument('--video', type=str, default='vid/people_walking.mp4',
                    help='Path to video file (default: vid/people_walking.mp4)')
args = parser.parse_args()

# ── Task 1 & 2: Load YOLOv8n with pre-trained ImageNet weights ────────────────
print("Loading YOLOv8n model...")
model = YOLO('yolov8n.pt')          # downloads automatically on first run
PERSON_CLASS = 0                    # COCO class 0 = person

# ── Task 3: Open video source ─────────────────────────────────────────────────
source = 0 if args.webcam else args.video
cap = cv2.VideoCapture(source)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video source: {source}")

print(f"Source: {'Webcam' if args.webcam else args.video}")
print("Press 'q' to quit.")

# FPS tracking
fps = 0.0
frame_times = []          # rolling window for smoother FPS

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    t_start = time.perf_counter()

    # Task 3: Read frame
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    # Task 4: Run YOLO inference — filter to person class only
    results = model(frame, classes=[PERSON_CLASS], verbose=False)

    # Task 5: Draw bounding boxes and display people count
    annotated_frame = results[0].plot()

    person_count = len(results[0].boxes)
    cv2.putText(
        annotated_frame,
        f'People: {person_count}',
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2, (0, 255, 0), 3
    )

    # Task 6: Calculate and display FPS (rolling average over last 10 frames)
    t_end = time.perf_counter()
    frame_times.append(t_end - t_start)
    if len(frame_times) > 10:
        frame_times.pop(0)
    fps = 1.0 / (sum(frame_times) / len(frame_times))

    cv2.putText(
        annotated_frame,
        f'FPS: {fps:.1f}',
        (10, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (0, 200, 255), 2
    )

    cv2.imshow('Lab 3.1 — People Detection', annotated_frame)

    # Check FPS target
    if fps >= 10:
        fps_status = f"✅ FPS target met ({fps:.1f} >= 10)"
    else:
        fps_status = f"⚠️ FPS below target ({fps:.1f} < 10)"

    print(f"\r{fps_status} | People in frame: {person_count}  ", end='', flush=True)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nDone. Final FPS: {fps:.1f}")
