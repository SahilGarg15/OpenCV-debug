import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict, deque

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Read video from file
video_path = 'vid/people_walking.mp4'
cap = cv2.VideoCapture(video_path)

# Create id to track people
id_map = {}
next_id = 0
trail = defaultdict(lambda: deque(maxlen=30))
appearance_count = defaultdict(int)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    res = model(frame, classes=[0], verbose=False)
    annotated_frame = res[0].plot()

    if res[0].boxes.id is not None:
        boxes = res[0].boxes.xyxy.cpu().numpy()
        ids = res[0].boxes.id.cpu().numpy()

        for box, old in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            appearance_count[old] += 1

            if appearance_count[old] > 5 and old not in id_map:
                id_map[old] = next_id
                next_id += 1

            if old in id_map:
                person_id = id_map[old]
                trail[old].append((cx, cy))

                # Draw green trail with fading effect
                points = list(trail[old])
                for i in range(1, len(points)):
                    # Fade: older points are darker, newer points are brighter green
                    alpha = i / len(points)
                    green_intensity = int(100 + 155 * alpha)  # ranges from 100 to 255
                    thickness = max(1, int(3 * alpha))        # ranges from 1 to 3
                    cv2.line(annotated_frame, points[i - 1], points[i],
                             (0, green_intensity, 0), thickness)

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f'ID: {person_id}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(annotated_frame, (cx, cy), 5, (0, 255, 0), -1)

    cv2.imshow('Tracking', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()