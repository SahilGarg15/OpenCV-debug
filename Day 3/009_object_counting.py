import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO('yolov8n.pt')

capture = cv2.VideoCapture("vid/bottles.mp4")

unique_objects = set()

while True:
    ret, frame = capture.read()
    
    results = model.track(frame, classes=[39], persist=True, verbose=False)  # Track only bottles (class 39)
    annotated_frame = results[0].plot()

    if results[0].boxes and results[0].boxes.id is not None:
        ids = results[0].boxes.id.numpy()
        for obj_id in ids:
            unique_objects.add(obj_id)
        cv2.putText(annotated_frame, f'Count: {len(unique_objects)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Object Counting', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break   

capture.release()
cv2.destroyAllWindows()