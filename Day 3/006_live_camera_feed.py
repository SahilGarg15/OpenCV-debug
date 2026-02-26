import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Capture my webcam
capture = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = capture.read()

    if not ret:
        break

    results = model(frame)

    # Draw bounding boxes and labels on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow('Live Camera Feed', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()