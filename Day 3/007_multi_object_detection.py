import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt') 

# Read from video file
video_path = 'vid/street.mp4'  # Replace with your video file path
capture = cv2.VideoCapture(video_path)

while True:
    ret, frame = capture.read()
    if not ret:
        break

    results = model(frame, classes=[2])  

    # Draw bounding boxes and labels on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow('Multi Object Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()