import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

image = cv2.imread('img/classroom.jpg')

results = model(image)

# Draw bounding boxes and labels on the image
annotated_image = results[0].plot()

# Display the annotated image
cv2.imshow('Object Detection', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()