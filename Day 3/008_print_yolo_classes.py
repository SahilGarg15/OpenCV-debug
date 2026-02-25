from ultralytics import YOLO

# Load a pretrained YOLOv8 model (e.g., yolov8n.pt)
model = YOLO('yolov8n.pt') 

# Print the dictionary of class IDs and names
print(model.names)
