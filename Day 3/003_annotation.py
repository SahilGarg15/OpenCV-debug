import cv2
import numpy as np

# Create a blank white canvas
canvas = np.ones((512, 512, 3), dtype=np.uint8)

cv2.line(canvas, (50, 50), (510, 510), (255, 0, 0), thickness=5)  # Blue line
cv2.rectangle(canvas, (100, 100), (400, 400), (0, 0, 255), thickness=3)  # Green rectangle

# Show the image
cv2.imshow('Canvas', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()