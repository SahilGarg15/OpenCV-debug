import cv2

# Read the image
image = cv2.imread('img/image.jpeg')

# Resize the image
resized_image = cv2.resize(image, (300, 300))

# Convert the image to grayscale
gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Detect edges using Canny edge detection
edges = cv2.Canny(blurred_image, 100, 200)

# Display the original, resized, grayscale, blurred, and edge-detected images
cv2.imshow('Original Image', image)
cv2.imshow('Resized Image', resized_image)
cv2.imshow('Grayscale Image', gray_image)
cv2
cv2.imshow('Edge Detected Image', edges)

cv2.waitKey(0)
cv2.destroyAllWindows()