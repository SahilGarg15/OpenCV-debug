import cv2

# Read the image
image = cv2.imread('img/image.jpeg')

#Display the image
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

