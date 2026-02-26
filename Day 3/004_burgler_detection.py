import cv2
import time

# Capture my webcam
capture = cv2.VideoCapture(0)

ret, frame = capture.read()
bg_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
bg_blurred = cv2.GaussianBlur(bg_gray, (21, 21), 0)

while True:
    # Read a frame from the webcam
    ret, frame = capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    diff = cv2.absdiff(bg_blurred, blurred)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY) 
    thresh = cv2.dilate(thresh, None, iterations=5)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  

    # find the largest contour
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 100:  # Adjust the threshold as needed
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  
            cv2.putText(frame, 'Intruder Detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.imwrite(f'img/intruder_{timestamp}.jpg', frame)  # Save the image of the intruder
            print(f'Intruder detected at {timestamp}')
    
    
    if not ret:
        break
    
    cv2.imshow('Webcam', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()