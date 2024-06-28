import cv2
import numpy as np

# Read grayscale video
cap = cv2.VideoCapture('grayscale_video.avi')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply Sobel filter for edge enhancement
    sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=7)
    sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=7)
    sharpened_frame = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

    cv2.imshow('7x7 Sobel Mask', sharpened_frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
