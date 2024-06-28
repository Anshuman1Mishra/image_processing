import cv2
import numpy as np

def power_law_transform(img, gamma=1):
    # Apply power law transformation
    result = np.uint8(np.power(img / 255.0, gamma) * 255)
    return result

# Read grayscale video
cap = cv2.VideoCapture('grayscale_video.avi')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply power law transformation with gamma=1.5
    transformed_frame = power_law_transform(frame, gamma=1.5)

    cv2.imshow('Power Law Transformation', transformed_frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
