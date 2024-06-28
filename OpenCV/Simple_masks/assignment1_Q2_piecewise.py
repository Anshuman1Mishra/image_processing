import cv2
import numpy as np

def piecewise_linear_transform(img):
    # Define piecewise linear transformation function
    lut = np.zeros(256, dtype=np.uint8)
    lut[0:100] = 0
    lut[100:150] = np.linspace(0, 255, 50)
    lut[150:] = 255

    # Apply transformation
    result = cv2.LUT(img, lut)
    return result

# Read grayscale video
cap = cv2.VideoCapture('grayscale_video.avi')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply piecewise linear transformation
    transformed_frame = piecewise_linear_transform(frame)

    cv2.imshow('Piecewise Linear Transformation', transformed_frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
