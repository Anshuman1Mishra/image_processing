import cv2

# Read grayscale video
cap = cv2.VideoCapture('grayscale_video.avi')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply 9x9 Gaussian mask
    smoothed_frame = cv2.GaussianBlur(frame, (9, 9), 0)

    cv2.imshow('9x9 Gaussian Mask', smoothed_frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
