import cv2

# Read grayscale video
cap = cv2.VideoCapture('grayscale_video.avi')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply 5x5 smoothing mask
    smoothed_frame = cv2.blur(frame, (5, 5))

    cv2.imshow('5x5 Smoothing Mask', smoothed_frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
