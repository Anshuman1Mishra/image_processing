import cv2

# Read grayscale video
cap = cv2.VideoCapture('grayscale_video.avi')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply Laplacian filter for edge detection
    laplacian = cv2.Laplacian(frame, cv2.CV_64F, ksize=7)
    sharpened_frame = cv2.addWeighted(frame, 1, laplacian, 1, 0)

    cv2.imshow('7x7 Laplacian Mask', sharpened_frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
