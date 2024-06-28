import cv2

# Read grayscale video
cap = cv2.VideoCapture('grayscale_video.avi')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply histogram equalization
    equalized_frame = cv2.equalizeHist(frame)

    cv2.imshow('Histogram Equalization', equalized_frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
