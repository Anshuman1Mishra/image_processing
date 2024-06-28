import cv2
import numpy as np

def bit_plane_slicing(img, bit):
    # Extract the bit-plane by bitwise AND with a mask
    mask = 1 << bit
    result = (img & mask) * 255
    return result

def gray_video(input_path, output_path):

    # Reading the video
    input_video = cv2.VideoCapture(input_path)

    # Video properties
    width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(input_video.get(cv2.CAP_PROP_FPS))


    # Creating a VideoWriter object

    grayscale_video = cv2.VideoWriter(output_path, cv2.VideoWriter_forucc(*'XVID'),fps , (width,height))

    # Processing each frame
    for i in range(frame_count):
        ret,frame = input_video.read()
        if ret == 0:
            break

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Write the processed frame to the output video
        grayscale_video.write(gray_frame)


    # Releasing the objects
    input_video.release()
    grayscale_video.release()

    return grayscale_video





# Main
src = r'D:\4-2\EEE F435 DIP\assignment1_video.mp4'
capt = gray_video(cv2.VideoCapture(src))

while capt.isOpened():
    ret, frame = capt.read()
    if not ret:
        break
    
    # Apply bit-plane slicing for bit=5
    transformed_frame = bit_plane_slicing(frame, 5)

    cv2.imshow('Bit-Plane Slicing', transformed_frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

capt.release()
cv2.destroyAllWindows()
