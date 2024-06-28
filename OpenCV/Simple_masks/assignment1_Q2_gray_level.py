import cv2
import numpy as np


# Function for gray level slicing
def gray_level_slice(img, min_intensity, max_intensity):
    # Create a mask where intensity values are within the specified range
    mask = cv2.inRange(img, min_intensity, max_intensity)
    # Apply the mask to the original image
    result = cv2.bitwise_and(img, img, mask=mask)
    return result

def main(input_video_path, output_video_path, min_intensity, max_intensity):
    capt = cv2.VideoCapture(input_video_path)
    
    if not capt.isOpened():
        print("Error: Could not open video.")
        return
    
    fps = capt.get(cv2.CAP_PROP_FPS)
    frame_width = int(capt.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capt.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capt.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), isColor=False)
    
    while capt.isOpened():
        ret, frame = capt.read()
        
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        sliced_frame = gray_level_slice(gray_frame, min_intensity, max_intensity)
        
        out.write(sliced_frame)
        
    capt.release()
    out.release()
    cv2.destroyAllWindows()


# Main function
if __name__ == "__main__":
    input_video_path = r'D:\4-2\EEE F435 DIP\assignment1_video.mp4'
    output_video_path = "output_video_gray_level.avi"
    min_intensity = 100  # Example: lower bound for gray level slicing
    max_intensity = 200  # Example: upper bound for gray level slicing
    
    main(input_video_path, output_video_path, min_intensity, max_intensity)
