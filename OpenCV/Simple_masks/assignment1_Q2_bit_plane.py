import cv2
import numpy as np

def bit_plane_slice(img, bit_plane):
    return (img >> bit_plane) & 1

def main(input_video_path, output_video_path, bit_plane):
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
        
        sliced_frame = bit_plane_slice(gray_frame, bit_plane)
        
        # Convert back to 8-bit for visualization
        sliced_frame = (sliced_frame * 255).astype(np.uint8)
        
        out.write(sliced_frame)
        
    capt.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video_path = r'D:\4-2\EEE F435 DIP\assignment1_video.mp4'
    output_video_path = "output_video_bit_plane.avi"
    bit_plane = 7  # Example: use the 7th bit (0 to 7) for bit-plane slicing
    
    main(input_video_path, output_video_path, bit_plane)
