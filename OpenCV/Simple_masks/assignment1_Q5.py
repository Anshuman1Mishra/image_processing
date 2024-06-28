import cv2
import numpy as np


# Function defined for Fourier transform 
def apply_dft(img):
    f_img = np.fft.fft2(img)
    fshift = np.fft.fftshift(f_img)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    return magnitude_spectrum


# Function for video prcessing
def process_video(input_video_path, output_video_path):
    capt = cv2.VideoCapture(input_video_path)
    if not capt.isOpened():
        print("Error: Could not open video.")
        return
    
    # Video properties
    fps = capt.get(cv2.CAP_PROP_FPS)
    frame_width = int(capt.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capt.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), isColor=False)
    
    while capt.isOpened():
        ret, frame = capt.read()
        if not ret:
            break
        
        # Converting to gray
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dft_frame = apply_dft(gray_frame)
        
        # Normalize the DFT frame for visualization
        dft_frame = cv2.normalize(dft_frame, None, 0, 255, cv2.NORM_MINMAX)
        dft_frame = dft_frame.astype(np.uint8)
        
        out.write(dft_frame)
    
    capt.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video_path = r'D:\4-2\EEE F435 DIP\assignment1_video.mp4'
    output_video_path = "output_video_fourier_Q5.avi"
    
    process_video(input_video_path, output_video_path)
