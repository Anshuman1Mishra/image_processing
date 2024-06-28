import cv2


def resize_frame(frame, scale_factor, interpolation_method):
    # Resize the frame using the specified interpolation method
    width = int(frame.shape[1]*scale_factor)
    height = int(frame.shape[0]*scale_factor)
    resized_frame = cv2.resize(frame, (width, height), interpolation=interpolation_method)
    return resized_frame



def process_video(input_path, output_path, scale_factor, interpolation_method):

    # Opens the input video file
    input_video = cv2.VideoCapture(input_path)


    # Get video properties
    frame_count = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))


    # Create VideoWriter object to save the output video
    output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    # Process each frame
    for i in range(frame_count):
        ret, frame = input_video.read()
        if ret == 0:
            break

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize the frame using the specified interpolation method
        resized_frame = resize_frame(gray_frame, scale_factor, interpolation_method)

        # Write the processed frame to the output video
        output_video.write(resized_frame)
        
    # Releasing the objects
    input_video.release()
    output_video.release()

    # Getting size of the output video
    output_video = cv2.VideoCapture(output_path)
    output_frame_count = int(output_video.get(cv2.CAP_PROP_FRAME_COUNT))
    output_frame_width = int(output_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_frame_height = int(output_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Output Video Size (Frames): {output_frame_width}x{output_frame_height} (Total Frames: {output_frame_count})")

    # Release VideoCapture object
    output_video.release()


# Main function
if __name__ == "__main__":
    input_video_path = r'D:\4-2\EEE F435 DIP\assignment1_video.mp4'        # Location of video
    output_video_nearest_path = "output_video_nearest_Q1.avi"
    output_video_bilinear_path = "output_video_bilinear_Q1.avi"
    output_video_bicubic_path = "output_video_bicubic_Q1.avi"

    scale_factor = 2  # Increase by a scale factor of 2
    interpolation_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC]

    for method in interpolation_methods:
        print(f"Processing video with interpolation method: {method}")
        if method == cv2.INTER_NEAREST:
            output_video_path = output_video_nearest_path
        elif method == cv2.INTER_LINEAR:
            output_video_path = output_video_bilinear_path
        else:
            output_video_path = output_video_bicubic_path
        process_video(input_video_path, output_video_path, scale_factor, method)



