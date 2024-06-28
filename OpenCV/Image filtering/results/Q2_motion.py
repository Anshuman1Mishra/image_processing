import cv2
import numpy as np
from scipy.signal import convolve2d

def wiener_filter(img, kernel, K):
    kernel /= np.sum(kernel)
    dummy = np.copy(img)
    dummy = np.fft.fft2(dummy)
    kernel = np.fft.fft2(kernel, s = img.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    dummy = dummy * kernel
    dummy = np.abs(np.fft.ifft2(dummy))
    return dummy

def blur(img, kernel):
    return convolve2d(img, kernel, mode='same')

def horizontal_motion_blur(img):
    size = 15
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    return blur(img, kernel_motion_blur)

def vertical_motion_blur(img):
    size = 15
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[:, int((size-1)/2)] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    return blur(img, kernel_motion_blur)

# Load the images
horizontal_img = cv2.imread(r'd:\4-2\EEE F435 DIP\Lectures\Lenna_horizontal.jpg', 0)
vertical_img = cv2.imread(r'd:\4-2\EEE F435 DIP\Lectures\Lenna_vertical.jpg', 0)

# Apply the motion blur
horizontal_blurred = horizontal_motion_blur(horizontal_img)
vertical_blurred = vertical_motion_blur(vertical_img)

# Apply the Weiner filter
K = 10
horizontal_deblurred = wiener_filter(horizontal_blurred, horizontal_blurred, K)
vertical_deblurred = wiener_filter(vertical_blurred, vertical_blurred, K)

# Save the deblurred images
cv2.imwrite('horizontal_deblurred.jpg', horizontal_deblurred)
cv2.imwrite('vertical_deblurred.jpg', vertical_deblurred)
