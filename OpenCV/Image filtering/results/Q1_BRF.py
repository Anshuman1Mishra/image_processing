import numpy as np
import cv2
from scipy import fftpack
import matplotlib.pyplot as plt


def ideal_bandreject_filter(shape, cutoff_low, cutoff_high):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            if distance <= cutoff_high and distance >= cutoff_low:
                mask[i, j] = 0
    return mask

def gaussian_bandreject_filter(shape, cutoff_low, cutoff_high):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            mask[i, j] = 1 - np.exp(-0.5 * (distance ** 2 / cutoff_high ** 2)) * \
                         (1 - np.exp(-0.5 * (distance ** 2 / cutoff_low ** 2)))
    return mask

def butterworth_bandreject_filter(shape, cutoff_low, cutoff_high, order):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            mask[i, j] = 1 - 1 / (1 + (distance / cutoff_high) ** (2 * order)) * \
                         (1 - 1 / (1 + (distance / cutoff_low) ** (2 * order)))
    return mask

def apply_filter(image, filter):
    fshift = fftpack.fftshift(fftpack.fft2(image))
    fshift_filtered = fshift * filter
    f_filtered = fftpack.ifft2(fftpack.ifftshift(fshift_filtered)).real
    return f_filtered

# Load the image
image = cv2.imread(r"D:\4-2\EEE F435 DIP\Lectures\Lenna.jpg", cv2.IMREAD_GRAYSCALE)

# Define cutoff frequencies
cutoff_low = 30
cutoff_high = 80

# Generate filters
ideal_filter = ideal_bandreject_filter(image.shape, cutoff_low, cutoff_high)
gaussian_filter = gaussian_bandreject_filter(image.shape, cutoff_low, cutoff_high)
butterworth_filter = butterworth_bandreject_filter(image.shape, cutoff_low, cutoff_high, order=2)

# Apply filters
filtered_ideal = apply_filter(image, ideal_filter)
filtered_gaussian = apply_filter(image, gaussian_filter)
filtered_butterworth = apply_filter(image, butterworth_filter)

# Display the results
cv2.imwrite('Lenna_brf_original.jpg',image)
cv2.imwrite('Lenna_brf_ideal.jpg',filtered_ideal)
cv2.imwrite('Lenna_brf_gaussian.jpg',filtered_gaussian)
cv2.imwrite('Lenna_brf_butterworth.jpg',filtered_butterworth)
# plt.figure(figsize=(10, 6))

# plt.subplot(2, 2, 1)
# plt.imshow(image, cmap='gray')
# plt.title('Original Image')

# plt.subplot(2, 2, 2)
# plt.imshow(filtered_ideal, cmap='gray')
# plt.title('Ideal Band reject Filter')

# plt.subplot(2, 2, 3)
# plt.imshow(filtered_gaussian, cmap='gray')
# plt.title('Gaussian Band reject Filter')

# plt.subplot(2, 2, 4)
# plt.imshow(filtered_butterworth, cmap='gray')
# plt.title('Butterworth Ban reject Filter')

# plt.tight_layout()
# plt.show()
