import cv2
import numpy as np
from matplotlib import pyplot as plt



def wiener_filter(img, kernel_size, noise_var):
    # Estimate noise variance
    k = np.mean(np.square(img)) / noise_var

    # Wiener filter
    denoised_img = cv2.fastNlMeansDenoising(img, None, h=k)

    return denoised_img

# Load contaminated image
contaminated_img = cv2.imread(r'D:\4-2\EEE F435 DIP\Lectures\contaminated.JPG')

# Convert to grayscale
gray_img = cv2.cvtColor(contaminated_img, cv2.COLOR_BGR2GRAY)

# Estimate noise variance
noise_var = 25  # You may need to adjust this value based on the actual noise level

# Apply Wiener filter
filtered_img = wiener_filter(gray_img, 9, noise_var)  # Adjust kernel size as needed

# Display original and filtered images


cv2.imshow('Original Image', gray_img)
cv2.imshow('Filtered Image', filtered_img)
cv2.imwrite("Lenna_gaussian_filtered.jpg",filtered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# plt.figure(figsize=(10, 10))

# plt.subplot(1, 2, 1)
# plt.imshow(contaminated_img, cmap='gray')
# plt.title('Original Image')

# plt.subplot(1, 2, 2)
# plt.imshow(filtered_img, cmap='gray')
# plt.title('Filtered Image')

# plt.tight_layout()
# plt.show()
