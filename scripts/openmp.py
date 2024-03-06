import numpy as np
import numba as nb
from numba import njit, prange, jit
import multiprocessing

@njit(parallel=True)
def grayscale(image):
    gray = np.zeros_like(image[:, :, 0])  # Assuming image is in BGR format
    for i in prange(image.shape[0]):
        for j in range(image.shape[1]):
            # Convert each pixel to grayscale using luminosity method
            gray[i, j] = 0.21 * image[i, j, 0] + 0.72 * image[i, j, 1] + 0.07 * image[i, j, 2]
    return gray

####################

@njit(parallel=True)
def sepia(image):
    sepia_filter = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
    sepia_image = np.zeros_like(image, dtype=np.float64)
    for i in prange(image.shape[0]):
        for j in range(image.shape[1]):
            pixel = image[i, j].astype(np.float64)
            sepia_pixel = sepia_filter @ pixel
            sepia_image[i, j] = np.clip(sepia_pixel, 0, 255)
    return sepia_image.astype(np.uint8)

####################

@njit(parallel=True)
def blur(image):
    height, width, _ = image.shape
    blurred_image = np.zeros_like(image, dtype=np.float32)
    
    # Apply blur to each channel separately
    for c in prange(3):
        for i in prange(2, height - 2):
            for j in prange(2, width - 2):
                blurred_pixel = 0.0
                # Convolution operation with 5x5 kernel
                blurred_pixel += image[i - 2, j - 2, c] + image[i - 2, j - 1, c] + image[i - 2, j, c] + image[i - 2, j + 1, c] + image[i - 2, j + 2, c]
                blurred_pixel += image[i - 1, j - 2, c] + image[i - 1, j - 1, c] + image[i - 1, j, c] + image[i - 1, j + 1, c] + image[i - 1, j + 2, c]
                blurred_pixel += image[i, j - 2, c] + image[i, j - 1, c] + image[i, j, c] + image[i, j + 1, c] + image[i, j + 2, c]
                blurred_pixel += image[i + 1, j - 2, c] + image[i + 1, j - 1, c] + image[i + 1, j, c] + image[i + 1, j + 1, c] + image[i + 1, j + 2, c]
                blurred_pixel += image[i + 2, j - 2, c] + image[i + 2, j - 1, c] + image[i + 2, j, c] + image[i + 2, j + 1, c] + image[i + 2, j + 2, c]
                blurred_pixel /= 25.0  # Divide by total number of pixels in the kernel
                blurred_image[i, j, c] = blurred_pixel
                
    return blurred_image.astype(np.uint8)

####################

@njit(parallel=True)
def contrast(image, alpha=1.5, beta=0):
    adjusted = np.zeros_like(image)
    for i in prange(image.shape[0]):
        for j in range(image.shape[1]):
            adjusted[i, j] = np.clip(alpha * image[i, j] + beta, 0, 255)
    return adjusted

####################

@njit(parallel=True)
def edge_detection(image):
    edges = np.zeros_like(image)
    for i in prange(1, image.shape[0] - 1):
        for j in prange(1, image.shape[1] - 1):
            # Sobel edge detection
            gx = image[i + 1, j - 1] + 2 * image[i + 1, j] + image[i + 1, j + 1] - \
                 image[i - 1, j - 1] - 2 * image[i - 1, j] - image[i - 1, j + 1]
            gy = image[i - 1, j + 1] + 2 * image[i, j + 1] + image[i + 1, j + 1] - \
                 image[i - 1, j - 1] - 2 * image[i, j - 1] - image[i + 1, j - 1]
            edges[i, j] = np.sqrt(gx ** 2 + gy ** 2)
    return edges

@njit(parallel=True)
def process_image(image, operation):
    result = np.zeros_like(image)
    for i in prange(image.shape[0]):
        result[i] = operation(image[i])
    return result