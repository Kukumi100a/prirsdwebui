import numpy as np
from scipy.ndimage import sobel
import cv2

# Define operations
def grayscaleMPI(image_chunk):
    gray = np.zeros_like(image_chunk[:, :, 0])
    for i in range(image_chunk.shape[0]):
        for j in range(image_chunk.shape[1]):
            gray[i, j] = 0.21 * image_chunk[i, j, 0] + 0.72 * image_chunk[i, j, 1] + 0.07 * image_chunk[i, j, 2]
    return gray

def sepiaMPI(image_chunk):
    sepia_filter = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
    sepia_image = np.zeros_like(image_chunk, dtype=np.float64)
    for i in range(image_chunk.shape[0]):
        for j in range(image_chunk.shape[1]):
            pixel = image_chunk[i, j].astype(np.float64)
            sepia_pixel = sepia_filter @ pixel
            sepia_image[i, j] = np.clip(sepia_pixel, 0, 255)
    return sepia_image.astype(np.uint8)

def blurMPI(image_chunk):
    blurred_image = cv2.GaussianBlur(image_chunk, (11, 11), 0)
    return blurred_image

def contrastMPI(image_chunk, alpha=1.5, beta=0):
    adjusted = np.zeros_like(image_chunk)
    for i in range(image_chunk.shape[0]):
        for j in range(image_chunk.shape[1]):
            adjusted[i, j] = np.clip(alpha * image_chunk[i, j] + beta, 0, 255)
    return adjusted

def edge_detectionMPI(image_chunk):
    # Convert image to grayscale
    gray_image = np.dot(image_chunk[...,:3], [0.2989, 0.5870, 0.1140])
    
    # Apply Sobel filter to detect edges
    edges_x = sobel(gray_image, axis=0, mode='reflect')
    edges_y = sobel(gray_image, axis=1, mode='reflect')
    
    # Combine edge images using gradient magnitude
    edges = np.sqrt(edges_x**2 + edges_y**2)
    
    return edges.astype(np.uint8)