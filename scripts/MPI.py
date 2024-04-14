import numpy as np
from scipy.ndimage import sobel
import cv2

# Define operations
def grayscaleMPI(image_chunk):
    gray = np.dot(image_chunk[..., :3], [0.2989, 0.5870, 0.1140])
    return gray.astype(np.uint8)

def sepiaMPI(image_chunk):
    sepia_filter = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
    sepia_image = np.dot(image_chunk, sepia_filter.T).clip(0, 255)
    return sepia_image.astype(np.uint8)

def blurMPI(image_chunk):
    blurred_image = cv2.GaussianBlur(image_chunk, (11, 11), 0)
    return blurred_image

def contrastMPI(image_chunk, alpha=1.5, beta=0):
    adjusted = np.clip(alpha * image_chunk + beta, 0, 255)
    return adjusted.astype(np.uint8)

def edge_detectionMPI(image_chunk):
    gray_image = np.dot(image_chunk[..., :3], [0.2989, 0.5870, 0.1140])
    edges_x = sobel(gray_image, axis=0, mode='reflect')
    edges_y = sobel(gray_image, axis=1, mode='reflect')
    edges = np.sqrt(edges_x**2 + edges_y**2)
    return edges.astype(np.uint8)