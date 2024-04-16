import numpy as np
import torch
import torch.nn.functional as F

def grayscaleCUDA(image):
    if isinstance(image, np.ndarray):
        # Convert image to tensor and normalize to range [-1, 1]
        image_tensor = torch.from_numpy(image).float()
    elif isinstance(image, torch.Tensor):
        image_tensor = image.float()
    else:
        raise ValueError("Input must be a NumPy array or a PyTorch tensor.")
    
    image_normalized = (image_tensor / 255.0) * 2.0 - 1.0
    # Calculate grayscale using luminance method
    gray = torch.sum(image_normalized * torch.tensor([0.299, 0.587, 0.114]).to(image_tensor.device), dim=2)
    return ((gray + 1.0) * 127.5).cpu().numpy().astype(np.uint8)

def sepiaCUDA(image):
    if isinstance(image, np.ndarray):
        # Convert image to tensor and normalize to range [-1, 1]
        image_tensor = torch.from_numpy(image).float()
    elif isinstance(image, torch.Tensor):
        image_tensor = image.float()
    else:
        raise ValueError("Input must be a NumPy array or a PyTorch tensor.")
    
    image_normalized = (image_tensor / 255.0) * 2.0 - 1.0
    # Define sepia filter
    sepia_filter = torch.tensor([[0.393, 0.769, 0.189],
                                 [0.349, 0.686, 0.168],
                                 [0.272, 0.534, 0.131]]).to(image_tensor.device).float()
    # Apply sepia filter
    sepia_image = torch.matmul(image_normalized, sepia_filter.T)
    sepia_image = torch.clamp(sepia_image, -1, 1)  # Clamp to [-1, 1]
    return ((sepia_image + 1.0) * 127.5).cpu().numpy().astype(np.uint8)  # Denormalize to [0, 255]

def blurCUDA(image):
    if isinstance(image, np.ndarray):
        # Convert image to tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
    elif isinstance(image, torch.Tensor):
        image_tensor = image.permute(2, 0, 1).float().unsqueeze(0)
    else:
        raise ValueError("Input must be a NumPy array or a PyTorch tensor.")
    
    # Apply separable blur along each channel
    blurred_channels = []
    for channel in range(3):
        channel_tensor = image_tensor[:, channel:channel+1, :, :]
        blurred_channel = F.conv2d(channel_tensor, torch.ones(1, 1, 5, 5) / 25.0, padding=2)
        blurred_channels.append(blurred_channel)
    
    blurred_image = torch.cat(blurred_channels, dim=1)
    
    blurred_image = torch.clamp(blurred_image, 0, 255)
    return blurred_image.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

def contrastCUDA(image, alpha=1.5, beta=0):
    if isinstance(image, np.ndarray):
        # Convert image to tensor and normalize to range [-1, 1]
        image_tensor = torch.from_numpy(image).float()
    elif isinstance(image, torch.Tensor):
        image_tensor = image.float()
    else:
        raise ValueError("Input must be a NumPy array or a PyTorch tensor.")
    
    image_normalized = (image_tensor / 255.0) * 2.0 - 1.0
    # Adjust contrast
    adjusted = torch.clamp(alpha * image_normalized + beta, -1, 1)  # Clamp to [-1, 1]
    return ((adjusted + 1.0) * 127.5).cpu().numpy().astype(np.uint8)  # Denormalize to [0, 255]

def edge_detectionCUDA(image):
    # Convert image to tensor and move to CPU
    image_tensor = torch.tensor(image, dtype=torch.float32, device='cpu')

    # Convert image to grayscale if it has 3 channels
    if image_tensor.ndim == 3 and image_tensor.shape[-1] == 3:
        image_tensor = torch.mean(image_tensor, dim=-1, keepdim=True)

    # Add batch dimension
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # Change shape to [batch_size, channels, height, width]

    # Define horizontal and vertical Sobel kernels
    horizontal_kernel = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device='cpu').unsqueeze(0).unsqueeze(0)
    vertical_kernel = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device='cpu').unsqueeze(0).unsqueeze(0)

    # Apply horizontal and vertical Sobel filters separately
    horizontal_edges = F.conv2d(image_tensor, horizontal_kernel, padding=1)
    vertical_edges = F.conv2d(image_tensor, vertical_kernel, padding=1)

    # Combine horizontal and vertical edges to obtain the magnitude of the gradient
    gradient_magnitude = torch.sqrt(horizontal_edges ** 2 + vertical_edges ** 2)

    # Normalize gradient magnitude
    gradient_magnitude_normalized = gradient_magnitude / gradient_magnitude.max()

    # Convert back to NumPy array and return
    return gradient_magnitude_normalized.squeeze().cpu().numpy()