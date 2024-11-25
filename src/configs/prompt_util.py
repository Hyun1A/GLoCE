import torch
import torch.nn.functional as F

def gaussian_kernel(kernel_size=5, sigma=1):
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Compute the 2-dimensional gaussian kernel
    gaussian_kernel = (1./(2.*3.14159*variance)) * \
                      torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) / (2*variance))

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).repeat(4,1,1,1)

    return gaussian_kernel

def smooth_tensor(input_tensor, kernel_size=5, sigma=1):
    # Generate the Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma)
        
    # Apply convolution using the Gaussian kernel
    smoothed_tensor = F.conv2d(input_tensor, kernel, padding=kernel_size//2, groups=4)
    
    return smoothed_tensor.squeeze(0)
