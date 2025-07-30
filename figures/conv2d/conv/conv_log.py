#!/usr/bin/env python3
"""
Laplacian of Gaussian (LoG) Convolution Example
Generates edge detection using LoG kernel on sample.png

This script demonstrates edge detection using the Laplacian of Gaussian kernel:
[[ 0,  0, -1,  0,  0],
 [ 0, -1, -2, -1,  0],
 [-1, -2, 16, -2, -1],
 [ 0, -1, -2, -1,  0],
 [ 0,  0, -1,  0,  0]]

The kernel combines Gaussian smoothing with Laplacian edge detection.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d
import os

def apply_log_convolution(image_path, output_path):
    """
    Apply Laplacian of Gaussian convolution to detect edges.
    
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save output image
    """
    # Load image and convert to grayscale
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    
    # Define Laplacian of Gaussian kernel for edge detection
    log_kernel = np.array([
        [ 0,  0, -1,  0,  0],
        [ 0, -1, -2, -1,  0],
        [-1, -2, 16, -2, -1],
        [ 0, -1, -2, -1,  0],
        [ 0,  0, -1,  0,  0]
    ])
    
    # Apply convolution
    # Note: scipy.signal.convolve2d flips the kernel, so we need to flip it back
    # to get the correct LoG effect
    convolved = convolve2d(img_array, log_kernel[::-1, ::-1], mode='same')
    
    # Normalize the result to 0-255 range
    convolved = np.clip(convolved, 0, 255).astype(np.uint8)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(img_array, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Kernel visualization
    im = axes[1].imshow(log_kernel, cmap='RdBu_r', vmin=-2, vmax=16)
    axes[1].set_title('LoG Kernel\n(Laplacian of Gaussian)')
    axes[1].axis('off')
    
    # Add kernel values as text
    for i in range(5):
        for j in range(5):
            axes[1].text(j, i, str(log_kernel[i, j]), 
                        ha='center', va='center', 
                        color='black', fontweight='bold', fontsize=8)
    
    # Convolved result
    axes[2].imshow(convolved, cmap='gray')
    axes[2].set_title('LoG Result\n(Combined Smoothing and Edge Detection)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Laplacian of Gaussian convolution completed. Result saved to: {output_path}")

if __name__ == "__main__":
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_path = os.path.join(script_dir, "..", "..", "sample.png")
    output_path = os.path.join(script_dir, "conv_log.png")
    
    # Check if sample.png exists
    if not os.path.exists(sample_path):
        print(f"Error: sample.png not found at {sample_path}")
        exit(1)
    
    # Apply convolution and save result
    apply_log_convolution(sample_path, output_path) 