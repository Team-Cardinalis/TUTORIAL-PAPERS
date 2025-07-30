#!/usr/bin/env python3
"""
Laplacian Convolution Example
Generates edge detection using Laplacian kernel on sample.png

This script demonstrates edge detection using the Laplacian kernel:
[[ 0, -1,  0],
 [-1,  4, -1],
 [ 0, -1,  0]]

The kernel detects edges in all directions by computing the second derivative.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d
import os

def apply_laplacian_convolution(image_path, output_path):
    """
    Apply Laplacian convolution to detect edges in all directions.
    
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save output image
    """
    # Load image and convert to grayscale
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    
    # Define Laplacian kernel for edge detection
    laplacian = np.array([
        [ 0, -1,  0],
        [-1,  4, -1],
        [ 0, -1,  0]
    ])
    
    # Apply convolution
    # Note: scipy.signal.convolve2d flips the kernel, so we need to flip it back
    # to get the correct Laplacian effect
    convolved = convolve2d(img_array, laplacian[::-1, ::-1], mode='same')
    
    # Normalize the result to 0-255 range
    convolved = np.clip(convolved, 0, 255).astype(np.uint8)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img_array, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Kernel visualization
    im = axes[1].imshow(laplacian, cmap='RdBu_r', vmin=-1, vmax=4)
    axes[1].set_title('Laplacian Kernel\n(All Direction Edge Detection)')
    axes[1].axis('off')
    
    # Add kernel values as text
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, str(laplacian[i, j]), 
                        ha='center', va='center', 
                        color='black', fontweight='bold')
    
    # Convolved result
    axes[2].imshow(convolved, cmap='gray')
    axes[2].set_title('Laplacian Result\n(All Edges)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Laplacian convolution completed. Result saved to: {output_path}")

if __name__ == "__main__":
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_path = os.path.join(script_dir, "..", "..", "sample.png")
    output_path = os.path.join(script_dir, "conv_laplacian.png")
    
    # Check if sample.png exists
    if not os.path.exists(sample_path):
        print(f"Error: sample.png not found at {sample_path}")
        exit(1)
    
    # Apply convolution and save result
    apply_laplacian_convolution(sample_path, output_path) 