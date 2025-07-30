#!/usr/bin/env python3
"""
Emboss Convolution Example
Generates relief effect using Emboss kernel on sample.png

This script demonstrates relief effect using the Emboss kernel:
[[-2, -1, 0],
 [-1,  1, 1],
 [ 0,  1, 2]]

The kernel creates a 3D embossed effect by emphasizing directional gradients.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d
import os

def apply_emboss_convolution(image_path, output_path):
    """
    Apply Emboss convolution to create relief effect.
    
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save output image
    """
    # Load image and convert to grayscale
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    
    # Define Emboss kernel for relief effect
    emboss = np.array([
        [-2, -1, 0],
        [-1,  1, 1],
        [ 0,  1, 2]
    ])
    
    # Apply convolution
    # Note: scipy.signal.convolve2d flips the kernel, so we need to flip it back
    # to get the correct Emboss effect
    convolved = convolve2d(img_array, emboss[::-1, ::-1], mode='same')
    
    # Normalize the result to 0-255 range
    convolved = np.clip(convolved, 0, 255).astype(np.uint8)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img_array, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Kernel visualization
    im = axes[1].imshow(emboss, cmap='RdBu_r', vmin=-2, vmax=2)
    axes[1].set_title('Emboss Kernel\n(Relief Effect)')
    axes[1].axis('off')
    
    # Add kernel values as text
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, str(emboss[i, j]), 
                        ha='center', va='center', 
                        color='black', fontweight='bold')
    
    # Convolved result
    axes[2].imshow(convolved, cmap='gray')
    axes[2].set_title('Emboss Result\n(Relief Effect)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Emboss convolution completed. Result saved to: {output_path}")

if __name__ == "__main__":
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_path = os.path.join(script_dir, "..", "..", "sample.png")
    output_path = os.path.join(script_dir, "conv_emboss.png")
    
    # Check if sample.png exists
    if not os.path.exists(sample_path):
        print(f"Error: sample.png not found at {sample_path}")
        exit(1)
    
    # Apply convolution and save result
    apply_emboss_convolution(sample_path, output_path) 