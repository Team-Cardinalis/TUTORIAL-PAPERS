#!/usr/bin/env python3
"""
Convolution Output Dimensions Calculator
Generates a visualization showing how convolution output dimensions are calculated

This script creates a comprehensive visualization showing the relationship between
input size, kernel size, stride, padding, and dilation on output dimensions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import os

def calculate_output_dimensions(H_in, W_in, k, s, p, d):
    """
    Calculate output dimensions for 2D convolution.
    
    Args:
        H_in, W_in: Input height and width
        k: Kernel size
        s: Stride
        p: Padding
        d: Dilation
    
    Returns:
        H_out, W_out: Output height and width
    """
    H_out = (H_in + 2*p - d*(k-1) - 1) // s + 1
    W_out = (W_in + 2*p - d*(k-1) - 1) // s + 1
    return H_out, W_out

def create_output_dimension_visualization(output_path):
    """
    Create a comprehensive visualization of convolution output dimensions.
    
    Args:
        output_path (str): Path to save the visualization
    """
    # Set up the figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Create a grid layout
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # Example 1: Basic convolution (no padding, stride=1)
    ax1 = fig.add_subplot(gs[0, 0])
    H_in, W_in = 6, 6
    k, s, p, d = 3, 1, 0, 1
    H_out, W_out = calculate_output_dimensions(H_in, W_in, k, s, p, d)
    
    # Draw input
    input_rect = Rectangle((0, 0), W_in, H_in, linewidth=2, edgecolor='blue', 
                          facecolor='lightblue', alpha=0.7, label=f'Input: {H_in}×{W_in}')
    ax1.add_patch(input_rect)
    
    # Draw kernel
    kernel_rect = Rectangle((1, 1), k, k, linewidth=2, edgecolor='red', 
                           facecolor='red', alpha=0.5, label=f'Kernel: {k}×{k}')
    ax1.add_patch(kernel_rect)
    
    # Draw output
    output_rect = Rectangle((0, 0), W_out, H_out, linewidth=2, edgecolor='green', 
                           facecolor='lightgreen', alpha=0.7, label=f'Output: {H_out}×{W_out}')
    ax1.add_patch(output_rect)
    
    ax1.set_xlim(-0.5, W_in + 0.5)
    ax1.set_ylim(-0.5, H_in + 0.5)
    ax1.set_aspect('equal')
    ax1.set_title('No Padding, Stride=1\nOutput = (H+2p-k)/s + 1')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Example 2: With padding
    ax2 = fig.add_subplot(gs[0, 1])
    H_in, W_in = 6, 6
    k, s, p, d = 3, 1, 1, 1
    H_out, W_out = calculate_output_dimensions(H_in, W_in, k, s, p, d)
    
    # Draw padded input
    padded_rect = Rectangle((-p, -p), W_in + 2*p, H_in + 2*p, linewidth=2, 
                           edgecolor='purple', facecolor='lightpurple', alpha=0.5)
    ax2.add_patch(padded_rect)
    
    # Draw original input
    input_rect = Rectangle((0, 0), W_in, H_in, linewidth=2, edgecolor='blue', 
                          facecolor='lightblue', alpha=0.7, label=f'Input: {H_in}×{W_in}')
    ax2.add_patch(input_rect)
    
    # Draw kernel
    kernel_rect = Rectangle((0, 0), k, k, linewidth=2, edgecolor='red', 
                           facecolor='red', alpha=0.5, label=f'Kernel: {k}×{k}')
    ax2.add_patch(kernel_rect)
    
    # Draw output
    output_rect = Rectangle((0, 0), W_out, H_out, linewidth=2, edgecolor='green', 
                           facecolor='lightgreen', alpha=0.7, label=f'Output: {H_out}×{W_out}')
    ax2.add_patch(output_rect)
    
    ax2.set_xlim(-p-0.5, W_in + p + 0.5)
    ax2.set_ylim(-p-0.5, H_in + p + 0.5)
    ax2.set_aspect('equal')
    ax2.set_title('With Padding (p=1)\nSame output size as input')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Example 3: With stride
    ax3 = fig.add_subplot(gs[0, 2])
    H_in, W_in = 8, 8
    k, s, p, d = 3, 2, 0, 1
    H_out, W_out = calculate_output_dimensions(H_in, W_in, k, s, p, d)
    
    # Draw input
    input_rect = Rectangle((0, 0), W_in, H_in, linewidth=2, edgecolor='blue', 
                          facecolor='lightblue', alpha=0.7, label=f'Input: {H_in}×{W_in}')
    ax3.add_patch(input_rect)
    
    # Draw kernel positions
    for i in range(0, H_in - k + 1, s):
        for j in range(0, W_in - k + 1, s):
            kernel_rect = Rectangle((j, i), k, k, linewidth=1, edgecolor='red', 
                                   facecolor='red', alpha=0.3)
            ax3.add_patch(kernel_rect)
    
    # Draw output
    output_rect = Rectangle((0, 0), W_out, H_out, linewidth=2, edgecolor='green', 
                           facecolor='lightgreen', alpha=0.7, label=f'Output: {H_out}×{W_out}')
    ax3.add_patch(output_rect)
    
    ax3.set_xlim(-0.5, W_in + 0.5)
    ax3.set_ylim(-0.5, H_in + 0.5)
    ax3.set_aspect('equal')
    ax3.set_title('Stride=2\nReduces output size')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Formula visualization
    ax4 = fig.add_subplot(gs[1, :])
    formula_text = r"""
    Convolution Output Dimensions Formula:
    
    $H_{out} = \frac{H_{in} + 2p - d \cdot (k-1) - 1}{s} + 1$
    $W_{out} = \frac{W_{in} + 2p - d \cdot (k-1) - 1}{s} + 1$
    
    Where:
    • $H_{in}, W_{in}$ = Input height and width
    • $H_{out}, W_{out}$ = Output height and width  
    • $k$ = Kernel size
    • $s$ = Stride
    • $p$ = Padding
    • $d$ = Dilation
    """
    ax4.text(0.5, 0.5, formula_text, transform=ax4.transAxes, 
             fontsize=14, ha='center', va='center', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    # Parameter effects visualization
    ax5 = fig.add_subplot(gs[2, :])
    
    # Create parameter comparison table
    params = [
        ('Input Size', '6×6', '6×6', '8×8', '10×10'),
        ('Kernel Size', '3×3', '5×5', '3×3', '3×3'),
        ('Stride', '1', '1', '2', '1'),
        ('Padding', '0', '0', '0', '1'),
        ('Dilation', '1', '1', '1', '1'),
        ('Output Size', '4×4', '2×2', '3×3', '10×10')
    ]
    
    table = ax5.table(cellText=params, 
                      colLabels=['Example 1', 'Example 2', 'Example 3', 'Example 4'],
                      cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    ax5.set_title('Parameter Effects on Output Dimensions', fontsize=16, pad=20)
    ax5.axis('off')
    
    # Interactive dimension calculator
    ax6 = fig.add_subplot(gs[3, :])
    
    # Create a simple calculator interface
    calculator_text = """
    Quick Dimension Calculator:
    
    For input size 224×224 with 3×3 kernel:
    • No padding, stride=1: 222×222
    • Padding=1, stride=1: 224×224 (same)
    • No padding, stride=2: 111×111
    • Padding=1, stride=2: 112×112
    
    For input size 32×32 with 5×5 kernel:
    • No padding, stride=1: 28×28
    • Padding=2, stride=1: 32×32 (same)
    • No padding, stride=2: 14×14
    """
    
    ax6.text(0.5, 0.5, calculator_text, transform=ax6.transAxes, 
             fontsize=12, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    
    plt.suptitle('Convolution Output Dimensions Calculator', fontsize=20, y=0.95)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Output dimension visualization completed. Result saved to: {output_path}")

if __name__ == "__main__":
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "output_dimensions.png")
    
    # Create visualization
    create_output_dimension_visualization(output_path) 