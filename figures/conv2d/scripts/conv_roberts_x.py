#!/usr/bin/env python3
"""
Roberts X Convolution Example
Generates diagonal edge detection using Roberts X kernel on sample.png

This script demonstrates diagonal edge detection using the Roberts X kernel:
[[ 1,  0],
 [ 0, -1]]

The kernel detects diagonal edges by computing differences along the diagonal.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d
import os
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import print as rprint

# Initialize Rich console
console = Console()

def apply_roberts_x_convolution(image_path, output_path):
    """
    Apply Roberts X convolution to detect diagonal edges.
    
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save output image
    """
    with console.status("[bold green]Loading image...", spinner="dots"):
        # Load image and convert to grayscale
        img = Image.open(image_path).convert('L')
        img_array = np.array(img)
        console.log(f"[green]✓[/green] Image loaded: {img_array.shape[0]}x{img_array.shape[1]} pixels")
    
    # Define Roberts X kernel for diagonal edge detection
    roberts_x = np.array([
        [ 1,  0],
        [ 0, -1]
    ])
    
    # Display kernel information
    kernel_table = Table(title="Roberts X Kernel Configuration")
    kernel_table.add_column("Parameter", style="cyan")
    kernel_table.add_column("Value", style="yellow")
    kernel_table.add_row("Kernel Size", "2x2")
    kernel_table.add_row("Purpose", "Diagonal Edge Detection")
    kernel_table.add_row("Method", "Diagonal Difference")
    console.print(kernel_table)
    
    with console.status("[bold blue]Applying convolution...", spinner="dots"):
        # Apply convolution
        # Note: scipy.signal.convolve2d flips the kernel, so we need to flip it back
        # to get the correct Roberts X effect
        convolved = convolve2d(img_array, roberts_x[::-1, ::-1], mode='same')
        
        # Normalize the result to 0-255 range
        convolved = np.clip(convolved, 0, 255).astype(np.uint8)
        console.log(f"[blue]✓[/blue] Convolution completed")
    
    with console.status("[bold magenta]Creating visualization...", spinner="dots"):
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img_array, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Kernel visualization
        im = axes[1].imshow(roberts_x, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1].set_title('Roberts X Kernel\n(Diagonal Edge Detection)')
        axes[1].axis('off')
        
        # Add kernel values as text
        for i in range(2):
            for j in range(2):
                axes[1].text(j, i, str(roberts_x[i, j]), 
                            ha='center', va='center', 
                            color='black', fontweight='bold')
        
        # Convolved result
        axes[2].imshow(convolved, cmap='gray')
        axes[2].set_title('Roberts X Result\n(Diagonal Edges)')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        console.log(f"[magenta]✓[/magenta] Visualization created")
    
    # Display results summary
    results_table = Table(title="Processing Results")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="yellow")
    results_table.add_row("Input Image Size", f"{img_array.shape[0]}x{img_array.shape[1]}")
    results_table.add_row("Output Image Size", f"{convolved.shape[0]}x{convolved.shape[1]}")
    results_table.add_row("Output File", os.path.basename(output_path))
    results_table.add_row("Output Path", output_path)
    console.print(results_table)
    
    console.print(Panel(f"[bold green]Roberts X convolution completed successfully![/bold green]\n"
                       f"Result saved to: [yellow]{output_path}[/yellow]", 
                       title="Processing Complete"))

def main():
    """
    Main function to run the Roberts X convolution script.
    """
    parser = argparse.ArgumentParser(description='Roberts X Convolution Example')
    parser.add_argument('--output-dir', '-o', type=str, default='.',
                       help='Output directory for generated images (default: current directory)')
    parser.add_argument('--input-image', '-i', type=str, default=None,
                       help='Input image path (default: sample.png in figures directory)')
    
    args = parser.parse_args()
    
    console.print(Panel("[bold blue]Roberts X Convolution Script[/bold blue]\n"
                       "Diagonal edge detection using Roberts X kernel", 
                       title="Starting Processing"))
    
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set input image path
    if args.input_image:
        sample_path = args.input_image
    else:
        sample_path = os.path.join(script_dir, "..", "..", "sample.png")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set output path
    output_path = os.path.join(args.output_dir, "conv_roberts_x.png")
    
    # Check if sample.png exists
    if not os.path.exists(sample_path):
        console.print(f"[bold red]Error:[/bold red] Input image not found at [yellow]{sample_path}[/yellow]")
        exit(1)
    
    console.log(f"[green]✓[/green] Input file found: [yellow]{sample_path}[/yellow]")
    console.log(f"[green]✓[/green] Output will be saved to: [yellow]{output_path}[/yellow]")
    
    # Apply convolution and save result
    apply_roberts_x_convolution(sample_path, output_path)

if __name__ == "__main__":
    main() 