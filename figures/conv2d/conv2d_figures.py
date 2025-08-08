#!/usr/bin/env python3
"""
Master Script for Convolution Examples
Runs all convolution examples and generates all visualizations

This script executes all the individual convolution scripts to generate
complete visualizations for the CONV2D.md tutorial.
"""

import os
import subprocess
import sys
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

# Initialize Rich console
console = Console()

def run_script(script_path, output_dir, input_image=None):
    """
    Run a Python script and handle any errors.
    
    Args:
        script_path (str): Path to the script to run
        output_dir (str): Output directory for generated images
        input_image (str): Input image path (optional)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        console.print(f"[blue]Running:[/blue] {os.path.basename(script_path)}")
        
        # Build command with arguments
        cmd = [sys.executable, script_path, "--output-dir", output_dir]
        if input_image:
            cmd.extend(["--input-image", input_image])
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        console.print(f"[green]✓[/green] Success: {os.path.basename(script_path)}")
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[red]✗[/red] Error running {os.path.basename(script_path)}:")
        console.print(f"  [red]Error:[/red] {e.stderr}")
        return False
    except FileNotFoundError:
        console.print(f"[red]✗[/red] Script not found: {script_path}")
        return False

def main():
    """
    Run all convolution example scripts.
    """
    parser = argparse.ArgumentParser(description='Master Script for Convolution Examples')
    parser.add_argument('--output-dir', '-o', type=str, default='./output',
                       help='Output directory for all generated images (default: ./output)')
    parser.add_argument('--input-image', '-i', type=str, default=None,
                       help='Input image path (default: sample.png in figures directory)')
    
    args = parser.parse_args()
    
    console.print(Panel("[bold blue]Master Script for Convolution Examples[/bold blue]\n"
                       "Generating all convolution visualizations", 
                       title="Starting Processing"))
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define all the scripts to run
    scripts = [
        # Convolution examples
        os.path.join(script_dir, "scripts", "conv_sobel_x.py"),
        os.path.join(script_dir, "scripts", "conv_sobel_y.py"),
        os.path.join(script_dir, "scripts", "conv_laplacian.py"),
        os.path.join(script_dir, "scripts", "conv_prewitt_x.py"),
        os.path.join(script_dir, "scripts", "conv_prewitt_y.py"),
        os.path.join(script_dir, "scripts", "conv_roberts_x.py"),
        os.path.join(script_dir, "scripts", "conv_roberts_y.py"),
        os.path.join(script_dir, "scripts", "conv_emboss.py"),
        os.path.join(script_dir, "scripts", "conv_log.py"),
        
        # Output dimension calculator
        os.path.join(script_dir, "scripts", "output_dimensions.py"),
    ]
    
    # Set input image path if not provided
    if not args.input_image:
        args.input_image = os.path.join(script_dir, "..", "sample.png")
    
    # Check if input image exists
    if not os.path.exists(args.input_image):
        console.print(f"[bold red]Error:[/bold red] Input image not found at [yellow]{args.input_image}[/yellow]")
        console.print("Please ensure the input image exists or specify a different path with --input-image")
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    console.log(f"[green]✓[/green] Input image found: [yellow]{args.input_image}[/yellow]")
    console.log(f"[green]✓[/green] Output directory: [yellow]{args.output_dir}[/yellow]")
    
    console.print("\n[bold]Starting convolution example generation...[/bold]")
    console.print("=" * 60)
    
    # Run all scripts
    successful = 0
    total = len(scripts)
    
    for script in scripts:
        if run_script(script, args.output_dir, args.input_image):
            successful += 1
        console.print()
    
    # Summary
    console.print("=" * 60)
    console.print(f"[bold]Generation complete:[/bold] {successful}/{total} scripts successful")
    
    if successful == total:
        console.print("[bold green]✓[/bold green] All convolution examples generated successfully!")
        
        # List generated files
        console.print("\n[bold]Generated files:[/bold]")
        if os.path.exists(args.output_dir):
            for file in os.listdir(args.output_dir):
                if file.endswith('.png'):
                    console.print(f"  [yellow]•[/yellow] {os.path.join(args.output_dir, file)}")
        
        # Display summary table
        summary_table = Table(title="Generation Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="yellow")
        summary_table.add_row("Total Scripts", str(total))
        summary_table.add_row("Successful", str(successful))
        summary_table.add_row("Failed", str(total - successful))
        summary_table.add_row("Success Rate", f"{(successful/total)*100:.1f}%")
        summary_table.add_row("Output Directory", args.output_dir)
        console.print(summary_table)
        
        return True
    else:
        console.print(f"[bold red]✗[/bold red] {total - successful} scripts failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 