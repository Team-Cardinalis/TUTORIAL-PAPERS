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

def run_script(script_path):
    """
    Run a Python script and handle any errors.
    
    Args:
        script_path (str): Path to the script to run
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"Running: {script_path}")
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, check=True)
        print(f"✓ Success: {script_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running {script_path}:")
        print(f"  Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"✗ Script not found: {script_path}")
        return False

def main():
    """
    Run all convolution example scripts.
    """
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define all the scripts to run
    scripts = [
        # Convolution examples
        os.path.join(script_dir, "conv", "conv_sobel_x.py"),
        os.path.join(script_dir, "conv", "conv_sobel_y.py"),
        os.path.join(script_dir, "conv", "conv_laplacian.py"),
        os.path.join(script_dir, "conv", "conv_prewitt_x.py"),
        os.path.join(script_dir, "conv", "conv_prewitt_y.py"),
        os.path.join(script_dir, "conv", "conv_roberts_x.py"),
        os.path.join(script_dir, "conv", "conv_roberts_y.py"),
        os.path.join(script_dir, "conv", "conv_emboss.py"),
        os.path.join(script_dir, "conv", "conv_log.py"),
        
        # Output dimension calculator
        os.path.join(script_dir, "out_dim_calc", "output_dimensions.py"),
    ]
    
    print("Starting convolution example generation...")
    print("=" * 50)
    
    # Check if sample.png exists
    sample_path = os.path.join(script_dir, "..", "sample.png")
    if not os.path.exists(sample_path):
        print(f"Error: sample.png not found at {sample_path}")
        print("Please ensure sample.png is in the figures directory.")
        return False
    
    # Run all scripts
    successful = 0
    total = len(scripts)
    
    for script in scripts:
        if run_script(script):
            successful += 1
        print()
    
    # Summary
    print("=" * 50)
    print(f"Generation complete: {successful}/{total} scripts successful")
    
    if successful == total:
        print("✓ All convolution examples generated successfully!")
        print("\nGenerated files:")
        
        # List generated files
        conv_dir = os.path.join(script_dir, "conv")
        out_dim_dir = os.path.join(script_dir, "out_dim_calc")
        
        if os.path.exists(conv_dir):
            for file in os.listdir(conv_dir):
                if file.endswith('.png'):
                    print(f"  - figures/conv2d/conv/{file}")
        
        if os.path.exists(out_dim_dir):
            for file in os.listdir(out_dim_dir):
                if file.endswith('.png'):
                    print(f"  - figures/conv2d/out_dim_calc/{file}")
        
        return True
    else:
        print(f"✗ {total - successful} scripts failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 