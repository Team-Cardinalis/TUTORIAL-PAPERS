<a href="https://www.teamcardinalis.com/">
   <img src="Logo/teamcardinalis.png" alt="Team Cardinalis" width="100">
</a>

# Understanding 2D Convolution: A Comprehensive Guide

## Abstract

This tutorial provides a thorough introduction to 2D convolution, the fundamental operation that powers convolutional neural networks in computer vision. Designed for beginners in deep learning, we start with intuitive visual concepts and progressively build to mathematical formulations and practical PyTorch implementations. Through concrete examples, visual demonstrations, and real-world applications, readers will understand how convolution extracts features from images and how to configure Conv2D layers for various computer vision tasks.

## Table of Contents

1. [What is Convolution?](#what-is-convolution)
2. [The Intuition Behind Convolution](#the-intuition-behind-convolution)
3. [Basic Convolution Operation](#basic-convolution-operation)
4. [Convolution Parameters](#convolution-parameters)
5. [PyTorch Implementation](#pytorch-implementation)
6. [Real-World Applications](#real-world-applications)
7. [Advanced Concepts](#advanced-concepts)
8. [Mathematical Foundation](#mathematical-foundation)

## What is Convolution?

Convolution is a mathematical operation that combines two functions to produce a third function. In computer vision, we use convolution to extract features from images by sliding a small filter (called a kernel) over the image and computing a weighted sum at each position.

Think of convolution as a way to "look for patterns" in an image. Just like how your brain recognizes edges, textures, and shapes in what you see, convolution helps computers identify these same features.

## The Intuition Behind Convolution

### Why Do We Need Convolution?

Imagine you're trying to identify if there's a cat in a photo. You don't need to look at the entire image at once - you can recognize a cat by looking for specific features:
- Pointed ears
- Whiskers
- A tail
- Fur texture

Convolution works the same way. Instead of processing the entire image at once, it looks for specific patterns in small regions, then combines these local observations to understand the whole image.

### The Sliding Window Concept

Convolution uses a "sliding window" approach:
1. Place a small filter (kernel) over a region of the image
2. Multiply corresponding pixels and sum the results
3. Move the filter to the next position
4. Repeat until the entire image is processed

This creates a new image (called a feature map) that highlights where the pattern was found.

## Basic Convolution Operation

### Simple Example: Edge Detection

Let's start with a concrete example. We want to detect vertical edges in an image.

**Step 1: Define the Kernel**
```python
import torch
import torch.nn as nn

# Vertical edge detection kernel
vertical_edge_kernel = torch.tensor([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
]).float()
```

**Step 2: Apply Convolution**
```python
# Create a simple test image (6x6)
test_image = torch.tensor([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
]).float()

# Add batch and channel dimensions
test_image = test_image.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 6, 6)
kernel = vertical_edge_kernel.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 3, 3)

# Apply convolution
conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, 
                      stride=1, padding=0, bias=False)
conv_layer.weight.data = kernel

with torch.no_grad():
    output = conv_layer(test_image)
    print("Input shape:", test_image.shape)
    print("Output shape:", output.shape)
    print("Output feature map:")
    print(output.squeeze())
```

**What happens:**
- The kernel slides over the image
- At each position, it multiplies the kernel values with the image pixels
- The sum becomes the output pixel value
- Vertical edges (where pixel values change from left to right) produce strong responses

## Convolution Parameters

### 1. Kernel Size

The kernel size determines the receptive field - how much of the input each output pixel can "see".

```python
# Different kernel sizes
small_kernel = nn.Conv2d(1, 1, kernel_size=3)  # 3x3 kernel
medium_kernel = nn.Conv2d(1, 1, kernel_size=5)  # 5x5 kernel
large_kernel = nn.Conv2d(1, 1, kernel_size=7)  # 7x7 kernel

print("Small kernel parameters:", small_kernel.weight.shape)
print("Medium kernel parameters:", medium_kernel.weight.shape)
print("Large kernel parameters:", large_kernel.weight.shape)
```

**When to use each:**
- **3x3**: Most common, good balance of efficiency and effectiveness
- **5x5**: Captures larger patterns, but more parameters
- **7x7**: Very large receptive field, used in early layers

### 2. Stride

Stride controls how much the kernel moves between positions.

```python
# Same kernel, different strides
stride_1 = nn.Conv2d(1, 1, kernel_size=3, stride=1)
stride_2 = nn.Conv2d(1, 1, kernel_size=3, stride=2)

test_input = torch.randn(1, 1, 8, 8)
output_1 = stride_1(test_input)
output_2 = stride_2(test_input)

print("Input shape:", test_input.shape)
print("Output with stride=1:", output_1.shape)
print("Output with stride=2:", output_2.shape)
```

**Effects of stride:**
- **Stride=1**: Kernel moves one pixel at a time (dense output)
- **Stride=2**: Kernel moves two pixels at a time (reduces output size by half)
- **Higher stride**: Faster computation, smaller output, may lose fine details

### 3. Padding

Padding adds zeros around the input to control output size.

```python
# Same kernel, different padding
no_padding = nn.Conv2d(1, 1, kernel_size=3, padding=0)
with_padding = nn.Conv2d(1, 1, kernel_size=3, padding=1)

test_input = torch.randn(1, 1, 6, 6)
output_no_pad = no_padding(test_input)
output_with_pad = with_padding(test_input)

print("Input shape:", test_input.shape)
print("Output without padding:", output_no_pad.shape)
print("Output with padding:", output_with_pad.shape)
```

**Types of padding:**
- **No padding**: Output is smaller than input
- **Same padding**: Output has same spatial dimensions as input
- **Valid padding**: No padding, output is smaller

### 4. Number of Channels

Channels represent different types of features.

```python
# Single channel (grayscale)
conv_1ch = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)

# Multiple channels (RGB)
conv_3ch = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)

# Multiple output channels
conv_multi = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)

print("1 channel input -> 16 outputs:", conv_1ch.weight.shape)
print("3 channel input -> 16 outputs:", conv_3ch.weight.shape)
print("3 channel input -> 64 outputs:", conv_multi.weight.shape)
```

## PyTorch Implementation

### Basic Conv2D Layer

```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Create a simple Conv2D layer
conv_layer = nn.Conv2d(
    in_channels=1,      # Grayscale image
    out_channels=1,     # One output feature map
    kernel_size=3,      # 3x3 kernel
    stride=1,           # Move one pixel at a time
    padding=1,          # Add padding to maintain size
    bias=False          # No bias term for simplicity
)

# Initialize with a specific kernel (edge detection)
edge_kernel = torch.tensor([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
]).float()

conv_layer.weight.data = edge_kernel.unsqueeze(0).unsqueeze(0)

# Test with a simple image
test_image = torch.randn(1, 1, 8, 8)
output = conv_layer(test_image)

print("Input shape:", test_image.shape)
print("Output shape:", output.shape)
print("Kernel shape:", conv_layer.weight.shape)
```

### Multiple Feature Maps

```python
# Create a layer that outputs multiple feature maps
multi_conv = nn.Conv2d(
    in_channels=1,
    out_channels=6,     # 6 different feature maps
    kernel_size=3,
    padding=1
)

# Test with input
test_input = torch.randn(1, 1, 10, 10)
output = multi_conv(test_input)

print("Input shape:", test_input.shape)
print("Output shape:", output.shape)  # (1, 6, 10, 10)
print("Number of parameters:", sum(p.numel() for p in multi_conv.parameters()))
```

### Convolutional Neural Network Example

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # First conv block
        x = self.pool1(self.relu1(self.conv1(x)))
        
        # Second conv block
        x = self.pool2(self.relu2(self.conv2(x)))
        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        
        return x

# Create and test the network
model = SimpleCNN()
test_input = torch.randn(1, 1, 28, 28)  # MNIST-like input
output = model(test_input)

print("Input shape:", test_input.shape)
print("Output shape:", output.shape)
print("Total parameters:", sum(p.numel() for p in model.parameters()))
```

## Real-World Applications

### 1. Image Classification

```python
# Example: Classifying handwritten digits
import torchvision
import torchvision.transforms as transforms

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                     download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                        shuffle=True)

# Training loop
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.3f}')
```

### 2. Edge Detection

```python
# Apply different edge detection kernels
def apply_edge_detection(image_tensor):
    # Sobel X kernel (vertical edges)
    sobel_x = torch.tensor([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]).float()
    
    # Sobel Y kernel (horizontal edges)
    sobel_y = torch.tensor([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ]).float()
    
    # Laplacian kernel (all edges)
    laplacian = torch.tensor([
        [ 0, -1,  0],
        [-1,  4, -1],
        [ 0, -1,  0]
    ]).float()
    
    # Create convolution layers
    conv_x = nn.Conv2d(1, 1, 3, padding=1, bias=False)
    conv_y = nn.Conv2d(1, 1, 3, padding=1, bias=False)
    conv_lap = nn.Conv2d(1, 1, 3, padding=1, bias=False)
    
    # Set kernels
    conv_x.weight.data = sobel_x.unsqueeze(0).unsqueeze(0)
    conv_y.weight.data = sobel_y.unsqueeze(0).unsqueeze(0)
    conv_lap.weight.data = laplacian.unsqueeze(0).unsqueeze(0)
    
    # Apply convolutions
    edges_x = conv_x(image_tensor)
    edges_y = conv_y(image_tensor)
    edges_lap = conv_lap(image_tensor)
    
    return edges_x, edges_y, edges_lap

# Test with a simple image
test_img = torch.randn(1, 1, 10, 10)
edges_x, edges_y, edges_lap = apply_edge_detection(test_img)

print("Original image shape:", test_img.shape)
print("Sobel X output shape:", edges_x.shape)
print("Sobel Y output shape:", edges_y.shape)
print("Laplacian output shape:", edges_lap.shape)
```

### 3. Feature Extraction

```python
# Extract features from a pre-trained model
import torchvision.models as models

# Load pre-trained ResNet
resnet = models.resnet18(pretrained=True)

# Remove the final classification layer
feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

# Test with a sample image
sample_image = torch.randn(1, 3, 224, 224)  # RGB image
features = feature_extractor(sample_image)

print("Input image shape:", sample_image.shape)
print("Extracted features shape:", features.shape)
print("Number of features:", features.numel())
```

## Advanced Concepts

### 1. Dilation

Dilation increases the receptive field without increasing parameters.

```python
# Regular convolution
regular_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

# Dilated convolution
dilated_conv = nn.Conv2d(1, 1, kernel_size=3, padding=2, dilation=2)

test_input = torch.randn(1, 1, 10, 10)
output_regular = regular_conv(test_input)
output_dilated = dilated_conv(test_input)

print("Regular conv output shape:", output_regular.shape)
print("Dilated conv output shape:", output_dilated.shape)
```

### 2. Grouped Convolution

Grouped convolution processes channels in groups, reducing computation.

```python
# Regular convolution
regular_conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)

# Grouped convolution (groups=2)
grouped_conv = nn.Conv2d(64, 128, kernel_size=3, padding=1, groups=2)

test_input = torch.randn(1, 64, 32, 32)
output_regular = regular_conv(test_input)
output_grouped = grouped_conv(test_input)

print("Regular conv parameters:", sum(p.numel() for p in regular_conv.parameters()))
print("Grouped conv parameters:", sum(p.numel() for p in grouped_conv.parameters()))
```

### 3. Depthwise Separable Convolution

Depthwise separable convolution reduces parameters significantly.

```python
# Regular convolution
regular_conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)

# Depthwise separable convolution
depthwise_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64)
pointwise_conv = nn.Conv2d(64, 128, kernel_size=1)

test_input = torch.randn(1, 64, 32, 32)
output_regular = regular_conv(test_input)
output_separable = pointwise_conv(depthwise_conv(test_input))

print("Regular conv parameters:", sum(p.numel() for p in regular_conv.parameters()))
print("Separable conv parameters:", 
      sum(p.numel() for p in depthwise_conv.parameters()) +
      sum(p.numel() for p in pointwise_conv.parameters()))
```

## Mathematical Foundation

### The Convolution Formula

The 2D convolution operation is mathematically defined as:

$$O(i,j) = \sum_{m=0}^{K_h-1}\sum_{n=0}^{K_w-1} I(i \cdot s + m - p, j \cdot s + n - p) \cdot K(m,n)$$

Where:
- $I$ is the input feature map
- $K$ is the convolution kernel of size $K_h \times K_w$
- $s$ is the stride
- $p$ is the padding
- $O(i,j)$ is the output at position $(i,j)$

### Multi-Channel Convolution

For multi-channel inputs, the formula becomes:

$$O(i,j) = \sum_{c=1}^{C_{in}} \sum_{m=0}^{K_h-1}\sum_{n=0}^{K_w-1} I_c(i \cdot s + m - p, j \cdot s + n - p) \cdot K_c(m,n)$$

Where $C_{in}$ is the number of input channels.

### Output Dimension Calculation

The output dimensions are calculated as:

$$H_{out} = \frac{H_{in} + 2p - d \cdot (K_h - 1) - 1}{s} + 1$$

$$W_{out} = \frac{W_{in} + 2p - d \cdot (K_w - 1) - 1}{s} + 1$$

Where:
- $H_{in}, W_{in}$ are input height and width
- $H_{out}, W_{out}$ are output height and width
- $d$ is the dilation factor

### Backpropagation

The gradient with respect to the weights is:

$$\frac{\partial L}{\partial K_c(m,n)} = \sum_{i,j} \frac{\partial L}{\partial O(i,j)} \cdot I_c(i \cdot s + m - p, j \cdot s + n - p)$$

And the gradient with respect to the bias is:

$$\frac{\partial L}{\partial b} = \sum_{i,j} \frac{\partial L}{\partial O(i,j)}$$

## Visual Examples

### Edge Detection Kernels

The following examples demonstrate how different kernels extract different features from images:

**Sobel X Kernel (Vertical Edges):**
```
-1  0  1
-2  0  2
-1  0  1
```

**Sobel Y Kernel (Horizontal Edges):**
```
-1 -2 -1
 0  0  0
 1  2  1
```

**Laplacian Kernel (All Edges):**
```
 0 -1  0
-1  4 -1
 0 -1  0
```

**Prewitt X Kernel (Vertical Edges):**
```
-1  0  1
-1  0  1
-1  0  1
```

**Prewitt Y Kernel (Horizontal Edges):**
```
-1 -1 -1
 0  0  0
 1  1  1
```

**Roberts X Kernel (Diagonal Edges):**
```
1  0
0 -1
```

**Roberts Y Kernel (Diagonal Edges):**
```
0  1
-1 0
```

**Emboss Kernel (Relief Effect):**
```
-2 -1  0
-1  1  1
 0  1  2
```

**Laplacian of Gaussian Kernel:**
```
 0  0 -1  0  0
 0 -1 -2 -1  0
-1 -2 16 -2 -1
 0 -1 -2 -1  0
 0  0 -1  0  0
```

![Sobel X output](Figures/conv2d/conv/conv_sobel_x.png)
*Figure 1: Sobel X - vertical edges*

![Sobel Y output](Figures/conv2d/conv/conv_sobel_y.png)
*Figure 2: Sobel Y - horizontal edges*

![Laplacian output](Figures/conv2d/conv/conv_laplacian.png)
*Figure 3: Laplacian - all edges*

![Prewitt X output](Figures/conv2d/conv/conv_prewitt_x.png)
*Figure 4: Prewitt X - vertical edges*

![Prewitt Y output](Figures/conv2d/conv/conv_prewitt_y.png)
*Figure 5: Prewitt Y - horizontal edges*

![Roberts X output](Figures/conv2d/conv/conv_roberts_x.png)
*Figure 6: Roberts X - diagonal edges (↗ direction)*

![Roberts Y output](Figures/conv2d/conv/conv_roberts_y.png)
*Figure 7: Roberts Y - diagonal edges (↘ direction)*

![Emboss output](Figures/conv2d/conv/conv_emboss.png)
*Figure 8: Emboss - relief effect in a given direction*

![Laplacian of Gaussian output](Figures/conv2d/conv/conv_log.png)
*Figure 9: LoG (Laplacian of Gaussian) - combined smoothing and edge enhancement*

### Output Dimension Visualization

<img src="Figures/conv2d/out_dim_calc/output_dimensions.png" alt="Output dimensions" width="400">

*Figure 10: Convolution Output Dimensions - Output size of a convolution operation as a function of kernel size (k), stride (s), padding (p), and dilation (d)*

## Sources

PyTorch documentation
> https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

TensorFlow documentation
> https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D

CS231n Deep Learning for Computer Vision
> http://cs231n.github.io/convolutional-networks/

Dive into Deep Learning
> https://d2l.ai/chapter_convolutional-neural-networks/conv.html

Deep Learning an MIT Press book
> https://www.deeplearningbook.org/

## Contributors

Killian OTT
> <a href="https://www.linkedin.com/in/killian-ott/">
>  <img src="Logo/linkedin.png" alt="LinkedIn" width="20">
> </a>

Sasha MARMAIN
> <a href="https://www.linkedin.com/in/sasha-marmain-7a9645294/">
>  <img src="Logo/linkedin.png" alt="LinkedIn" width="20">
> </a>
