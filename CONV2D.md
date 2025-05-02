**CONV2D**

**ABSTRACT**

2D convolution underlies convolutional neural network architectures in computer vision. This tutorial, aimed at beginners in deep learning and computer vision, provides a comprehensive presentation from the mathematical foundations of the operation through practical implementation and optimization in PyTorch. We systematically examine key parameters such as stride, padding and dilation and elucidate their effects through annotated code examples and visualizations before and after convolution. Upon completing this tutorial, readers will be able to design and configure their own Conv2D layers and justify hyperparameter choices for use cases ranging from edge detection to image classification.

# **INTRODUCTION**

This tutorial paper provides a comprehensive introduction to the Conv2D layer, a core component of Convolutional Neural Networks (CNNs), with a focus on its mathematical foundation, configuration parameters, and role in feature extraction for computer vision tasks.

# **CONFIGURATION**

| Parameter | Description |
| :---- | :---- |
| in\_channels | Number of channels in the input image. |
| out\_channels | Number of output feature maps produced by the convolutional layer. |
| kernel\_size | Dimensions of the convolutional filter applied to the input. |
| stride | Step size that determines how the filter moves across the input. |
| padding | Number of pixels added around the input to control the output size. |
| dilation | Spacing between elements within the kernel to expand the receptive field. |
| bias | Indicates whether a learnable bias is added to the output. |

# **CONVOLUTION OPERATION**

In a Convolutional Neural Network, the convolution operation is the fundamental mechanism by which local features are extracted from input data. In this process, a kernel (or filter) of fixed dimensions is systematically applied to the input feature map by moving it across spatial positions. For each location (i,j) in the output feature map, the convolution operation computes a weighted sum of the values in the receptive field of the input. This is achieved by aligning the kernel with a corresponding patch of the input, performing an element-wise multiplication between the kernel weights and the input values, and summing the results.

The mathematical formulation of this operation is given by :

O(i,j)=m=0Kh-1n=0Kw-1I(is \+ m \-p,j s \+ n \- p) K(m,n)

- I denotes the input feature map  
- K represents the convolution kernel of dimensions KhKw   
- s is the stride, which dictates the step size for sliding the kernel over the input  
- p corresponds to the amount of zero-padding added to the input border

The role of the stride and padding is essential in controlling both the resolution of the output feature map and the preservation of spatial information at the edges of the input. This equation encapsulates the local aggregation process, thereby enabling the network to build up complex representations by hierarchically combining simple features detected in the early layers.

# **OUTPUT DIMENSION CALCULATION**

In Convolutional Neural Networks, it is crucial to compute the dimensions of the output feature map resulting from the convolution operation. This computation ensures that the network architecture is correctly designed and that the spatial structure of the data is maintained or intentionally altered as needed. The output height and width are determined by four main parameters

\*Equation\*

Hout=

**SOURCES**

[https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)

**CONTRIBUTORS**

Sasha MARMAIN  
Killian OTT