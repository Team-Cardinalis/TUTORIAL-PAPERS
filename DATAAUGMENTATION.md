<a href="https://www.teamcardinalis.com/">
   <img src="Logo/teamcardinalis.png" alt="Team Cardinalis" width="100">
</a>

## ABSTRACT

<p align="justify">
Data augmentation is a foundational technique for enhancing the generalization and performance of deep learning models in computer vision by artificially expanding the training dataset. Designed for beginners, this tutorial provides a complete introduction—covering everything from the core principles of data augmentation to practical implementation and optimization using PyTorch. We explore widely used methods such as flipping, rotation, and cropping, along with color jittering and normalization, illustrating their impact through annotated code and side-by-side visual comparisons that highlight the transformations applied. By the end of this tutorial, readers will be able to build robust augmentation pipelines and make informed choices tailored to tasks like image classification and object detection.
</p>

<br>

## INTRODUCTION

<p align="justify">
This tutorial paper offers a thorough introduction to Data augmentation, a fundamental technique in deep learning for computer vision. Designed for beginners, it covers the conceptual foundations, key methods, and the vital role data augmentation plays in improving model performance and generalization.
</p>

<br>

## CONFIGURATION

| Parameter    | Description                                                           | Too high effect                                         | Too low effect                                                          |
| :----------- | :-------------------------------------------------------------------- | :------------------------------------------------------ | :---------------------------------------------------------------------- |
| flip  | Randomly flips the image horizontally and/or vertically.                                | Image becomes unrecognizable or confusing (e.g. mirrored text).         | Missed opportunity to increase data diversity.extraction.                                        |
| rotation | Rotates the image by a random degree within a specified range.      | Loss of orientation cues, distorted semantics.  |Limited robustness to rotated inputs.                                    |
| translation  | Shifts the image horizontally and/or vertically.          | Important regions moved out of frame.         | Reduced invariance to object location.patterns.                                  |
| zoom       | Randomly zooms in or out of the image.        | Cropped context or exaggerated features.   | Insufficient variation in object scale.over-decomposition.                               |
| shear      | Applies shear transformations that slant the image.     | Skewed images losing semantic meaning.          | Lack of diversity in geometric distortion.                                             |
| brightness     | Adjusts the image brightness randomly.| Washed out or overly dark images.        | Model overfits to fixed lighting conditions. capture.                      |
| color_jitter         | Randomly changes brightness, contrast, saturation, and hue.              | Unrealistic or inconsistent color profiles.  | Inadequate exposure to lighting/color variation.                             |
| crop         | Randomly crops a region from the image.              | Important features may be lost.  | Fewer variations in object location and framing.                             |
| padding         | Adds pixels around the image to simulate different input sizes.              | Artificial borders may mislead the model.  | Limited positional variance, possible loss of context.                             |
| noise_injection         | Adds random noise (e.g. Gaussian) to the image.              | Drowns signal with too much noise.  | No robustness to noisy or low-quality inputs.                             |
| normalization         | Scales pixel values to a standardized range.              | Potential information loss if improperly scaled.  | Unstable or slow model convergence due to unbalanced input.                             |



<br>

## CONVOLUTION OPERATION

<p align="justify">
In a Convolutional Neural Network, the convolution operation is the fundamental mechanism by which local features are extracted from input data. In this process, a kernel (or filter) of fixed dimensions is systematically applied to the input feature map by moving it across spatial positions. For each location (i,j) in the output feature map, the convolution operation computes a weighted sum of the values in the receptive field of the input. This is achieved by aligning the kernel with a corresponding patch of the input, performing an element-wise multiplication between the kernel weights and the input values, and summing the results.
</p>

The mathematical formulation of this operation is given by :

<br>

$$O(i,j)= \sum_{m=0}^{K_h-1}\sum_{n=0}^{K_w-1}I\bigl(i\,s + m - p,\; j\,s + n - p\bigr)\;K(m,n)$$

<br>

- $I$ denotes the input feature map
- $K$ represents the convolution kernel of dimensions $K_h \times K_w$
- $s$ is the stride, which dictates the step size for sliding the kernel over the input
- $p$ corresponds to the amount of zero-padding added to the input border

<p align="justify">
The role of the stride and padding is essential in controlling both the resolution of the output feature map and the preservation of spatial information at the edges of the input. This equation encapsulates the local aggregation process, thereby enabling the network to build up complex representations by hierarchically combining simple features detected in the early layers.
</p>

<br>

### VISUAL EXAMPLES

<p align="justify">
To illustrate how convolutional filters uncover image structure, we apply three edge detectors to the same grayscale input. The Sobel X kernel approximates the horizontal intensity derivative, causing vertical features to stand out. The Sobel Y kernel approximates the vertical derivative, making horizontal transitions more visible. The Laplacian kernel computes the second derivative of intensity and highlights every edge irrespective of orientation. In each example you will see the input image on the left, the kernel visualization in the center, and the resulting feature map on the right.
</p>

<br>

$$
O(i,j)= \sum_{m=0}^{2}\sum_{n=0}^{2}I\bigl(i\,s + m - p,\; j\,s + n - p\bigr)\;K(m,n)
\quad\text{with}\quad
K = \begin{bmatrix}
-1 & 0 & 1\\
-2 & 0 & 2\\
-1 & 0 & 1
\end{bmatrix}
$$

<br>

![Sobel X output](Figures/conv2d/conv/conv_sobel_x.png)  
<sub>Figure 1: **Sobel X** – vertical edges.</sub>

---

<br>

$$
O(i,j)= \sum_{m=0}^{2}\sum_{n=0}^{2}I\bigl(i\,s + m - p,\; j\,s + n - p\bigr)\;K(m,n)
\quad\text{with}\quad
K = \begin{bmatrix}
-1 & -2 & -1\\
0  &  0 &  0\\
1  &  2 &  1
\end{bmatrix}
$$

<br>

![Sobel Y output](Figures/conv2d/conv/conv_sobel_y.png)  
<sub>Figure 2: **Sobel Y** – horizontal edges.</sub>

---

<br>

$$
O(i,j)= \sum_{m=0}^{2}\sum_{n=0}^{2}I\bigl(i\,s + m - p,\; j\,s + n - p\bigr)\;K(m,n)
\quad\text{with}\quad
K = \begin{bmatrix}
0 & 1 & 0\\
1 & -4 & 1\\
0 & 1 & 0
\end{bmatrix}
$$

<br>

![Laplacian output](Figures/conv2d/conv/conv_laplacian.png)  
<sub>Figure 3: **Laplacian** – all edges.</sub>

---

<br>

$$
O(i,j)= \sum_{m=0}^{2}\sum_{n=0}^{2}I\bigl(i\,s + m - p,\; j\,s + n - p\bigr)\;K(m,n)
\quad\text{with}\quad
K = \begin{bmatrix}
-1 &  0 & 1\\
-1 &  0 & 1\\
-1 &  0 & 1
\end{bmatrix}
$$

<br>

![Prewitt X output](Figures/conv2d/conv/conv_prewitt_x.png)  
<sub>Figure 4: **Prewitt X** – vertical edges.</sub>

---

<br>

$$
O(i,j)= \sum_{m=0}^{2}\sum_{n=0}^{2}I\bigl(i\,s + m - p,\; j\,s + n - p\bigr)\;K(m,n)
\quad\text{with}\quad
K = \begin{bmatrix}
-1 & -1 & -1\\
 0 &  0 &  0\\
 1 &  1 &  1
\end{bmatrix}
$$

<br>

![Prewitt Y output](Figures/conv2d/conv/conv_prewitt_y.png)  
<sub>Figure 5: **Prewitt Y** – horizontal edges.</sub>

---

<br>

$$
O(i,j)= \sum_{m=0}^{1}\sum_{n=0}^{1}I\bigl(i\,s + m - p,\; j\,s + n - p\bigr)\;K(m,n)
\quad\text{with}\quad
K = \begin{bmatrix}
1 &  0\\
0 & -1
\end{bmatrix}
$$

<br>

![Roberts X output](Figures/conv2d/conv/conv_roberts_x.png)  
<sub>Figure 6: **Roberts X** – diagonal edges (↗︎ direction).</sub>

---

<br>

$$
O(i,j)= \sum_{m=0}^{1}\sum_{n=0}^{1}I\bigl(i\,s + m - p,\; j\,s + n - p\bigr)\;K(m,n)
\quad\text{with}\quad
K = \begin{bmatrix}
0 &  1\\
-1 & 0
\end{bmatrix}
$$

<br>

![Roberts Y output](Figures/conv2d/conv/conv_roberts_y.png)  
<sub>Figure 7: **Roberts Y** – diagonal edges (↘︎ direction).</sub>

---

<br>

$$
O(i,j)= \sum_{m=0}^{2}\sum_{n=0}^{2}I\bigl(i\,s + m - p,\; j\,s + n - p\bigr)\;K(m,n)
\quad\text{with}\quad
K = \begin{bmatrix}
-2 & -1 & 0\\
-1 &  1 & 1\\
 0 &  1 & 2
\end{bmatrix}
$$

<br>

![Emboss output](Figures/conv2d/conv/conv_emboss.png)  
<sub>Figure 8: **Emboss** – relief effect in a given direction.</sub>

---

<br>

$$
O(i,j)= \sum_{m=0}^{4}\sum_{n=0}^{4}I\bigl(i\,s + m - p,\; j\,s + n - p\bigr)\;K(m,n)
\quad\text{with}\quad
K = \begin{bmatrix}
0  & 0  & -1 & 0  & 0\\
0  & -1 & -2 & -1 & 0\\
-1 & -2 & 16 & -2 & -1\\
0  & -1 & -2 & -1 & 0\\
0  & 0  & -1 & 0  & 0
\end{bmatrix}
$$

<br>

![Laplacian of Gaussian output](Figures/conv2d/conv/conv_log.png)  
<sub>Figure 9: **LoG (Laplacian of Gaussian)** – combined smoothing and edge enhancement.</sub>

<br>

## OUTPUT DIMENSION CALCULATION

<p align="justify">
In Convolutional Neural Networks, it is crucial to compute the dimensions of the output feature map resulting from the convolution operation. This computation ensures that the network architecture is correctly designed and that the spatial structure of the data is maintained or intentionally altered as needed. The output height and width are determined by four main parameters:
</p>

<br>

$$
H_{out} = \frac{H_{in} + 2p - d \cdot (K_h - 1) - 1}{s} + 1
$$  
$$
W_{out} = \frac{W_{in} + 2p - d \cdot (K_w - 1) - 1}{s} + 1
$$

<br>

- $H_{in}, W_{in}$ denote the height and width of the input feature map  
- $H_{out}, W_{out}$ denote the height and width of the output feature map  
- $K_h, K_w$ represent the height and width of the kernel, respectively  
- $s$ is the stride, which dictates the step size for sliding the kernel over the input  
- $p$ corresponds to the amount of zero-padding added to the input border  
- $d$ is the dilation, which specifies the spacing between elements of the kernel

<br>

<img src="Figures/conv2d/out_dim_calc/output_dimensions.png" alt="Convolution output dimensions" width="400"/>  
<sub>Figure 8: <strong>Convolution Output Dimensions</strong> – Output size of a convolution operation as a function of kernel size (k), stride (s), padding (p), and dilation (d).</sub>

<br>
<br>

## MULTI-CHANNEL CONVOLUTION

<p align="justify">
In multi-channel convolution, each filter is applied to every input channel. The outputs are summed across channels to produce each output feature map. For an input with \(C_{in}\) channels, the operation is given by:
</p>

<br>

$$
O(i,j) = \sum_{c=1}^{C_{in}} \sum_{m=0}^{K_h-1}\sum_{n=0}^{K_w-1} I_c(i\,s + m - p,\; j\,s + n - p)\;K_c(m,n)
$$

<br>

## BIAS

<p align="justify">
After convolution, a learnable bias is added to each output channel. This allows the output to be shifted and is computed as:
</p>

<br>

$$
O(i,j) = \left( \sum_{c=1}^{C_{in}} \sum_{m=0}^{K_h-1}\sum_{n=0}^{K_w-1} I_c(i\,s + m - p,\; j\,s + n - p)\;K_c(m,n) \right) + b
$$

<br>

## BACKPROPAGATION

<p align="justify">
Backpropagation computes the gradients of the loss function with respect to each parameter using the chain rule. In convolutional layers, error gradients are propagated by convolving the gradient of the output with rotated filters.
</p>

<br>

## GRADIENT

<p align="justify">
The gradient with respect to the weights is given by:
</p>

<br>

$$
\frac{\partial L}{\partial K_c(m,n)} = \sum_{i,j} \frac{\partial L}{\partial O(i,j)} \cdot I_c(i\,s + m - p,\; j\,s + n - p)
$$

<br>

<p align="justify">
The gradient with respect to the bias is computed as:
</p>

<br>

$$
\frac{\partial L}{\partial b} = \sum_{i,j} \frac{\partial L}{\partial O(i,j)}
$$

<br>

## SOURCES

PyTorch documentation
> https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

<br>

Tensorflow documentation
> https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D

<br>

CS231n Deep Learning for Computer Vision
> http://cs231n.github.io/convolutional-networks/

<br>

Dive into Deep Learning
> https://d2l.ai/chapter_convolutional-neural-networks/conv.html

<br>

Deep Learning an MIT Press book
> https://www.deeplearningbook.org/

<br>

## CONTRIBUTORS

Sasha MARMAIN
> <a href="https://www.linkedin.com/in/sasha-marmain-7a9645294/">
>  <img src="Logo/linkedin.png" alt="LinkedIn" width="20">
> </a>

<br>

Killian OTT
> <a href="https://www.linkedin.com/in/killian-ott/">
>  <img src="Logo/linkedin.png" alt="LinkedIn" width="20">
> </a>
