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

## DATA AUGMENTATION OPERATION

<p align="justify">
In machine learning pipelines, Data augmentation is a fundamental technique used to artificially expand the training dataset by generating modified versions of existing data samples. This is achieved by applying a set of transformations—such as rotation, scaling, flipping, cropping, or noise injection—to the original inputs. For each training instance, these transformations alter certain spatial, color, or statistical properties while preserving the core semantics of the data. By systematically introducing such variations, data augmentation helps the model generalize better to unseen data, reduces overfitting, and improves robustness to real-world variations.
</p>

## Here is the Maths

## 1. Normalization
Rescales pixel values to a standard range, typically [0, 1] or mean-centered.

**Equation:**

$$
x' = \frac{x - \mu}{\sigma}
$$

Where:
- \( x \) is the original pixel value.
- \( \mu \) is the dataset mean.
- \( \sigma \) is the standard deviation.

---

## 2. Noise Injection
Adds random noise to the input to improve robustness.

**Equation:**

$$
x' = x + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
$$

---

## 3. Padding
Adds extra pixels around the image, usually with zeros or reflected edges.

**Operation (Conceptual):**

$$
x' = \text{pad}(x, p)
$$

Where:
- \( p \) is the number of pixels to pad on each side.

---

## 4. Crop
Removes parts of the image, typically from edges.

**Operation (Conceptual):**

$$
x' = x[i:i+h,\ j:j+w]
$$

Where:
- \( h, w \) are the crop height and width.
- \( i, j \) are the starting row and column.

---

## 5. Color Jitter
Randomly changes brightness, contrast, saturation, and hue.

**Equation (general form):**

$$
x' = \alpha \cdot x + \beta
$$

Where:
- \( \alpha \) controls contrast.
- \( \beta \) shifts brightness.

---

## 6. Brightness
Adjusts image intensity.

**Equation:**

$$
x' = x + \Delta b
$$

Where:
- \( \Delta b \) is a brightness offset.

---

## 7. Shear
Applies a slant to the image shape.

**Equation (Shear in x-direction):**

$$
\begin{bmatrix}
x' \\
y'
\end{bmatrix}
=
\begin{bmatrix}
1 & \lambda \\
0 & 1
\end{bmatrix}
\begin{bmatrix}
x \\
y
\end{bmatrix}
$$

Where:
- \( \lambda \) is the shear factor.

---

## 8. Zoom (Scaling)
Scales the image up or down.

**Equation:**

$$
x' = s_x \cdot x,\quad y' = s_y \cdot y
$$

Where:
- \( s_x, s_y \) are scale factors along x and y axes.

---

## 9. Translation
Shifts the image in space.

**Equation:**

$$
x' = x + t_x,\quad y' = y + t_y
$$

Where:
- \( t_x, t_y \) are translation values.

---

## 10. Rotation
Rotates the image around the center.

**Equation:**

$$
\begin{bmatrix}
x' \\
y'
\end{bmatrix}
=
\begin{bmatrix}
\cos(\theta) & -\sin(\theta) \\
\sin(\theta) & \cos(\theta)
\end{bmatrix}
\begin{bmatrix}
x \\
y
\end{bmatrix}
$$

Where:
- \( \theta \) is the rotation angle.

---

## 11. Flip
Reverses the image along a specific axis.

**Horizontal Flip:**

$$
x' = W - x,\quad y' = y
$$

**Vertical Flip:**

$$
x' = x,\quad y' = H - y
$$

Where:
- \( W \) is the image width.
- \( H \) is the image height.   

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
