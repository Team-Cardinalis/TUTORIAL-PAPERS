## ABSTRACT

<p align="justify">
Average Pooling reduces the spatial dimensions of feature maps by computing the average value over non-overlapping regions. This tutorial provides an introduction to Average Pooling 2D, explaining its mathematical basis, configuration parameters, and application in deep neural networks.
</p>

<br>

## INTRODUCTION

<p align="justify">
This document explains the Average Pooling 2D layer in deep learning. Unlike convolution, it aggregates information by averaging values over a window, which helps reduce spatial dimensions and control overfitting.
</p>

<br>

## OPERATION

<p align="justify">
For a given pooling window of size K × K, the output at position (i,j) is calculated as the average of input values:
</p>

<br>

$$
O(i,j)= \frac{1}{K \times K}\sum_{m=0}^{K-1}\sum_{n=0}^{K-1} I\bigl(i\,s + m,\; j\,s + n\bigr)
$$

<br>

<p align="justify">
This operation is applied with a stride (s) that determines the step between pooling regions.
</p>

<br>

## CONFIGURATION

| Parameter    | Description                                                       | Large kernel effect                                    | Small kernel effect                                   |
| :----------- | :---------------------------------------------------------------- | :----------------------------------------------------- | :---------------------------------------------------- |
| kernel_size  | Dimensions of the pooling window.                                 | Excessive smoothing, loss of details.                  | Less spatial reduction.                             |
| stride       | Steps to move the window over the input feature map.              | Over-smoothing and aggressive downsampling.            | Reduced downsampling effect.                        |
| padding      | Extra pixels added to the input borders before pooling.           | Can lead to boundary bias in pooling results.          | Minimal influence if set to zero.                   |

<br>

## OUTPUT DIMENSION CALCULATION

<p align="justify">
The output dimensions are determined similarly to convolution:
</p>

<br>

$$
H_{out} = \left\lfloor \frac{H_{in} + 2p - K}{s} + 1 \right\rfloor \quad,\quad
W_{out} = \left\lfloor \frac{W_{in} + 2p - K}{s} + 1 \right\rfloor
$$

<br>

## SOURCES

PyTorch documentation
> https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html

<br>

## CONTRIBUTORS

Killian OTT  
> <a href="https://www.linkedin.com/in/killian-ott/">
>  <img src="Logo/linkedin.png" alt="LinkedIn" width="20">
> </a>
