## ABSTRACT

<p align="justify">
Adaptive Max Pooling automatically adjusts pooling regions to output a fixed size, regardless of the input dimension. This tutorial introduces the concept and configuration of Adaptive Max Pooling 2D.
</p>

<br>

## INTRODUCTION

<p align="justify">
Adaptive Max Pooling is used to ensure that the output of a pooling layer has predetermined spatial dimensions. It dynamically partitions the input into regions and selects the maximum value from each region.
</p>

<br>

## OPERATION

<p align="justify">
Unlike fixed pooling, Adaptive Max Pooling computes pooling regions based on the desired output size (H_out, W_out). For each region, the operation outputs:
</p>

$$
O(i,j)= \max_{(m,n) \in R_{ij}} I(m,n)
$$

<br>

<p align="justify">
where R_{ij} represents the computed pooling region for each output element.
</p>

<br>

## CONFIGURATION

| Parameter         | Description                                                            | High output size effect                     | Low output size effect                |
| :---------------- | :--------------------------------------------------------------------- | :------------------------------------------ | :------------------------------------- |
| output_size       | Desired spatial dimensions of the output feature map.                 | Finer grain pooling with higher resolution. | Coarser feature extraction.            |

<br>

## OUTPUT DIMENSION CALCULATION

<p align="justify">
In Adaptive Max Pooling, the output dimensions (H_out, W_out) are set by the user. The pooling regions are computed to evenly partition the input, ensuring that:
</p>

<br>

$$
\text{Each region size} \approx \left(\frac{H_{in}}{H_{out}}, \frac{W_{in}}{W_{out}}\right)
$$

<br>

## SOURCES

PyTorch and Tensorflow documentations for Adaptive Pooling layers.

<br>

## CONTRIBUTORS

Killian OTT  
> <a href="https://www.linkedin.com/in/killian-ott/">
>  <img src="Logo/linkedin.png" alt="LinkedIn" width="20">
> </a>
