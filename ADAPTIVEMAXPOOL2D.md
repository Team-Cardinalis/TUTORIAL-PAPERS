## ABSTRACT

<p align="justify">
L'opération d'Adaptive Max Pooling 2D permet de fixer la dimension de sortie indépendamment de la taille de l'entrée. Ce tutoriel explique comment cette opération divise l'image d'entrée en régions adaptatives et extrait le maximum de chacune, garantissant une sortie de taille constante, ce qui est particulièrement utile dans les architectures de réseaux de neurones.
</p>

<br>

## INTRODUCTION

<p align="justify">
Adaptive Max Pooling est une technique de sous-échantillonnage qui, contrairement aux méthodes classiques basées sur une taille de kernel fixe, ajuste dynamiquement les régions à pooler afin d'obtenir une dimension de sortie prédéterminée. Cette propriété en fait un outil précieux pour gérer des entrées de tailles variables dans les architectures de deep learning.
</p>

<br>

## CONFIGURATION

| Paramètre    | Description                                                                  | Trop élevé effet                                        | Trop faible effet                                       |
| :----------- | :--------------------------------------------------------------------------- | :------------------------------------------------------ | :------------------------------------------------------ |
| output_size  | Dimension souhaitée de la sortie (hauteur et/ou largeur).                    | Sortie d'une taille trop grande, possible saturation. | Sortie d'une taille trop petite, perte d'information.  |
| return_indices (option) | Indique si les indices maximaux doivent être retournés.         | Augmente la mémoire nécessaire pour stocker les indices.| Moins flexible pour les opérations nécessitant un suivi. |

<br>

## ADAPTIVE MAX POOL OPERATION

<p align="justify">
L'opération d'Adaptive Max Pooling divise l'entrée en régions non chevauchantes de taille variable, puis sélectionne la valeur maximale dans chaque région. Pour une entrée donnée I et une sortie de dimensions fixées, l'opération est donnée par :
</p>

<br>

$$
O(i,j)= \max_{(m,n)\in R(i,j)} I(m,n)
$$

<br>

<p align="justify">
où R(i,j) représente la région d'entrée correspondant à la position (i,j) de la sortie. La division de l'entrée en régions se fait de manière à répartir équitablement les pixels, assurant ainsi que le résultat final ait exactement la dimension spécifiée.
</p>

<br>

## OUTPUT DIMENSION CALCULATION

<p align="justify">
Avec Adaptive Max Pooling, la dimension de sortie est directement déterminée par le paramètre <code>output_size</code>. Ainsi, quelle que soit la dimension de l'entrée, la sortie sera redimensionnée pour correspondre à cette valeur, éliminant le besoin de calculs complexes souvent associés aux opérations classiques de pooling.
</p>

<br>

## MULTI‑CHANNEL POOLING

<p align="justify">
Tout comme dans les opérations classiques de pooling, l'opération Adaptive Max Pooling est appliquée indépendamment sur chaque canal de l'entrée. Cela garantit que les caractéristiques de chaque canal sont préservées lors de la réduction de dimension.
</p>

<br>

## SOURCES

https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveMaxPool2d.html

<br>

## CONTRIBUTORS

[Sasha MARMAIN](https://www.linkedin.com/in/sasha-marmain-7a9645294/)  
[Killian OTT](https://www.linkedin.com/in/killian-ott/)
