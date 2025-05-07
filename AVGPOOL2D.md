## ABSTRACT

<p align="justify">
L'opération d'Average Pooling 2D est souvent utilisée dans les réseaux de neurones convolutionnels pour réduire la dimension spatiale des feature maps tout en capturant les informations moyennées des régions. Ce tutoriel présente les fondements mathématiques et pratiques de l'Average Pooling, ainsi que ses paramètres de configuration essentiels.
</p>

<br>

## INTRODUCTION

<p align="justify">
L'Average Pooling est une technique de sous-échantillonnage qui permet de réduire la taille des représentations intermédiaires dans un réseau de neurones tout en conservant une information moyenne sur des zones locales. Contrairement à la convolution qui pondère les entrées, l'Average Pooling calcule simplement la moyenne des pixels dans une fenêtre donnée.
</p>

<br>

## CONFIGURATION

| Paramètre           | Description                                                             | Effet trop élevé                                         | Effet trop faible                                             |
| :------------------ | :---------------------------------------------------------------------- | :------------------------------------------------------- | :------------------------------------------------------------- |
| kernel_size         | Dimensions de la fenêtre de pooling.                                    | Une fenêtre trop grande peut perdre des détails locaux.  | Une fenêtre trop petite peut ne pas réduire suffisamment les dimensions.  |
| stride              | Pas de déplacement de la fenêtre sur l'image d'entrée.                  | Sous-échantillonnage excessif, perte d'information spatiale. | Recouvrement important, moins de réduction de dimension.       |
| padding             | Nombre de pixels ajoutés autour de l'entrée avant pooling.              | Risque d'introduire des bordures artificielles.          | Dimension de sortie trop réduite.                              |
| ceil_mode (option)  | Utilisation du plafond dans le calcul de la dimension de sortie.         | Une sortie plus grande que prévu.                        | Une légère réduction supplémentaire des dimensions.          |

<br>

## AVGPOOL OPERATION

<p align="justify">
L'opération d'Average Pooling calcule la moyenne des valeurs dans chaque fenêtre définie par le kernel. Pour une feature map en entrée I, l'opération est formulée comme suit :
</p>

<br>

$$
O(i,j)=\frac{1}{K_h \times K_w}\sum_{m=0}^{K_h-1}\sum_{n=0}^{K_w-1}I(i\,s + m - p,\; j\,s + n - p)
$$

<br>

- \(K_h, K_w\) représentent la hauteur et la largeur de la fenêtre de pooling  
- \(s\) est le stride  
- \(p\) est le padding appliqué autour de l'entrée  

<br>

## OUTPUT DIMENSION CALCULATION

<p align="justify">
La dimension de la sortie après pooling est généralement calculée par :
</p>

<br>

$$
H_{out} = \left\lfloor \frac{H_{in} + 2p - K_h}{s} + 1 \right\rfloor\\[10pt]
W_{out} = \left\lfloor \frac{W_{in} + 2p - K_w}{s} + 1 \right\rfloor
$$

<br>

<p align="justify">
Notez que l'option <code>ceil_mode</code> peut être utilisée pour arrondir au plafond et obtenir des dimensions légèrement différentes.
</p>

<br>

## MULTI-CHANNEL POOLING

<p align="justify">
L'Average Pooling est appliqué de manière indépendante sur chaque canal de la feature map. Ainsi, pour une entrée comportant plusieurs canaux, l'opération est effectuée sur chacun sans mélange inter-canal.
</p>

<br>

## SOURCES

https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html

<br>

## CONTRIBUTORS
 
[Killian OTT](https://www.linkedin.com/in/killian-ott/)