## Brief Overview

The basic idea of the project is to identify various textures or patterns from a given image. There are several images and each of those images have a mixture of various patterns. The task is to segment these various textures or to segregate them so that it can be easily understood that how the textures are different. For example, in the picture provided below, a human eye can clearly see and understand that there are 3 different patterns in the image and our task is to segment each of them individually such that each of the segmented mask can be used to select a particular pattern. However, the problem is not very trivial to be solved by a computer algorithm as the patterns, although are repeating -to an extent, can come in various types and sizes.



The method implemented to solve the problem is by using Gaussian Mixture of Models and Expectation Maximization.

## Sample Input / Output



## plot of log-likelihood values vs number of components over 20 iterations
