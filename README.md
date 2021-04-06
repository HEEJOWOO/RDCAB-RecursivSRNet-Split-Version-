## Single Image Super Resolution Based on Residual Dense Channel Attention Block-RecursiveSRNet (IPIU, 2021.02.03~05) [YouTube](https://www.youtube.com/watch?v=BW7Z-MUu7m4) [IPIU](http://www.ipiu.or.kr/2021/index.php)

# RDCAB-RecursivSRNet(Split-Version)
## Abstract
RDCAB-RecursiveSRNet, which was submitted to the existing IPIU conference, has about 10 times less number of parameters than RDN, and has been reduced by 1.7 times from 1309G to 750G based on a 4x magnification factor, even in terms of Multi-Adds. However, the recursion structure increases or widens the depth of the network to prevent recursion and loss of recursion, and thus requires a lot of computation and memory consumption, which is difficult to apply to a real-time or low-power computer. Therefore, I am continuously researching to lighten RDCAB-RecursiveSRNet, and the final goal is to mount it on an embedded board.
* Additional information : Transmitting information from large and heavy models to small and light models to train them to make more accurate inferences.
## Differences from existing RDCAB-RecursvieSRNet
1) The input image is made at the same magnification as the output using the bicubic interpolation method, and the final reconstructed image and the elementwise sum are performed.
2) After specifying the splitting ratio using the split operation of the information distillation mechanism, the features input to the block are divided into 16 as retain features and 48 as refine features, and the refine features are used to extract features continuously. Finally, the retain features extracted hierarchically are concated.
3) A technique called Channel Attention has been widely used to make better use of useful information, and when a large number of filters are stacked, a large number of parameters follow, and when the number of parameters increases, an over-fitting problem occurs during learning, which prevents pooling. It has been mainly used to reduce the dimensionality by reducing the number of parameters used in the filter. In the case of global average pooling using this, it is a technique introduced as a method to eliminate the fully connected layer normally used in classifier by reducing the number of features more rapidly than conventional pooling, that is, making it a one-dimensional vector. Existing channel attention is more suitable for high level, that is, detection or classification, and the global average pooling used for channel attention uses global information, although it can increase the value of PSNR, it saves texture or edge when used for low level SR. It is said that it was confirmed that the structural similarity was rather low due to lack of information. Therefore, contrast aware channel attention was used to replace the existing global average pooling with the sum of the mean and variance by using a method of spreading the pixel distribution of an image called contrast over a wider area.
4) With reference to AWSRN, the upsample process was configured with Adaptive Weight Multi Scale (AWMS), and it was confirmed that the use of AWMS structures of 3x3 Conv, 5x5 Conv, 7x7 Conv, and 9x9 Conv is not much different from using only 3x3 Conv. Therefore, 3x3 Conv and independent weights were used.


## Experiments
Train : DIV2K
Test : Set5, Set14, BSD100, Urban100

|x4|Set5/ProcessTime|Set14/ProcessTime|BSD100/ProcessTime|Urban100/ProcessTime|
|--|----------------|-----------------|------------------|--------------------|
|RDN|32.47 / 0.1573|28.81 / 0.1920|27.72 / 0.021|26.61 / 0.22