# Variational Autoencoder (VAE) | 变分自编码器

### 选择语言 | Language

[中文简介](#简介) | [English](#introduction)

### 结果 | Result
从先验分布中随机采样生成的图像：
<img width="1200" height="1200" alt="samples" src="https://github.com/user-attachments/assets/0bc563c2-8641-44bc-ae3f-f44ac55745cc" />

训练损失曲线：
<img width="1050" height="600" alt="loss_curve" src="https://github.com/user-attachments/assets/f4359e40-13ce-4a4d-9c0d-8fc6996c6c2f" />

重建效果：
<img width="1680" height="480" alt="recon" src="https://github.com/user-attachments/assets/58a12108-1524-403a-9a3b-92efcf18247c" />

生成采样与潜在空间可视化（t-SNE）：
<img width="1050" height="900" alt="latent_space_tsne" src="https://github.com/user-attachments/assets/c6b4caa9-29f5-4855-b538-5d2bd0298a62" />

生成采样与潜在空间可视化（PCA）：
<img width="900" height="750" alt="latent_space" src="https://github.com/user-attachments/assets/4fdd3a15-ba71-4153-9531-27e3284d0331" />

---


## 简介

变分自编码器（Variational Autoencoder，简称 VAE）由 **Diederik P. Kingma 和 Max Welling** 于 2013 年提出，是深度生成模型领域的里程碑式工作。其核心贡献在于**将变分贝叶斯推断与深度神经网络完美结合，并提出了“重参数化技巧”（Reparameterization Trick）**，使得包含随机采样步骤的隐变量模型能够通过标准反向传播进行端到端训练。VAE 打破了传统自编码器（AE）仅能进行数据压缩和重建的局限，赋予了模型生成全新、逼真样本的能力，为后续的生成对抗网络（GAN）以及现今火热的扩散模型（Diffusion Models）奠定了重要的理论基础。

在 VAE 出现之前，传统的自编码器通过学习一个确定性的隐变量映射来完成数据重建，其潜在空间往往是不连续、不规则的，导致无法直接从中采样生成新数据。而 VAE 假设数据由一个不可观测的连续随机变量（潜变量）生成，通过编码器网络预测隐变量的均值（μ）和方差（σ²），强制潜在空间服从标准正态分布 N(0, I)。这种设计不仅保证了潜在空间的连续性与平滑性，也使得模型具备了极强的生成能力。

本复现基于 PyTorch 从零实现完整的 VAE 架构，严格遵循原论文的编码器-解码器对称结构、KL散度正则化与重参数化设计。采用 **MNIST 手写数字数据集** 进行验证，使用负变分下界（Negative ELBO）作为核心评估指标，完整包含数据预处理、模型训练、测试评估、图像重建、先验随机采样生成、以及**基于 t-SNE 的潜空间降维可视化**全流程。

## 架构

本次复现严格遵循原论文的标准架构，整体分为「编码器（Encoder）」「重参数化（Reparameterization）」「解码器（Decoder）」三大模块，完整保留了生成模型的所有核心数学推导。

- **编码器 q<sub>φ</sub>(z|x)**：输入图像首先被展平为一维向量（如 MNIST 的 784 维）。通过多层感知机（MLP）映射到低维潜在空间，输出两个独立向量：**均值 μ** 和 **方差对数 logσ²**。这一设计将输入数据编码为一个连续的多元高斯分布，而非一个固定的点。
- **重参数化技巧（Reparameterization Trick）**：这是 VAE 能够成功训练的绝对核心。由于从高斯分布中采样 z ~ N(μ, σ²) 是随机操作，梯度无法反向传播。代码中通过引入独立的标准高斯噪声 ε ~ N(0, I)，将采样操作重构为确定性的计算：**z = μ + ε ⊙ exp(0.5 · logσ²)**。这使得梯度能够穿过采样节点，顺利回流至编码器。
- **解码器 p<sub>θ</sub>(x|z)**：将采样得到的隐变量 z 送入解码器（MLP），映射回原始图像维度（如 784 维），并在最后一层使用 **Sigmoid** 激活函数，将输出约束在 [0, 1] 范围内，作为像素属于伯努利分布的均值预测。
- **损失函数（ELBO）**：VAE 的优化目标是最大化证据下界（ELBO），等价于最小化**负 ELBO**。其包含两部分：
  - **重建损失（BCE）**：强制解码器生成的图像与原图尽可能一致。
  - **KL 散度（KL Divergence）**：衡量编码器输出的分布与标准正态分布之间的差异。KL 散度起到了极强的正则化作用，迫使所有样本的特征分布均匀且平滑地铺满整个潜在空间，从而填补了潜在空间的“空洞”。

**注意**：为降低训练开销、便于快速复现验证，本实现采用轻量化 Mini-VAE 配置（隐藏层维度 400、潜在维度 20），数据集使用 MNIST 小型图像语料，但模型结构、ELBO 计算、重参数化设计、训练范式均严格对齐原论文标准。

## 数据集

本次复现使用 **MNIST 手写数字数据集**，是深度学习和生成模型最经典的基准数据集。

- **任务类型**：图像重建与无条件生成任务。
- **数据规模**：训练集 60000 张图像，测试集 10000 张图像；所有图像均为 28×28 分辨率的单通道灰度图片，包含 0 到 9 共 10 个手写数字类别。
- **预处理方式**：仅使用 `ToTensor()` 将像素值标准化到 [0, 1] 范围，不应用任何翻转或裁剪等数据增强（因为生成任务要求模型学习原始数据的真实流形分布）。
- **评估指标**：采用生成模型领域的通用标准指标 **负变分下界（Negative ELBO，Nats/Sample）**。该指标包含重建误差（Binary Cross Entropy）和 KL 散度，数值越低代表重建越精确且隐变量分布越规范。

| 负 ELBO 损失 | 效果说明 |
| --- | --- |
| 110 ~ 120 | 模型刚开始收敛，重建图像模糊，KL散度未充分下降 |
| 90 ~ 100 | 具备基础生成与重建能力，数字轮廓可辨认，潜空间较连续 |
| 80 ~ 90 | 高精度重建，数字清晰，KL 散度收敛，随机采样生成的图像有一定可读性 |
| < 80 | 优秀的重建性能，接近该全连接架构上的性能上限（进一步需引入卷积结构） |

数据集官方地址：[http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

---

## Introduction

The Variational Autoencoder (VAE), proposed by **Diederik P. Kingma and Max Welling** in 2013, is a milestone work in the field of deep generative models. Its core contribution is that it **perfectly combines variational Bayesian inference with deep neural networks and introduces the "Reparameterization Trick"**, making latent variable models containing stochastic sampling steps trainable end-to-end through standard backpropagation. VAE breaks the limitation of traditional Autoencoders (AE) which can only perform data compression and reconstruction, endowing models with the ability to generate novel and realistic samples. It laid a crucial theoretical foundation for subsequent Generative Adversarial Networks (GANs) and today's Diffusion Models.

Before VAE, traditional autoencoders learned deterministic latent mappings for data reconstruction. Their latent spaces were often discontinuous and irregular, making it impossible to sample directly to generate new data. VAE assumes that data is generated from unobservable continuous random variables (latent variables). Through the encoder network, it predicts the mean (μ) and variance (σ²) of the latent variables, forcing the latent space to follow a standard normal distribution N(0, I). This design not only ensures the continuity and smoothness of the latent space but also equips the model with strong generative capabilities.

This reproduction implements the complete VAE architecture from scratch based on PyTorch, strictly following the original paper's symmetric encoder-decoder structure, KL divergence regularization, and reparameterization design. It is validated on the **MNIST handwritten digits dataset** and uses the negative Evidence Lower Bound (Negative ELBO) as the core evaluation metric, covering the full pipeline of data preprocessing, model training, test evaluation, image reconstruction, prior sampling generation, and **t-SNE based latent space dimensionality reduction visualization**.

## Architecture

This implementation faithfully follows the standard architecture from the original paper, consisting of three main modules: "Encoder", "Reparameterization", and "Decoder", with all core mathematical derivations of the generative model preserved.

- **Encoder q<sub>φ</sub>(z|x)**: The input image is first flattened into a 1D vector (e.g., 784 dimensions for MNIST). Through a Multi-Layer Perceptron (MLP), it is mapped to a low-dimensional latent space, outputting two independent vectors: **mean μ** and **log-variance logσ²**. This design encodes the input data as a continuous multivariate Gaussian distribution rather than a fixed point.
- **Reparameterization Trick**: This is the absolute core of VAE's successful training. Since sampling z ~ N(μ, σ²) from a Gaussian distribution is stochastic, gradients cannot backpropagate through it. The code introduces independent standard Gaussian noise ε ~ N(0, I) and reconstructs the sampling operation as a deterministic computation: **z = μ + ε ⊙ exp(0.5 · logσ²)**. This allows gradients to flow back to the encoder seamlessly.
- **Decoder p<sub>θ</sub>(x|z)**: The sampled latent variable z is fed into the decoder (MLP), mapped back to the original image dimension (e.g., 784), and a **Sigmoid** activation function is applied at the final layer to constrain the output to [0, 1], serving as the mean prediction of a Bernoulli distribution for the pixels.
- **Loss Function (ELBO)**: The optimization objective of VAE is to maximize the Evidence Lower Bound (ELBO), which is equivalent to minimizing the **Negative ELBO**. It consists of two parts:
  - **Reconstruction Loss (BCE)**: Forces the decoder's generated images to be as consistent as possible with the originals.
  - **KL Divergence**: Measures the difference between the encoder's output distribution and the standard normal distribution. It acts as strong regularization, forcing all sample feature distributions to spread smoothly and evenly across the latent space, thereby filling the "holes" in the latent space.

**Note**: To reduce training cost and facilitate quick reproduction and verification, this implementation uses a lightweight Mini-VAE configuration (hidden dim 400, latent dim 20) and the small MNIST image corpus. However, model structure, ELBO computation, reparameterization design, and training paradigm are all strictly aligned with the original paper standard.

## Dataset

This reproduction uses the **MNIST handwritten digits dataset**, the most classic benchmark in deep learning and generative models.

- **Task**: Image reconstruction and unconditional generation task.
- **Data Scale**: 60,000 training images and 10,000 test images. All images are 28×28 resolution single-channel grayscale pictures, containing 10 handwritten digit categories from 0 to 9.
- **Preprocessing**: Only `ToTensor()` is used to normalize pixel values to the [0, 1] range. No data augmentation like flipping or cropping is applied (since generative tasks require the model to learn the true manifold distribution of the original data).
- **Evaluation Metric**: The universal standard metric for generative models, **Negative Evidence Lower Bound (Negative ELBO, Nats/Sample)**. This metric includes reconstruction error (Binary Cross Entropy) and KL divergence. Lower values indicate more accurate reconstruction and more standardized latent variable distributions.

Official dataset page: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

---

## 原文章 | Original article

Kingma D P, Welling M. Auto-encoding variational bayes[C]//International Conference on Learning Representations (ICLR). 2014.
