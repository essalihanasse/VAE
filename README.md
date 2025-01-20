# Variational Autoencoder Implementation

This repository contains an implementation of a Variational Autoencoder (VAE) as described in "An Introduction to Variational Autoencoders." The implementation demonstrates the core concepts of VAEs including the encoder-decoder architecture, latent space representation, and the reparameterization trick.

## Overview

Variational Autoencoders are deep learning models that learn to encode data into a compressed latent representation and then decode it back to reconstruct the original input. Unlike traditional autoencoders, VAEs learn a probabilistic mapping that enables both data compression and generation of new samples.

Key features of this implementation:
- Encoder network that maps input data to latent space parameters (μ and σ)
- Reparameterization trick for backpropagation through random sampling
- Decoder network that reconstructs input data from latent representations
- Implementation of the VAE loss function (reconstruction loss + KL divergence)

## Requirements

```
pytorch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.2
matplotlib>=3.3.4
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```


