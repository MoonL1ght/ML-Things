# Complementary GAN

Generative adversarial networks (GAN) consist of two models: a generative model and a discriminative model.

The discriminator model is a classifier that determines whether a given sample looks like a real sample from the dataset or like an artificially created sample. This is basically a binary classifier.

The generator model takes random input values and transforms them into samples.

Based on [article](https://arxiv.org/pdf/1803.01798.pdf)

Generated data is a simple spherical 1 class data.

*regular_gan.py - regular gan network architecture*
*complementary_gan.py - complementary gan network architecture*
*generate_data.py - generating nested spherical artificial data*

jupyter notebook v4.4.0
python v3.6.5
conda v4.5.4
tensorflow v1.10.0