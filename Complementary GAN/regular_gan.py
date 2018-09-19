import tensorflow as tf

G_sample = generator(Z)
r_logits, r_rep = discriminator(X)
f_logits, g_rep = discriminator(G_sample,reuse=True)

class GAN:
  def __init__(self,
               dimension=2,
               gen_layers=[16, 16],
               gen_activation=tf.nn.leaky_relu,
               discr_layers=[16, 16],
               discr_activation=tf.nn.leaky_relu):
    self.dimension = dimension
    self.gen_layers = gen_layers
    self.gen_activation = gen_activation
    self.discr_layers = discr_layers
    self.discr_activation = discr_activation

  def generator(Z, dimension, layers, reuse=False):
    with tf.variable_scope("GAN/Generator", reuse=reuse):
      h1 = tf.layers.dense(Z, layers[0], activation=tf.nn.leaky_relu)
      h2 = tf.layers.dense(h1, layers[1], activation=tf.nn.leaky_relu)
      out = tf.layers.dense(h2, dimension)
    return out

  def discriminator(X, dimension, layers, reuse=False):
    with tf.variable_scope("GAN/Discriminator", reuse=reuse):
      h1 = tf.layers.dense(X, layers[0], activation=tf.nn.leaky_relu)
      h2 = tf.layers.dense(h1, layers[1], activation=tf.nn.leaky_relu)
      h3 = tf.layers.dense(h2, dimension)
      out = tf.layers.dense(h3, 1)
    return out, h3
