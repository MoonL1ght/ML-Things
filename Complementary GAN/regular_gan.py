import tensorflow as tf

class GAN:
  def __init__(self,
               dimension=2,
               learning_rate=0.01,
               gen_layers=[16, 16],
               gen_activation=tf.nn.leaky_relu,
               discr_layers=[16, 16],
               discr_activation=tf.nn.leaky_relu,
               optimizer=tf.train.RMSPropOptimizer):
    self.dimension = dimension
    self.lr = learning_rate
    self.gen_layers = gen_layers
    self.gen_activation = gen_activation
    self.discr_layers = discr_layers
    self.discr_activation = discr_activation
    self.optimizer = optimizer

    self.X = tf.placeholder(tf.float32, [None, self.dimension])
    self.Z = tf.placeholder(tf.float32, [None, self.dimension])

    self.gen_samples = GAN.generator(self.Z, self.dimension, self.gen_layers, self.gen_activation)
    self.real_logits, self.real_repr = GAN.discriminator(self.X, self.dimension, self.discr_layers, self.discr_activation)
    self.gen_logits, self.gen_repr = GAN.discriminator(self.gen_samples, self.dimension, self.discr_layers, self.discr_activation, reuse=True)

    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_logits, labels=tf.ones_like(self.real_logits)))
    D_loss_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.gen_logits, labels=tf.zeros_like(self.gen_logits)))
    self.D_loss = D_loss_real + D_loss_gen
    self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.gen_logits, labels=tf.ones_like(self.gen_logits)))

    gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Generator")
    disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Discriminator")

    self.gen_training_op = self.optimizer(learning_rate=self.lr).minimize(self.G_loss, var_list=gen_vars)
    self.disc_training_op = self.optimizer(learning_rate=self.lr).minimize(self.D_loss, var_list=disc_vars)

  def train_step(self, sess, X_batch, Z_batch):
    _, dloss = sess.run([self.disc_training_op, self.D_loss], feed_dict={self.X: X_batch, self.Z: Z_batch})
    _, gloss = sess.run([self.gen_training_op, self.G_loss], feed_dict={self.Z: Z_batch})
    return dloss, gloss

  def generator(Z, dimension, layers, activation, reuse=False):
    with tf.variable_scope("GAN/Generator", reuse=reuse):
      h1 = tf.layers.dense(Z, layers[0], activation=activation)
      h2 = tf.layers.dense(h1, layers[1], activation=activation)
      out = tf.layers.dense(h2, dimension)
    return out

  def discriminator(X, dimension, layers, activation, reuse=False):
    with tf.variable_scope("GAN/Discriminator", reuse=reuse):
      h1 = tf.layers.dense(X, layers[0], activation=activation)
      h2 = tf.layers.dense(h1, layers[1], activation=activation)
      h3 = tf.layers.dense(h2, dimension)
      out = tf.layers.dense(h3, 1)
    return out, h3