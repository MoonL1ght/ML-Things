import tensorflow as tf

def generator(Z, layers, activation, reuse=False):
  with tf.variable_scope("CGAN/Generator", reuse=reuse):
    outputs = []
    for (i, layer) in enumerate(layers):
      if i == 0:
        outputs.append(tf.layers.dense(Z, layer, activation=activation))
      else:
        outputs.append(tf.layers.dense(outputs[-1], layer, activation=activation))
    out = tf.layers.dense(outputs[-1], Z.get_shape()[-1])
    logit = tf.nn.sigmoid(out)
  return logit, out

def discriminator(X, layers, activation, scope, reuse=False):
  with tf.variable_scope(scope, reuse=reuse):
    outputs = []
    for (i, layer) in enumerate(layers):
      if i == 0:
        outputs.append(tf.layers.dense(X, layer, activation=tf.nn.sigmoid))
      else:
        outputs.append(tf.layers.dense(outputs[-1], layer, activation=tf.nn.sigmoid))
    outputs.append(tf.layers.dense(outputs[-1], X.get_shape()[-1]))
    logit = tf.layers.dense(outputs[-1], 1)
    prob = tf.nn.sigmoid(logit)
  return prob, logit, outputs[-1]

class CGAN:
  def __init__(self,
               dimension=2,
               learning_rate=0.01,
               g_layers=[16, 16],
               g_activation=tf.nn.leaky_relu,
               d_layers=[16, 16],
               d_activation=tf.nn.leaky_relu,
               t_layers=[16, 16],
               t_activation=tf.nn.leaky_relu,
               optimizer=tf.train.RMSPropOptimizer):
    self.dimension = dimension
    self.lr = learning_rate
    self.g_layers = g_layers
    self.g_activation = g_activation
    self.d_layers = d_layers
    self.d_activation = d_activation
    self.t_layers = t_layers
    self.t_activation = t_activation
    self.optimizer = optimizer

    self.X = tf.placeholder(tf.float32, [None, self.dimension])
    self.Z = tf.placeholder(tf.float32, [None, self.dimension])
    self.T = tf.placeholder(tf.float32, [None, self.dimension])

    self.g_logit, self.g_samples = generator(self.Z, self.g_layers, self.g_activation)

    self.real_prob, self.real_logits, self.real_repr = discriminator(self.X,
      self.d_layers, self.d_activation, "CGAN/Discriminator")
    self.gen_prob, self.gen_logits, self.gen_repr = discriminator(self.g_logit,
      self.d_layers, self.d_activation, "CGAN/Discriminator", reuse=True)

    self.real_prob_t, self.real_logits_t, self.real_repr_t = discriminator(self.T,
      self.t_layers, self.t_activation, "CGAN/Discriminator_target")
    self.gen_prob_t, self.gen_logits_t, self.gen_repr_t = discriminator(self.g_logit,
      self.t_layers, self.t_activation, "CGAN/Discriminator_target", reuse=True)

    d_loss_real = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_logits,
        labels=tf.zeros_like(self.real_logits)))
    d_loss_gen = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=self.gen_logits,
        labels=tf.ones_like(self.gen_logits)))
    ent_real_loss = -tf.reduce_mean(tf.reduce_sum(
      tf.multiply(self.real_prob, tf.log(self.real_prob)), 1))
    self.d_loss = d_loss_real + d_loss_gen + 1.85 * ent_real_loss

    pt_loss = pull_away_loss(self.gen_repr_t)
    self.t_loss = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_logits_t,
        labels=tf.zeros_like(self.real_logits_t)))
    tar_thrld = tf.divide(tf.reduce_max(self.gen_prob_t[:,-1]) +
      tf.reduce_min(self.gen_prob_t[:,-1]), 2)
    indicator = tf.sign(tf.subtract(self.gen_prob_t[:,-1], tar_thrld))
    condition = tf.greater(tf.zeros_like(indicator), indicator)
    mask_tar = tf.where(condition, tf.zeros_like(indicator), indicator)
    g_ent_loss = tf.reduce_mean(tf.multiply(tf.log(self.gen_prob_t[:,-1]), mask_tar))
    fm_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(
      tf.square(self.real_logits - self.gen_logits), 1)))
    self.g_loss = pt_loss + g_ent_loss + fm_loss

    # self.g_loss = pt_loss + G_ent_loss + fm_loss

    # self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.gen_logits, labels=tf.ones_like(self.gen_logits)))

    g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="CGAN/Generator")
    d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="CGAN/Discriminator")
    t_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="CGAN/Discriminator_target")

    self.g_training_op = self.optimizer(learning_rate=self.lr).minimize(self.g_loss,
      var_list=g_vars)
    self.d_training_op = self.optimizer(learning_rate=self.lr).minimize(self.d_loss,
      var_list=d_vars)
    self.t_training_op = self.optimizer(learning_rate=self.lr).minimize(self.t_loss,
      var_list=t_vars)

  def train_step(self, sess, X_batch, Z_batch):
    _, dloss = sess.run([self.d_training_op, self.d_loss],
      feed_dict={self.X: X_batch, self.Z: Z_batch})
    _, gloss = sess.run([self.g_training_op, self.g_loss],
      feed_dict={self.X: X_batch, self.Z: Z_batch})
    return dloss, gloss

def pull_away_loss(g):
  Nor = tf.norm(g, axis=1)
  Nor_mat = tf.tile(tf.expand_dims(Nor, axis=1), [1, tf.shape(g)[1]])
  X = tf.divide(g, Nor_mat)
  X_X = tf.square(tf.matmul(X, tf.transpose(X)))
  mask = tf.subtract(tf.ones_like(X_X), tf.diag(tf.ones([tf.shape(X_X)[0]])))
  pt_loss = tf.divide(tf.reduce_sum(tf.multiply(X_X, mask)),
    tf.multiply(tf.cast(tf.shape(X_X)[0], tf.float32),
      tf.cast(tf.shape(X_X)[0]-1, tf.float32)))
  return pt_loss