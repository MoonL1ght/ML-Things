import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import dropout
import pandas as pd
import csv
from sklearn.model_selection import StratifiedKFold
from tensorflow.contrib.layers import fully_connected

train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv').as_matrix()

labels = train_data['label'].copy().as_matrix()
digits = train_data.drop('label', axis=1).as_matrix()

def create_val_set(data, labels):
  skf = StratifiedKFold(n_splits=10)
  for val_index, train_index in skf.split(data, labels):
    data_train = np.delete(data, train_index, axis=0)
    label_train = np.delete(labels, train_index, axis=0)
    data_val =np.delete(data, val_index, axis=0)
    label_val = np.delete(labels, val_index, axis=0)
    return data_train, label_train, data_val, label_val
  
pixel_depth = 255
t_digits, t_labels, v_digits, v_labels = create_val_set(digits, labels)
t_digits = t_digits.astype(float) / pixel_depth
v_digits = v_digits.astype(float) / pixel_depth
test_data = test_data.astype(float) / pixel_depth

image_size = 28
channels = 1
t_digits = np.reshape(t_digits, (-1, image_size, image_size, channels))
v_digits = np.reshape(v_digits, (-1, image_size, image_size, channels))
test_data = np.reshape(test_data, (-1, image_size, image_size, channels))

print('Train data shape:', t_digits.shape)
print('Validation data shape:', v_digits.shape)
print('Test data shape:', test_data.shape)

def fetch_batch(x, y, batch_index, batch_size):
  start = batch_index*batch_size
  end = batch_index*batch_size+batch_size
  x_batch = x[start:end]
  if y is not None:
    y_batch = y[start:end]
    return x_batch, y_batch
  else:
    return x_batch

X = tf.placeholder(tf.float32, shape=(None, image_size, image_size, channels))
y = tf.placeholder(tf.int32, shape=(None))
is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

class CNNModel():
  def __init__(self, learning_rate=0.001, conv_keep_prob=0.9, fc_keep_prob=0.5, num_classes=10):
    conv1 = self.create_conv(X, [3, 3, channels, 32], [1, 1, 1, 1], 'conv1')
    pool1 = tf.nn.max_pool(conv1,
      ksize=[1, 2, 2, 1],
      strides=[1, 2, 2, 1],
      padding='SAME',
      name='pool1')
    # drop1 = dropout(pool1, conv_keep_prob, is_training=is_training)

    conv2 = self.create_conv(pool1, [3, 3, 32, 32], [1, 1, 1, 1], 'conv2')
    pool2 = tf.nn.max_pool(conv2,
      ksize=[1, 2, 2, 1],
      strides=[1, 2, 2, 1],
      padding='SAME',
      name='pool2')
    # drop2 = dropout(pool2, conv_keep_prob, is_training=is_training)
    shape = int(np.prod(pool2.get_shape()[1:]))
    x_flat = tf.reshape(pool2, [-1, shape])

    fc1 = self.create_fully_connected(x_flat, 300, 'fc1', activation='relu')
#     bn1 = tf.contrib.layers.batch_norm(fc1, is_training=is_training)
    fcdrop1 = dropout(fc1, fc_keep_prob, is_training=is_training)
    self.y_pred = self.create_fully_connected(fcdrop1, num_classes, 'fc_out')
    self.prob_pred = tf.nn.softmax(self.y_pred)
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=self.y_pred)
    self.loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    self.training_op = optimizer.minimize(self.loss)
    self.correct = tf.nn.in_top_k(self.y_pred, y, 1)
    self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))

  def create_conv(self, x, kernel_shape, stride, name):
    with tf.name_scope(name):
      conv_kernel_init = tf.truncated_normal(kernel_shape, dtype=tf.float32, stddev=0.1)
      conv_kernel = tf.Variable(conv_kernel_init, name='weights_'+name)
      conv_bias = tf.Variable(tf.constant(0.0, shape=[kernel_shape[-1]], dtype=tf.float32),
        trainable=True, name='biases_'+name)
      conv = tf.nn.conv2d(x, conv_kernel, stride, padding='SAME')
      out = tf.nn.relu(tf.nn.bias_add(conv, conv_bias))
      return out

  def create_fully_connected(self, x, neurons, name, activation=None):
    with tf.name_scope(name):
      n_inputs = int(x.get_shape()[1])
      stddev = np.sqrt(2)*np.sqrt(2.0/(n_inputs+neurons))
      fc_init = tf.truncated_normal((n_inputs, neurons), dtype=tf.float32, stddev=stddev)
      fc_W =  tf.Variable(fc_init, name='fc_weights_'+name)
      fc_b = tf.Variable(tf.constant(0.0, shape=[neurons], dtype=tf.float32),
        trainable=True, name='fc_biases_'+name)
      z = tf.nn.bias_add(tf.matmul(x, fc_W), fc_b)
      if activation == 'relu':
        return tf.nn.relu(z)
      else:
        return z

  def train(self, sess, x_data, y_data, n_epoches=50, batch_size=100):
    n_batches = int(np.ceil(x_data.shape[0] / batch_size))
    for epoch in range(n_epoches):
      for batch_index in range(n_batches):
        x_batch, label_batch = fetch_batch(x_data, y_data,
          batch_index, batch_size)
        sess.run(self.training_op, feed_dict={X: x_batch, y: label_batch, is_training:True})
      loss_value = sess.run(self.loss, feed_dict={X: x_batch, y: label_batch, is_training:False})
      accuracy_value = sess.run(self.accuracy, feed_dict={X: x_batch, y: label_batch, is_training:False})
      print('Epoch: {}, batch accuracy: {:.5f}, batch loss: {:.5f}'\
        .format(epoch, (accuracy_value), loss_value))
      accuracy_valid = sess.run(self.accuracy, feed_dict={X: v_digits, y: v_labels, is_training:False})
      print('Epoch: {}, validation accuracy: {:.5f}'\
        .format(epoch, (accuracy_valid)))

  def predict(self, sess, x):
    return sess.run(self.prob_pred, feed_dict={X: x, is_training:False})

class Court():
  def __init__(self, jury, learning_rate=0.001, num_classes=10):
    n_jury = len(jury)
    self.jury = jury
    self.x = tf.placeholder(tf.float32, shape=(None, n_jury))
    self.y = tf.placeholder(tf.int32, shape=(None))
    h1 = fully_connected(self.x, 100, activation_fn='relu')
    out = fully_connected(h1, num_classes, activation_fn=None)
    self.prob_pred = tf.nn.softmax(self.y_pred)
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=self.y_pred)
    self.loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    self.training_op = optimizer.minimize(self.loss)
    self.correct = tf.nn.in_top_k(self.y_pred, y, 1)
    self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))

  def train(self, sess, x_data, y_data, n_epoches=50, batch_size=100):
    n_batches = int(np.ceil(x_data.shape[0] / batch_size))
    for epoch in range(n_epoches):
      for batch_index in range(n_batches):
        x_batch, label_batch = fetch_batch(x_data, y_data,
          batch_index, batch_size)
        # sess.run(self.training_op, feed_dict={X: x_batch, y: label_batch, is_training:True})
        label_preds = np.empty((0,0,0), np.float32)
        for model in self.jury:
          model_prob_pred = sess.run(model, feed_dict={X: v_digits, is_training:False})
          model_lbl_pred = sess.run(tf.argmax(model_prob_pred, axis=1))
          label_preds = np.append(label_preds, model_lbl_pred, axis=0)
        print(label_preds.shape)
      # loss_value = sess.run(self.loss, feed_dict={X: x_batch, y: label_batch, is_training:False})
      # accuracy_value = sess.run(self.accuracy, feed_dict={X: x_batch, y: label_batch, is_training:False})
      # print('Epoch: {}, batch accuracy: {:.5f}, batch loss: {:.5f}'\
      #   .format(epoch, (accuracy_value), loss_value))
      # accuracy_valid = sess.run(self.accuracy, feed_dict={X: v_digits, y: v_labels, is_training:False})
      # print('Epoch: {}, validation accuracy: {:.5f}'\
      #   .format(epoch, (accuracy_valid)))



model1 = CNNModel()
model2 = CNNModel()
model3 = CNNModel()
model4 = CNNModel()
model5 = CNNModel()
court = Court([model1, model2, model3, model4, model5])

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

model1.train(sess, t_digits, t_labels, n_epoches=1)
model2.train(sess, t_digits, t_labels, n_epoches=1)
model3.train(sess, t_digits, t_labels, n_epoches=1)
model4.train(sess, t_digits, t_labels, n_epoches=1)
model5.train(sess, t_digits, t_labels, n_epoches=1)
print('Ensemble Training Finished')

court.train(sess, t_digits, t_labels, n_epoches=1)




# print('Model 1 training')
# model1.train(sess, t_digits, t_labels, n_epoches=20)

# print('Model 2 training')
# model2.train(sess, t_digits, t_labels, n_epoches=20)

# label_prob_pred1 = model1.predict(sess, v_digits)
# label_prob_pred2 = model2.predict(sess, v_digits)

# correct_valid1 = sess.run(model1.correct, feed_dict={X: v_digits, y: v_labels, is_training:False})
# false_pred1 = np.array([i for i, x in enumerate(correct_valid1) if not x])
# print(false_pred1)

# correct_valid2 = sess.run(model2.correct, feed_dict={X: v_digits, y: v_labels, is_training:False})
# false_pred2 = np.array([i for i, x in enumerate(correct_valid2) if not x])
# print(false_pred2)



# both_false = np.array(list(set(false_pred1) & set(false_pred2)))
# ex_f = np.array(list(set(false_pred1) ^ set(false_pred2)))
# # for i, v in enumerate(false_pred1):
# #   if false_pred2[i] == v:
# #     both_fals
# print('==============')
# print(both_false)
# print(ex_f)

# np.set_printoptions(suppress=True)
# print(label_prob_pred1[both_false[0]])
# print(label_prob_pred2[both_false[0]])
# print(v_labels[both_false[0]])

# print('============')
# np.set_printoptions(suppress=True)
# print(label_prob_pred1[ex_f[0]])
# print(label_prob_pred2[ex_f[0]])
# print(v_labels[ex_f[0]])
     
     
# print('==========')   
# label_prob_pred = sess.run(prob_pred, feed_dict={X: v_digits, is_training:False})  
# label_pred = sess.run(tf.argmax(label_prob_pred, axis=1))
# correct_valid = sess.run(correct, feed_dict={X: v_digits, y: v_labels, is_training:False})
# #   correct_valid = correct_valid.astype(np.int)
# #   print(correct_valid)
# #   print(correct_valid[False])
# #   print(correct_valid[False].shape)
# false_pred = np.array([i for i, x in enumerate(correct_valid) if not x])
# true_pred = np.array([i for i, x in enumerate(correct_valid) if x])
# print(false_pred)
# print(false_pred.shape)
# print(true_pred)
# print(true_pred.shape)
# np.set_printoptions(suppress=True)
 
# label_pred_f = label_prob_pred[false_pred]
# label_pred_t = label_prob_pred[true_pred]
# #   print(label_pred_f)
# print(np.mean(np.std(label_prob_pred, axis=1)))
# mstd = np.mean(np.std(label_prob_pred, axis=1))
# sstd = np.std(np.std(label_prob_pred, axis=1))
# max_mstd = mstd + sstd
# min_mstd = mstd - sstd
# print(np.std(np.std(label_prob_pred, axis=1)))
# print(max_mstd)
# print(min_mstd)

# def detect_false(pred):
#   mstd = np.mean(np.std(pred, axis=1))
#   sstd = np.std(np.std(pred, axis=1))
#   min_mstd = mstd - sstd
#   false_pred = np.argwhere(np.std(pred, axis=1)<min_mstd)
#   return np.reshape(false_pred, -1)
 
# fp = detect_false(label_prob_pred)
# print(fp)
# print(fp.shape)

# print('-----------------')
# print('-----------------')

    
  
# print('true pred')
# print(np.mean(np.std(label_pred_t, axis=1)))
# #   for i in range(10):
# #     print(label_prob_pred[true_pred[i]])
# #     print(np.mean(label_prob_pred[true_pred[i]]))
# #     print(np.std(label_prob_pred[true_pred[i]]))
# #     print(np.var(label_prob_pred[true_pred[i]]))
# #     print(label_pred[true_pred[i]])
# #     print(v_labels[true_pred[i]])
# print('false pred')
# print(np.mean(np.std(label_pred_f, axis=1)))
# for i in range(10):
# #     print(label_prob_pred[false_pred[i]])
# #     print(np.mean(label_prob_pred[false_pred[i]]))
#   print(np.std(label_prob_pred[false_pred[i]]))
# #     print(np.var(label_prob_pred[false_pred[i]]))
# #     print(label_pred[false_pred[i]])
# #     print(v_labels[false_pred[i]])
    
# print('Training has finished')
# #   label_prob_pred = sess.run(prob_pred, feed_dict={X: test_data, is_training:False})
# #   label_pred = sess.run(tf.argmax(label_prob_pred, axis=1))
# #   with open('./res.csv', 'w', newline='') as csvfile:
# #     csvwriter = csv.writer(csvfile, delimiter=',')
# #     csvwriter.writerow(['ImageId', 'Label'])
# #     print(label_pred)
# #     for i, v in enumerate(label_pred):
# #       csvwriter.writerow([str(i+1), str(v)])