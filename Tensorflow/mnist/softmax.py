import Tensorflow.mnist.input_data as input_data
import tensorflow as tf
import numpy as np

ROWs = 28
COLs = 28
NUM_LABELs = 10

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# initial placeholder
x = tf.placeholder(dtype=tf.float32, shape=([ROWs*COLs, None]))
y = tf.placeholder(dtype=tf.float32, shape=([NUM_LABELs, None]))

# initial parameters
W = tf.Variable(dtype=tf.float32, initial_value=tf.zeros([NUM_LABELs, ROWs*COLs]))
b = tf.Variable(dtype=tf.float32, initial_value=tf.zeros([NUM_LABELs, 1]))
init_var = tf.global_variables_initializer()

# build graph
y_ = tf.nn.softmax(tf.matmul(W, x)+b)
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_), reduction_indices=[0]))
train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# accuracy
num_correct_prediction = tf.equal(tf.argmax(y, axis=0), tf.argmax(y_, axis=0))
accuracy = tf.reduce_mean(tf.cast(num_correct_prediction, dtype=tf.float32))

with tf.Session() as sess:
    sess.run(init_var)
    for _ in range(2000):
        x_batch, y_batch = mnist.train.next_batch(200)
        sess.run(train, feed_dict={x: np.transpose(x_batch), y: np.transpose(y_batch)})

    print(sess.run(accuracy, feed_dict={x: np.transpose(mnist.test.images), y: np.transpose(mnist.test.labels)}))









# 获得数据集
# train set shape: (num_images, rows * cols)
# label set shape: (num_labels, 10)
# W shape: (rows * cols, 10)
# b shape: (1, 10)
# tf.matmul(x, W) shape: (num_images, 10)
'''
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(dtype=tf.float32, shape=([None, LENGTH]), name="x")
y = tf.placeholder(dtype=tf.float32, shape=([None, 10]), name="y")

W = tf.Variable(initial_value=tf.zeros([LENGTH, 10]), dtype=tf.float32, expected_shape=None)
b = tf.Variable(initial_value=tf.zeros([1, 10]), dtype=tf.float32, expected_shape=None)
init = tf.global_variables_initializer()

y_ = tf.nn.softmax(tf.matmul(x, W) + b)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

print("start")

with tf.Session() as sess:
    sess.run(init)
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist .test.labels}))
'''