import Tensorflow.mnist.input_data as input_data
import tensorflow as tf

# 获得数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 占位符 x作为输入图像，y_作为label
# None 表示任意长度
x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, 10])

# 定义变量
# y = softmax(Wx+b)
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W)+b)

# 定义损失函数和反向传播算法
# softmax使用多项分布的最大似然概率作为损失函数
# 使用梯度下降算法
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 初始化变量
init = tf.global_variables_initializer()

# 构建Session
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
