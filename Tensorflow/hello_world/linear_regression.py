import tensorflow as tf

# train data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# input and output
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# parameters and init
W = tf.Variable([0.], tf.float32)
b = tf.Variable([0.], tf.float32)
y_ = W*x + b
init = tf.global_variables_initializer()

# loss and optimizer
loss = tf.reduce_sum(tf.square(y-y_))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# run the session
with tf.Session() as sess:
    sess.run(init)
    for _ in range(1000):
        sess.run(train, {x: x_train, y: y_train})
    print(sess.run([W, b, loss], {x: x_train, y: y_train}))