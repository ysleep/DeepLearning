import tensorflow as tf

X_train = [1, 2, 3, 4]
Y_train = [0, -1, -2, -3]

x = tf.placeholder(dtype=tf.float32, name="x")
y = tf.placeholder(dtype=tf.float32, name="y")

W = tf.Variable([0.], dtype=tf.float32)
b = tf.Variable([0.], dtype=tf.float32)
init = tf.global_variables_initializer()

y_ = W * x + b
loss = tf.reduce_sum(tf.square(y-y_))

optimizor = tf.train.GradientDescentOptimizer(0.01)
train = optimizor.minimize(loss)

with tf.Session() as sess:
    sess.run(init)
    for _ in range(1000):
        sess.run(train, feed_dict={x: X_train, y: Y_train})
    print(sess.run([W, b]))

