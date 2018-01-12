import tensorflow as tf

with tf.Session() as sess:
    m1 = tf.constant([[1], [2], [3]])
    m2 = tf.constant([[3, 4, 5], [6, 7, 8]])
    print(sess.run(m1+m2))