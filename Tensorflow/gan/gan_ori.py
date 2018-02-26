import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

mu_data = 3
sigma_data = 1.5
LENGTH = 1000


def data_sample(size, length=100):
    data_list = []
    for _ in range(size):
        data_unsorted = np.random.normal(mu_data, sigma_data, length)
        data_list.append(sorted(data_unsorted))
    return np.array(data_list)


def data_random(size, length=100):
    data_list = []
    for _ in range(size):
        data_list.append(np.random.random(length))
    return np.array(data_list)


def pre_process(x):
    return np.array([[np.mean(data), np.std(data)] for data in x])


input_D = tf.placeholder(dtype=tf.float32, shape=[None, 2])
label_D = tf.placeholder(dtype=tf.float32, shape=[None, 1])
input_G = tf.placeholder(dtype=tf.float32, shape=[None, LENGTH])

# build G net
# layer 1
W1_G = tf.Variable(tf.random_normal([LENGTH, 32]), name="W1_G")
b1_G = tf.Variable(tf.zeros([1, 32]) + 0.1, name="b1_G")
z1_G = tf.matmul(input_G, W1_G) + b1_G
o1_G = tf.nn.relu(z1_G)
# layer 2
W2_G = tf.Variable(tf.random_normal([32, 32]), name="W2_G")
b2_G = tf.Variable(tf.zeros([1, 32]) + 0.1, name="b2_G")
z2_G = tf.matmul(o1_G, W2_G) + b2_G
o2_G = tf.nn.sigmoid(z2_G)
# layer 3
W3_G = tf.Variable(tf.random_normal([32, LENGTH]), name="W3_G")
b3_G = tf.Variable(tf.zeros([1, LENGTH]) + 0.1, name="b3_G")
z3_G = tf.matmul(o2_G, W3_G) + b3_G
o3_G = z3_G
list_param_G = [W1_G, b1_G, W2_G, b2_G, W3_G, b3_G]

# build D net
W1_D = tf.Variable(tf.random_normal([2, 32]), name="W1_D")
b1_D = tf.Variable(tf.zeros([1, 32]) + 0.1, name="b1_D")
z1_D = tf.matmul(input_D, W1_D) + b1_D
o1_D = tf.nn.relu(z1_D)
# layer 2
W2_D = tf.Variable(tf.random_normal([32, 32]), name="W2_D")
b2_D = tf.Variable(tf.zeros([1, 32]) + 0.1, name="b2_D")
z2_D = tf.matmul(o1_D, W2_D) + b2_D
o2_D = tf.nn.sigmoid(z2_D)
# layer 3
W3_D = tf.Variable(tf.random_normal([32, 1]), name="W3_D")
b3_D = tf.Variable(tf.zeros([1, 1]) + 0.1, name="b3_D")
z3_D = tf.matmul(o2_D, W3_D) + b3_D
o3_D = tf.nn.sigmoid(z3_D)
list_param_D = [W1_D, b1_D, W2_D, b2_D, W3_D, b3_D]

# build D for GAN
mean_o3_G = tf.reduce_mean(o3_G, 1)
mean_T_o3_G = tf.transpose(tf.expand_dims(mean_o3_G, 0))
sigma_o3_G = tf.sqrt(tf.reduce_mean(tf.square(o3_G - mean_T_o3_G), 1))
input_D_GAN = tf.concat([mean_T_o3_G, tf.transpose(tf.expand_dims(sigma_o3_G, 0))], 1)

# layer 1
W1_D_GAN = tf.Variable(initial_value=tf.random_normal([2, 32]), name="W1_D_GAN")
b1_D_GAN = tf.Variable(initial_value=tf.zeros([1, 32]) + 0.1, name="b1_D_GAN")
z1_D_GAN = tf.matmul(input_D_GAN, W1_D_GAN) + b1_D_GAN
o1_D_GAN = tf.nn.relu(z1_D_GAN)
# layer 2
W2_D_GAN = tf.Variable(initial_value=tf.random_normal([32, 32]), name="W2_D_GAN")
b2_D_GAN = tf.Variable(initial_value=tf.zeros([1, 32]) + 0.1, name="b2_D_GAN")
z2_D_GAN = tf.matmul(o1_D_GAN, W2_D_GAN) + b2_D_GAN
o2_D_GAN = tf.nn.sigmoid(z2_D_GAN)
# layer 3
W3_D_GAN = tf.Variable(initial_value=tf.random_normal([32, 1]), name="W3_D_GAN")
b3_D_GAN = tf.Variable(initial_value=tf.zeros([1, 1]) + 0.1, name="b3_D_GAN")
z3_D_GAN = tf.matmul(o2_D_GAN, W3_D_GAN) + b3_D_GAN
o3_D_GAN = tf.nn.sigmoid(z3_D_GAN)
list_param_D_GAN = [W1_D_GAN, b1_D_GAN, W2_D_GAN, b2_D_GAN, W3_D_GAN, b3_D_GAN]

loss_D = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=o3_D, labels=label_D))
loss_GAN = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=o3_D_GAN, labels=label_D))

optimizer_D = tf.train.GradientDescentOptimizer(0.01).minimize(
    loss_D,
    global_step=tf.Variable(0),
    var_list=list_param_D
)

optimizer_GAN = tf.train.GradientDescentOptimizer(0.05).minimize(
    loss_GAN,
    global_step=tf.Variable(0),
    var_list=list_param_G
)

loss_history_D = []
loss_history_GAN = []
epoch = 80
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("train GAN")
    for step in range(epoch):
        for _ in range(100):
            data_real = data_sample(100, length=LENGTH)
            data_noise = data_random(100, length=LENGTH)
            data_generate = sess.run(o3_G, feed_dict={input_G: data_noise})
            X = list(data_real) + list(data_generate)
            X = pre_process(X)
            Y = [[1] for _ in range(len(data_real))] + [[0] for _ in range(len(data_generate))]
            value_loss_D, _ = sess.run([loss_D, optimizer_D], feed_dict={input_D: X, label_D: Y})
            loss_history_D.append(value_loss_D)

        result_list_param_D = sess.run(list_param_D)
        for i, v in enumerate(list_param_D_GAN):
            sess.run(v.assign(result_list_param_D[i]))

        for _ in range(100):
            data_noise = data_random(100, length=LENGTH)
            label_GAN = [[1] for _ in range(len(data_noise))]
            value_loss_GAN, _ = sess.run([loss_GAN, optimizer_GAN], feed_dict={input_G: data_noise, label_D: label_GAN})
            loss_history_GAN.append(value_loss_GAN)

        if step % 5 == 0 or step+1 == epoch:
             noise = data_random(1, length=LENGTH)
             generate = sess.run(o3_G, feed_dict={input_G: noise})
             result_list_param_G = sess.run(list_param_G)
             print(result_list_param_D)
             print("[%4d] GAN-d-loss: %.12f  GAN-g-loss: %.12f   generate-mean: %.4f   generate-std: %.4f" % (step,
                                                                                                              value_loss_D,
                                                                                                              value_loss_GAN,
                                                                                                              generate.mean(),
                                                                                                              generate.std()))
    plt.subplot(211)
    plt.plot(loss_history_D)
    plt.plot(loss_history_GAN, c="g")

    a = plt.subplot(212)

    real = data_sample(1, length=LENGTH)
    (data, bins) = np.histogram(real[0])
    plt.plot(bins[:-1], data, c="g")

    (data, bins) = np.histogram(noise[0])
    plt.plot(bins[:-1], data, c="b")

    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     generate = sess.run(G_output3, feed_dict={
    #             z: noise
    #     })
    (data, bins) = np.histogram(generate[0])
    plt.plot(bins[:-1], data, c="r")
    plt.show()
