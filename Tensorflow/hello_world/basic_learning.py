import tensorflow as tf

# 图的节点被称为op(operation)
# 一个op获得0个或多个Tensor, 执行计算, 产生0个或多个Tensor
# 构建源op(source op)
# 常量op
matrix1 = tf.constant([[3., 3.]]) #注意此处 matrix1 必须是一个二维数组，两个中括号
matrix2 = tf.constant([[2.], [2.]])

# 创建一个矩阵乘法 matmul op , 把 'matrix1' 和 'matrix2' 作为输入.
# 返回值 'product' 代表矩阵乘法的结果.
product = tf.matmul(matrix1, matrix2)

## 启动默认图
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

# 上下文管理协议：包含方法 __enter__() 和 __exit__()
# 上下文管理器：支持上下文管理协议的对象
# with context_expression [as target(s)]: context_expression返回上下文管理器对象，该对象的 __enter__() 方法的返回值赋给target(s)，语句体完成后调用 __exit__() 方法
with tf.Session() as sess:
    result = sess.run(product)
    print(result)

## 变量
# 创建一个变量, 初始化为标量 0.
state = tf.Variable(0, name="counter")

# 创建一个op, 作用是使state增加1
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# 启动图后, 变脸需要经过'初始化'op进行初始化
init_op = tf.global_variables_initializer()

# 启动图, 运行op
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(state))
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

## Fetch
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul = tf.multiply(input1, intermed)

with tf.Session() as sess:
    result = sess.run([intermed, mul])
    print(result)


## Feed
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run([output], feed_dict={input1: [7.], input2: [2.]}))
