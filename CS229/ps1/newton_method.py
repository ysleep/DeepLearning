import numpy as np
import matplotlib.pyplot as plt

# 载入数据
LOGISITIC_X_FILE = "data/logistic_x.txt"
LOGISITIC_Y_FILE = "data/logistic_y.txt"


# 数据预处理, 所有输入feature为列向量, 保留常数项1
# x.shape (length_feature+1, nums_data)
# y.shape (1, nums_data)
x = np.fromfile(LOGISITIC_X_FILE, dtype=float, sep=' ')
y = np.fromfile(LOGISITIC_Y_FILE, dtype=float, sep=' ')
m = len(y)
x = np.transpose(np.reshape(x, (int(m), 2)))
x = np.row_stack((np.ones([1, int(m)]), x))
y = y.T


# 牛顿法求最小经验误差
# J: J(theta)
# G: 梯度
# H: Hessian 矩阵
# H_inv: Hessian 矩阵的逆
# f_1: sigmod 一次导
# f_2: sigmod 二次导
# theta: 待优化变量
theta = np.zeros([3, 1])
J_old = 0
threehold = 0.0000000001
while True:
    z = np.dot(theta.T, x) * y
    e_z = np.exp(-z)
    J = 1/m * np.dot(np.log(1+e_z), np.ones([int(m), 1]))
    if np.abs(J-J_old) < threehold:
        break
    J_old = J
    f_1 = - e_z / (1+e_z)
    f_2 = e_z / np.square(1+e_z)
    H = 1/m * np.dot(f_2*x, x.T)
    G = 1/m * np.dot((f_1 * y * x), np.ones([int(m), 1]))
    H_inv = np.linalg.inv(H)
    theta = theta - np.dot(H_inv, G)
    print(J)
print(theta)


# 画图
axes = plt.subplot(111)
# 画出分类边界
x1 = np.linspace(np.min(x[1, :]), np.max(x[1, :]), 200)
x2 = - (x1 * theta[1] + theta[0]) / theta[2]
axes.plot(x1, x2, 'b', lw=1)
# 画出两类点
type1 = axes.scatter(x[1, :][y == 1], x[2, :][y == 1], s=40, c='red')
type2 = axes.scatter(x[1, :][y == -1], x[2, :][y == -1], s=40, c='green')
# 画出坐标轴标签
plt.xlabel('x1')
plt.ylabel('x2')  
plt.show()
