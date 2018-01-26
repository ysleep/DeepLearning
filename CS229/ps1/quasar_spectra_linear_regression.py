import numpy as np
import matplotlib.pyplot as plt

# 载入数据
QUASAR_TRAIN_FILE = "data/quasar_train.csv"
QUASAR_TEST_FILE = "data/quasar_test.csv"

# 常量
# 画图用的数据点个数
NUM_X_POINT = 1000

# 数据预处理
# header.shape (2, nums_wavelength)
# train.shape (nums_train_example, nums_wavelength)
# train.shape (nums_test_example, nums_wavelength)
train = np.genfromtxt(QUASAR_TRAIN_FILE, delimiter=',')
test = np.genfromtxt(QUASAR_TEST_FILE, delimiter=',')
header = train[0, :]
header = np.row_stack((np.ones([1, header.shape[0]]), header))
train = train[1:, :]
test = test[1:, :]

# (b) i 计算
# 无约束二次规划求极值
# 在计算是使用matrix
x = np.mat(header)
y = np.mat(train[0, :])
theta = (x * x.T).I * x * y.T
print(theta)

# (b) i 画图
# 画图时需要把theta从矩阵转为array
array_theta = np.array(theta)
# 生成一个新的画板
plt.figure()
figure_without_weight = plt.subplot(111)
# 画出回归曲线
x1 = np.linspace(np.min(header[1, :]), np.max(header[1, :]), NUM_X_POINT)
x2 = array_theta[0] + x1 * array_theta[1]
figure_without_weight.plot(x1, x2, 'm', lw=1)
# 画出example
type1 = figure_without_weight.scatter(header[1, :], train[0, :], s=5, c='black', marker='x')
# 画出坐标轴标签
plt.xlabel('x1')
plt.ylabel('x2')


# (b) ii 计算
# 不同的带宽及其颜色
list_TAU = [1, 5, 10, 100, 1000]
list_COLOR = ['c', 'g', 'y', 'r', 'm']

# 获得横坐标，因为权重和横坐标的值有关
# 这里需要注意的是，对于numpy.array，在一维向量的情况下，如果需要一个列向量，那么需要额外指定shape
x1 = np.linspace(np.min(header[1, :]), np.max(header[1, :]), NUM_X_POINT)
tile_x1 = np.tile(x1, (header.shape[1], 1))
column_1_header = header[1, :]
column_1_header.shape = (header.shape[1], 1)
# 存储不同TAU下的x2值
list_x2 = []
for TAU in list_TAU:
    x2 = np.zeros(NUM_X_POINT)
    # 计算权重，这里的权重是以列向量存储的，计算的时候需要reshape成对角阵
    diag_W = np.exp(-(tile_x1 - column_1_header) ** 2 / (2*TAU**2))
    for i in range(NUM_X_POINT):
        W = np.mat(np.diag(diag_W[:, i]))
        tmp_theta = (x * W * x.T).I * x * W * y.T
        x2[i] = tmp_theta[0] + x1[i] * tmp_theta[1]
    list_x2.append(x2)

# (b) ii 计算
plt.figure()
figure_with_weight = plt.subplot(111)
# 根据不同的TAU画出回归曲线并标记标签
list_legend = []
for index in range(len(list_TAU)):
    tmp_handle, = figure_with_weight.plot(x1, list_x2[index], list_COLOR[index], lw=1, label=('TAU='+str(list_TAU[index])))
    list_legend.append(tmp_handle)
plt.legend(handles=list_legend)
# 画出example
type1 = figure_with_weight.scatter(header[1, :], train[0, :], s=5, c='black', marker='x')
# 画出坐标轴标签
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()


