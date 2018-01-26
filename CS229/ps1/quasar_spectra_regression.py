import numpy as np
import matplotlib.pyplot as plt

# 载入数据
QUASAR_TRAIN_FILE = "data/quasar_train.csv"
QUASAR_TEST_FILE = "data/quasar_test.csv"

# 常量
TAU = 5                 # 带宽
K = 3                   # K近邻个数
RIGHT_LAMBDA = 1300     # Lyman-alpha 线右边, 不被氢原子吸收的波长最小值
LEFT_LAMBDA = 1200      # Lyman-alpha 线左边, 被氢原子吸收的波长最大值


# 通过局部加权回归来平滑数据
# ori_header: 带常数项（偏置项）的特征, 在这里特征为波长. shape: (2, nums_wavelength)
# ori_data: 数据集, 在这里数据集为光通量(流明). shape: (nums_train_example, nums_wavelength)
# tau: 带宽
def make_data_smooth(ori_header, ori_data, tau):
    # 计算权重矩阵W, 由于W是对角阵，所以这里W以列向量存储
    x1 = ori_header[1, :]
    mat_ori_header = np.mat(ori_header)
    tile_x1 = np.tile(x1, (ori_header.shape[1], 1))
    column_1_header = ori_header[1, :]
    column_1_header.shape = (ori_header.shape[1], 1)
    diag_W = np.exp(-(tile_x1 - column_1_header) ** 2 / (2 * tau ** 2))

    # 计算平滑后的矩阵, 注意到, 权重矩阵W是和x(波长)的值相关的, 所以平滑后的数据是按列计算的,
    # 一次计算某个波长对应的所有数据集的平滑项
    # y的每一行是一次测量结果, 每一列是一个波长对应的所有测试结果
    y = np.mat(ori_data)
    smooth_data = np.zeros(ori_data.shape)
    x = np.mat(ori_header)
    for j in range(ori_data.shape[1]):
        W = np.mat(np.diag(diag_W[:, j]))
        tmp_theta = (x * W * x.T).I * x * W * y.T
        smooth_data[:, j] = (tmp_theta.T * mat_ori_header[:, j:j+1]).T
    '''
    # 老版本的平滑计算, 这个版本参照quasar_spectra_linear_regression.py文件, 对每一个数据单独计算,
    # 计算量很大, 训练数据需要跑30s, 这里也能看到矩阵化带来的好处
    for i in range(ori_data.shape[0]):
        y = np.mat(ori_data[i, :])
        for j in range(ori_data.shape[1]):
            W = np.mat(np.diag(diag_W[:, j]))
            tmp_theta = (x * W * x.T).I * x * W * y.T
            smooth_data[i][j] = tmp_theta[0] + x1[j] * tmp_theta[1]
    '''
    return smooth_data


# 预测f_left
def predict_left(f_i, f_all, k, header):
    # 计算f_i_right到f_all_right的均方距离
    un_sorted_d = np.sum((f_i[header[1, :] >= RIGHT_LAMBDA] - f_all[:, header[1, :] >= RIGHT_LAMBDA]) ** 2, axis=1)
    # 对距离排序
    sorted_d = np.sort(un_sorted_d)
    # 获得参数h, 即最大距离
    h = sorted_d[-1]
    # 获得最近的K个距离
    d_neighb_k = un_sorted_d[un_sorted_d <= sorted_d[k-1]]
    # 获得权重, 计算公式为max(1-(d/h), 0), 可以看到, 距离越近, 权重越大
    # 这里可以尝试更多的权重公式，比如平方反比之类的
    rate_d_nieghb_k_vs_h = np.max(np.row_stack((np.zeros(d_neighb_k.shape[0]), 1-d_neighb_k / h)), axis=0)
    # 将变量reshape为2维数组
    rate_d_nieghb_k_vs_h.shape = (rate_d_nieghb_k_vs_h.shape[0], 1)
    # 取最近的K个距离对应的f_all_left
    # 这里要注意的是array[array1, array2], 如果array1和array2的长度，不能直接取, 需要写成array[array1, :][:, array2]
    f_left_neighb_k = f_all[un_sorted_d <= sorted_d[k-1], :][:, header[1, :] <= LEFT_LAMBDA]

    # 计算加权平均
    return np.sum((f_left_neighb_k * rate_d_nieghb_k_vs_h), axis=0) / sum(rate_d_nieghb_k_vs_h)


def main():
    # 数据预处理
    # header.shape (2, nums_wavelength)
    # train.shape (nums_train_example, nums_wavelength)
    # train.shape (nums_test_example, nums_wavelength)
    train = np.genfromtxt(QUASAR_TRAIN_FILE, delimiter=',')
    test = np.genfromtxt(QUASAR_TEST_FILE, delimiter=',')
    header = train[0, :]
    header = np.row_stack((np.ones([1, header.shape[0]]), header))
    ori_train = train[1:, :]
    ori_test = test[1:, :]

    # 平滑训练集和测试集
    smooth_train = make_data_smooth(header, ori_train, TAU)
    smooth_test = make_data_smooth(header, ori_test, TAU)

    # 预测
    f_observed = smooth_train[0, :]
    f_left_predict = predict_left(f_observed, smooth_train, K, header)

    # 画图
    plt.figure()
    figure_predict = plt.subplot(111)
    # 画出观察曲线(原始曲线)
    x1 = header[1, :]
    x2 = f_observed
    figure_predict.plot(x1, x2, 'm', lw=1)

    # 画出预测曲线
    x1_predict = header[1, header[1, :] <= LEFT_LAMBDA]
    x2_predict = f_left_predict
    figure_predict.plot(x1_predict, x2_predict, 'b', lw=1)
    plt.show()


if __name__ == "__main__":
    main()
