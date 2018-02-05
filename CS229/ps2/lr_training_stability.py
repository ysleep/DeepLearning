import numpy as np
import CS229.ps2.lr_debug as lr_debug
import matplotlib.pyplot as plt


def plot(x, y):
    plt.figure()
    # 画图
    axes = plt.subplot(111)
    # 画出两类点
    type1 = axes.scatter(x[:, 1][y == 1], x[:, 2][y == 1], s=40, c='red')
    type2 = axes.scatter(x[:, 1][y == -1], x[:, 2][y == -1], s=40, c='green')
    # 画出坐标轴标签
    plt.xlabel('x1')
    plt.ylabel('x2')


def main():
    Xa, Ya = lr_debug.load_data('data/data_a.txt')
    Xb, Yb = lr_debug.load_data('data/data_b.txt')

    plot(Xa, Ya)
    plot(Xb, Yb)


    plt.show()


if __name__ == "__main__":
    main()
