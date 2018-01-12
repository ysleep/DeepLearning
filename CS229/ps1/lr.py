import numpy as np
import matplotlib
import matplotlib.pyplot as plt

LOGISITIC_X_FILE = "data/logistic_x.txt"
LOGISITIC_Y_FILE = "data/logistic_y.txt"

x = np.fromfile(LOGISITIC_X_FILE, dtype=float, sep=' ')
y = np.fromfile(LOGISITIC_Y_FILE, dtype=float, sep=' ')
m = len(y)
x = np.transpose(np.reshape(x, (int(m), 2)))
y = y.T

theta = np.zeros([2, 1])
f_old = 0
threehold = 0.1
while True:
    z = np.dot(theta.T, x) * y
    e_z = np.exp(-z)
    f = 1/m * np.dot(np.log(1+e_z), np.ones([int(m), 1]))
    if np.abs(f-f_old) < threehold:
        break
    f_old = f
    f_1 = e_z / (1+e_z)
    f_2 = e_z / np.square(1+e_z)
    H = 1/m * np.dot(f_2*x, x.T)
    G = 1/m * np.dot((f_1 * y * x), np.ones([int(m), 1]))
    H_inv = np.linalg.inv(H)
    theta = theta - np.dot(H_inv, G)
    print(theta)

print(np.dot(H, H_inv))
print(f)


'''
axes = plt.subplot(111)
type1 = axes.scatter(x_pos[:, 0], x_pos[:, 1], s=40, c='red' )
type2 = axes.scatter(x_neg[:, 0], x_neg[:, 1], s=40, c='green')
plt.xlabel('x1')
plt.ylabel('x2')  
plt.show()
'''