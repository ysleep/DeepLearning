import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import pylab


def sigmoid(z):
    return 1 / (1+np.exp(-z))


def initialization(dim, mode="zero"):
    w = np.zeros((dim, 1))
    b = 0
    if mode == "random":
        w = np.random.randn(dim, 1)
        b = np.random.randn(1, 1)[0]
    return w, b

def propagate(X, Y, w, b):
    # calculate the propagate processing
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    L = -1/Y.shape[1] * np.sum(Y*np.log(A) + (1-Y)*np.log(1-A), axis=1)
    dw = 1/Y.shape[1] * np.dot(X, (A-Y).T)
    db = 1/Y.shape[1] * np.sum((A-Y), axis=1)

    # validate the parameters
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(L)
    assert (cost.shape == ())

    # return the result
    return dw, db, cost


def optimize(X, Y, w, b, num_iterations, rate_learning, flag_print=False):
    costs = []
    for i in range(num_iterations):
        dw, db, cost = propagate(X, Y, w, b)
        w = w - rate_learning * dw
        b = b - rate_learning * db

        if i%100 == 0:
            costs.append(cost)
            if flag_print:
                print("after the %dth iteration, the cost is %f" % (i, cost))

    return w, b, costs


def predict(w, b, X):
    Y = np.zeros((1, X.shape[1]))
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    for i in range(X.shape[1]):
        Y[0, i] = 1 if A[0, i] >= 0.5 else 0
    return Y


def get_image(image_name):
    image_path = "image/"
    image_height = 66
    image_width = 66
    X = np.zeros((64, 64, 3, 8*8*len(image_name)))
    Y = np.zeros((1, 8*8*len(image_name)))
    for i in range(len(image_name)):
        fname = image_path + image_name[i]
        image = np.array(ndimage.imread(fname, flatten=False))
        for j in range(8):
            for k in range(8):
                x_begin = j*image_width + 2
                x_end = (j+1)*image_width
                y_begin = k*image_height + 2
                y_end = (k+1)*image_height
                tmp_image = image[x_begin:x_end, y_begin:y_end, :]
                X[:, :, :, i*64+j*8+k] = tmp_image
                Y[:, i*64+j*8+k] = 1 if i<len(image_name)/2 else 0

    X = X.reshape(64*64*3, -1)
    X = X / 255
    return X, Y


def main():
    X_train, Y_train = get_image(["cat1.png", "cat2.png", "nocat1.jpg", "nocat2.jpg"])
    X_test, Y_test = get_image(["cat3.png", "nocat3.jpg"])
    #X_test = X_test[:,0:63]
    #Y_test = Y_test[:, 0:63]
    w, b = initialization(X_train.shape[0], "zero")
    w, b, costs = optimize(X_train, Y_train, w, b, 30000, 0.0005, False)
    print(str(costs))
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)
    print("train accuracy is {} %".format(100-np.mean(np.abs(Y_train-Y_prediction_train)*100)))
    print("test accuracy is {} %".format(100-np.mean(np.abs(Y_test-Y_prediction_test)*100)))

    #w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1, 2], [3, 4]]), np.array([[1, 0]])

    #w, b, dw, db, costs = optimize(X, Y, w, b, num_iterations=100, rate_learning=0.009, flag_print=False)

    #print("w = " + str(w))
    #print("b = " + str(b))
    #print("dw = " + str(dw))
    #print("db = " + str(db))

    return


def param():
    w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1, 2], [3, 4]]), np.array([[1, 0]])
    dw, db, w, b, costs = optimize(X, Y, w, b, 100, 0.009, False)

    print("w = " + str(w))
    print("b = " + str(b))
    print("dw = " + str(dw))
    print("db = " + str(db))

    print("predictions = " + str(predict(w, b, X)))


if __name__ == '__main__':
    print("hello world")
    #param()
    main()