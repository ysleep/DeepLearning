import numpy as np

def sigmoid(z):
    return 1 / (1+np.exp(-z))

def initialization(dim,mode="zero"):
    w = np.zeros((dim,1))
    b = 0
    if(mode=="random"):
        w = np.random.randn(dim,1)
        b = np.random.randn(1,1)[0]
    return w,b

def train(X_train,Y_train,iterationTimes,learningRate):
    #w, b = initialization(X_train.shape[0], "zero")
    w,b = np.array([[1],[2]]), 2
    n=0
    for i in range(iterationTimes):
        Z = np.dot(w.T,X_train) + b
        A = sigmoid(Z)
        L = -1/Y_train.shape[1] * np.sum(Y_train*np.log(A)+(1-Y_train)*np.log(1-A),axis=1)
        dw = 1/Y_train.shape[1] * np.dot(X_train,(A-Y_train).T)
        db = 1/Y_train.shape[1] * np.sum((A-Y_train),axis=1)
        w = w-learningRate*dw
        b = b-learningRate*db
        iterationTimes -= 1
        if n%100==0:
            print(L)

    return w,b


def main():
    X,Y = np.array([[1,2],[3,4]]), np.array([[1,0]])

    iterationTimes = 2000

    w,b = train(X,Y,100,0.009)

    print(w)
    print(b)

    return

if __name__ == '__main__':
    main()