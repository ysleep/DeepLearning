#在这里吐槽一下驼峰式的变量名。。。
import numpy as np


def readMatrix(file):
    fd = open(file, 'r')
    hdr = fd.readline()
    rows, cols = [int(s) for s in fd.readline().strip().split()]
    tokens = fd.readline().strip().split()
    matrix = np.zeros((rows, cols))
    Y = []
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        Y.append(nums[0])
        kv = np.array(nums[1:])
        k = np.cumsum(kv[:-1:2])
        v = kv[1::2]
        matrix[i, k] = v
    return matrix, tokens, np.array(Y)


def nb_train(matrix, category):
    state = {}
    N = matrix.shape[1]
    ###################
    # 根据作业提示，概率phi_k值很低，所以通过取对数把乘法变成加法
    # 计算phi_k_y
    matrix_y_1 = matrix[category == 1, :]
    matrix_y_0 = matrix[category == 0, :]
    sum_y_1 = np.sum(matrix_y_1, axis=(0, 1))
    sum_y_0 = np.sum(matrix_y_0, axis=(0, 1))
    sum_column_y_1 = np.sum(matrix_y_1, axis=0)
    sum_column_y_0 = np.sum(matrix_y_0, axis=0)
    phi_k_y_1 = (sum_column_y_1 + 1) / (sum_y_1 + N)
    phi_k_y_0 = (sum_column_y_0 + 1) / (sum_y_0 + N)
    state["log_phi_k_y_1"] = np.log(phi_k_y_1)
    state["log_phi_k_y_0"] = np.log(phi_k_y_0)

    # 计算phi_y
    phi_y = np.sum(category == 1) / category.shape[0]
    state["log_phi_y_1"] = np.log(phi_y)
    state["log_phi_y_0"] = np.log(1-phi_y)
    ###################
    return state


def nb_test(matrix, state):
    output = np.zeros(matrix.shape[0])
    ###################
    predict_y_1 = np.sum(matrix * state["log_phi_k_y_1"], axis=1) + state["log_phi_y_1"]
    predict_y_0 = np.sum(matrix * state["log_phi_k_y_0"], axis=1) + state["log_phi_y_0"]
    output = predict_y_1 > predict_y_0
    ###################
    return output


def evaluate(output, label):
    error = (output != label).sum() * 1. / len(output)
    print('Error: %1.4f' % error)


def most_indicative_spam_token(state, token_list, k):
    token_indicative = state["log_phi_k_y_1"] - state["log_phi_k_y_0"]
    index_token_max_k = np.argpartition(token_indicative, -k)[-k:]
    print([token_list[index] for index in index_token_max_k])


def different_train_size():
    list_train_file_suffix = [50, 100, 200, 400, 800, 1400]
    array_train_size = np.array(list_train_file_suffix)
    array_test_error = np.zeros(array_train_size.shape)
    matrix_test, list_token, category_test = readMatrix('MATRIX.TEST')

    for i, suffix in enumerate(list_train_file_suffix):
        matrix_train, list_train, category_train = readMatrix('MATRIX.TRAIN.' + str(suffix))
        state = nb_train(matrix_train, category_train)
        output = nb_test(matrix_test, state)
        array_test_error[i] = (output != category_test).sum() * 1. / len(output)

    print(array_test_error)


def main():
    trainMatrix, tokenlist, trainCategory = readMatrix('MATRIX.TRAIN')
    testMatrix, tokenlist, testCategory = readMatrix('MATRIX.TEST')

    state = nb_train(trainMatrix, trainCategory)
    output = nb_test(testMatrix, state)

    # 6(a)
    evaluate(output, testCategory)

    # 6(b)
    most_indicative_spam_token(state, tokenlist, 5)

    # 6(c)
    different_train_size()
    return

if __name__ == '__main__':
    main()
