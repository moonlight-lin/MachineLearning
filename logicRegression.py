# coding=utf-8
import numpy as np
import random


# 阶跃函数
def sigmoid(z):
    return 1.0/(1+np.exp(-z))


# 梯度下降算法
# 根据样本（X,Y) 算法最佳的 W
def gradDescent(sampleData, classLabels):
    """
    sampleData  - 样本特征，(n,m) 的二维数组，n 是样本数，m 是特征数
                  每行的第一个值 X0 固定为 1，从 X1 开始才是真正的特征值，目的是简化向量的计算
                   y = x1*w1 + ... + xm*wm + b
                     = x1*w1 + ... + xm*wm + x0w0
                     = XW

    classLabels - 样本标签，(1,n) 的一维数组
    """

    # 转为 NumPy 矩阵
    dataMatrix = np.mat(sampleData)

    # 将 (1,n) 转为 (n,1) 方便后面的矩阵计算
    labelMatrix = np.mat(classLabels).transpose()

    # n 个样本，m 个特征值
    n, m = np.shape(dataMatrix)

    # 梯度下降的步长
    alpha = 0.001

    # 最大迭代次数
    maxCycles = 500

    # 初始化 W 为 (m,1) 数组, 默认值为 1
    weights = np.ones((m, 1))

    # 迭代
    for k in range(maxCycles):
        # (n,m) 矩阵乘以 (m,1) 矩阵，得到 (n,1) 矩阵，再通过逻辑回归函数得到样本的 Y
        h = sigmoid(dataMatrix*weights)

        # 两个 (n,1) 矩阵，得到每个样本的误差
        error = (labelMatrix - h)

        # w = w + a*(X^T)*(Y-g(XW))
        #   = w + a*(X^T)*E
        weights = weights + alpha * dataMatrix.transpose() * error

    return weights


# 分类
def classify(data, weights):
    dataMatrix = np.mat(data)
    resultMatrix = sigmoid(dataMatrix * weights) > 0.5

    """
    for result in resultMatrix:
        print(result.item())
    """
    return resultMatrix


# 随机梯度
def stocGradDescent0(sampleData, classLabels):
    dataMatrix = np.mat(sampleData)
    labelMatrix = np.mat(classLabels).transpose()

    n, m = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones((m, 1))

    for i in range(n):
        # 每次迭代只取一个样本，迭代次数为样本个数
        h = sigmoid(dataMatrix[i]*weights)
        error = labelMatrix[i] - h
        weights = weights + alpha * dataMatrix[i].transpose() * error

    return weights


# 改进的随机梯度
def stocGradDescent1(sampleData, classLabels, numIter=150):
    dataMatrix = np.mat(sampleData)
    labelMatrix = np.mat(classLabels).transpose()

    n, m = np.shape(dataMatrix)
    weights = np.ones((m, 1))

    # 自己选择迭代次数
    for j in range(numIter):
        dataIndex = range(n)

        # 每次迭代又迭代了每一个样本
        for i in range(n):
            # 每次迭代都改变步长，0.0001 用于防止出现 0 的情况
            alpha = 4/(1.0+j+i)+0.0001

            # 随机选择一个样本
            randIndex = int(random.uniform(0, len(dataIndex)))

            h = sigmoid(dataMatrix[randIndex]*weights)
            error = labelMatrix[randIndex] - h

            # 计算新的 W
            weights = weights + alpha * dataMatrix[randIndex].transpose() * error

            # 删除该样本下标
            del(dataIndex[randIndex])

    return weights


# 测试
def test(f):
    samples = []
    labels = []
    for i in range(100):
        data = [1]
        for j in range(5):
            feature = random.uniform(0, 100)
            data.append(feature)
        samples.append(data)
        labels.append(random.choice([0, 1]))

    W = f(samples, labels)

    resultMatrix = classify(samples[:10], W)

    print(f.__name__)
    for result in resultMatrix:
        print(result.item())


test(gradDescent)
test(stocGradDescent0)
test(stocGradDescent1)




