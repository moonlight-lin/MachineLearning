# coding=utf-8
import numpy as np


def selectJrand(i, n):
    """
    在 (0, n) 之间随机选择一个不等于 i 的 值
    """
    j = i
    while j == i:
        j = int(np.random.uniform(0, n))
    return j


def clipAlpha(aj, H, L):
    """
    限制 aj 的值在 L 和 H 之间
    """
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    """
    dataMatIn   - 样本数据集，（n, m） 维数组，n 是样本数，m 是特征数
    classLabels - 标签，(1, n) 维数组，取值为 1 或 -1
    C           - 使 C > a > 0，以处理不 100% 线性可分的情况，加上 C 的限制后不会一直增加 a 值，同时允许部分数据错分
    toler       - 能容忍的误差范围
    maxIter     - 迭代次数
    """

    # 样本转为 numpy 矩阵
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()

    # 样本数，特征数
    n, m = np.shape(dataMatrix)

    # 初始化 a 和 b 为 0
    b = 0
    alphas = np.mat(np.zeros((n, 1)))

    # 如果连续 maxIter 迭代 a 值都没改变就跳出
    # 如果 a 值有变化要重新计算迭代次数
    iter = 0
    while iter < maxIter:
        alphaPairsChanged = 0

        # 遍历每个样本
        for i in range(n):
            # f = WX + b
            #   = (a . y)^T * (X * x^T) + b
            fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b

            # 计算误差
            Ei = fXi - float(labelMat[i])

            # 误差偏大，看能不能优化 a 值
            if (labelMat[i] * Ei < -toler and alphas[i] < C) or (labelMat[i] * Ei > toler and alphas[i] > 0):
                # 随机选择另一个样本
                j = selectJrand(i, n)

                # 计算误差
                fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])

                # 保存对应的 a 值
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                # 计算优化后的 a2 的取值范围
                if labelMat[i] != labelMat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])

                if L == H:
                    # 无法优化
                    print "L == H"
                    continue

                # a2-new = a2-old - y2(E1-E2)/(2 * x1 * x2^T - x1 * x1^T - x2 * x2^T)
                #        = a2-old - y2(E1-E2)/eta
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T \
                    - dataMatrix[i, :] * dataMatrix[i, :].T \
                    - dataMatrix[j, :] * dataMatrix[j, :].T

                if eta >= 0:
                    # 无法优化
                    print "eta>=0"
                    continue

                # 优化 aj
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta

                # 限制 aj
                alphas[j] = clipAlpha(alphas[j], H, L)

                if abs(alphas[j] - alphaJold) < 0.00001:
                    # 没优化
                    print "j not moving enough"
                    continue

                # 优化 ai
                # a1-new = a1-old + y1*y2*(a2-old - a2-new)
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])

                # 计算 b 值
                b1 = b - Ei \
                     - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T \
                     - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T

                b2 = b - Ej \
                     - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T \
                     - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T

                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0

                # 优化次数
                alphaPairsChanged += 1

                print "iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged)

        if alphaPairsChanged == 0:
            # 到这里说明遍历了整个数据集都没能优化
            # 但由于第二个样本是随机的，所以可以继续迭代，可能还是可以优化
            # 其实这里的代码还可以优化，如果每次的第一个样本就决定优化不了，那到这里就没必要继续迭代了
            # 而且这里不应该完全依赖迭代次数达成，应该有其他条件比如误差小到一定程度就可以停止计算
            iter += 1
        else:
            # 有优化，重新计算迭代次数
            iter = 0

        print "iteration number: %d" % iter

    return b, alphas


def calcWs(alphas, dataArr, classLabels):
    """
    计算 W 系数
    """
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(X)
    w = np.zeros((n, 1))

    for i in range(m):
        w += np.multiply(alphas[i]*labelMat[i], X[i, :].T)

    return w


def kernel(X, omiga):
    """
    用高斯核函数对样本数据 X 转换

    X 是二维矩阵

    返回矩阵 K

    K(i, j) = exp(-(||Xi - Xj||**2)/(2 * (omiga**2)))
    """

    m, n = np.shape(X)

    K = np.mat(np.zeros((m, m)))

    for i in range(m):
        for j in range(m):
            # 计算 ||Xi - Xj||**2
            deltaRow = X[j, :] - X[i, :]
            d = deltaRow * deltaRow.T

            K[i, j] = np.exp(d / (-2 * omiga ** 2))

    return K

