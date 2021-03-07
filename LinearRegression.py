# coding=utf-8
import numpy as np


def standRegres(xArr, yArr):
    """
    xArr - 样本特征，是 (m,n) 矩阵，每行的第一个值既 X0 固定为 1
    yArr - 样本标签，是 (1,m)
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T

    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        # 如果行列式等于 0，意味着没有逆矩阵
        return None

    # 也可以用 ws = np.linalg.solve(xTx, xMat.T*yMat)
    w = xTx.I * xMat.T * yMat

    # ws 是 (n,1) 矩阵
    return w


def lwlr(testPoint, xArr, yArr, k=1.0):
    """
    testPoint - 待预测的点 (1,n)
    xArr - 样本特征 (m,n)，每个样本的第一个值既 X0 固定为 1
    yArr - 样本标签 (1,m)
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T

    m = np.shape(xMat)[0]

    # eye 是单位矩阵，对角线是 1，其余是 0
    weights = np.mat(np.eye(m))

    # 遍历所有数据
    for j in range(m):
        # 计算权重
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))

    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        # 如果行列式等于 0，意味着没有逆矩阵
        return

    # 得出回归系数 (n,1)
    w = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * w


def ridgeRegres(xMat, yMat, lam=0.2):
    """
    xMat - 样本特征 (m,n)，每个样本的第一个值既 X0 固定为 1
    yMat - 样本标签 (1,m)
    """
    xTx = xMat.T * xMat

    # 加上 r*I 使得矩阵可逆
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        return

    w = denom.I * (xMat.T * yMat)
    return w


def ridgeTest(xArr, yArr):
    """
    xArr - 样本特征 (m,n)，每个样本的第一个值既 X0 固定为 1
    yArr - 样本标签 (1,m)
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T

    # Y 数据标准化，减去均值
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean

    # X 数据标准化，减去均值，除以方差
    xMeans = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMeans) / xVar

    numTestPts = 30
    # 初始化回归系数矩阵，每行是一次迭代产生的回归系数
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in range(numTestPts):
        # 迭代，尝试不同的 r 参数
        ws = ridgeRegres(xMat, yMat, np.exp(i - 10))
        wMat[i, :] = ws.T

    # 返回所有回归系数，为了定量地找到最佳参数值，还需要进行交叉验证
    # 一般讲，r 很小时就和普通回归系数一样，r 很大时回归系数趋向于 0
    return wMat


def stageWise(xArr, yArr, step=0.01, numIt=100):
    """
    xArr - 样本特征 (m,n)，每个样本的第一个值既 X0 固定为 1
    yArr - 样本标签 (1,m)
    step - 步长
    numIt - 迭代次数
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T

    # Y 数据标准化，减去均值
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean

    # X 数据标准化，减去均值，除以方差
    xMeans = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMeans) / xVar

    m, n = np.shape(xMat)

    # 初始化回归系数矩阵，每行是一次迭代产生的回归系数
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n, 1))
    wsMax = ws.copy()

    # 迭代
    for i in range(numIt):
        lowestError = np.inf

        # 每个系数
        for j in range(n):
            # 每个方向
            for sign in [-1, 1]:
                wsTest = ws.copy()

                # 在上一次迭代产生的系数向量的基础上，按指定的步长、指定的方向，调整指定的系数
                wsTest[j] += step * sign

                # 预测结果
                yTest = xMat * wsTest

                # 计算方差
                rssE = ((yMat.A - yTest.A) ** 2).sum()

                # 效果更好则保存该系数
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest

        # 得到本次迭代的最佳系数
        ws = wsMax.copy()

        # 保存该最佳系数
        returnMat[i, :] = ws.T

    # 返回结果
    return returnMat
