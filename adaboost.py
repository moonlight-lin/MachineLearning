# coding=utf-8
import numpy as np


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    """
    单层决策树 (decision stump，也称决策树桩)
        它仅基于单个特征来做决策，由于只有一次分裂过程，实际上就是一个树桩
        单层决策树的分类能力比较弱，是一种弱分类器，通过 adaboost 使用多个单层决策树可以构建强分类器

    dataMatrix - 要分类的数据集 (n, m) 矩阵
    dimen -      用于分类的特征
    threshVal -  判断分类的阀值
    threshIneq - 操作符 ('lt', 'gt') 决定是特征值大于阀值返回分类 -1，还是小于阀值返回分类 -1
    """

    # 初始化分类矩阵，默认为分类 1
    retArray = np.ones((np.shape(dataMatrix)[0], 1))

    if threshIneq == 'lt':
        # 当 dataMatrix[x, dimen] <= threshVal 时，将 retArray[x] 改为 -1
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0

    return retArray


def buildStump(dataArr, classLabels, D):
    """
    按照样本权值，寻找最佳的单层决策树，即寻找最佳的分类特征和分类阀值，以及操作符

    dataArr -     样本数据
    classLabels - 标签数据
    D -           样本权值
    """

    # 初始化矩阵并获取矩阵大小
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    n, m = np.shape(dataMatrix)

    # 阀值数
    # 将特征值从最大值到最小值之间，取 10 个间隔分出 11 个阀值，在这些阀值中选取最佳值
    numSteps = 10.0

    # 用于存储最佳决策树的配置，包括（特征，阀值，操作符）
    bestStump = {}

    # 用于保存最佳决策树的分类结果
    bestClasEst = np.mat(np.zeros((n, 1)))

    # 用于保存最佳决策树的错误率
    minError = np.inf

    # 遍历每一个特征
    for i in range(m):
        # 取该特征的最大最小值以及步长
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin)/numSteps

        # 遍历所有阀值
        for j in range(0, int(numSteps) + 1):

            # 遍历操作符
            for inequal in ['lt', 'gt']:
                # 取阀值
                threshVal = (rangeMin + float(j) * stepSize)

                # 以 (特征，阀值，操作符) 作为决策树，对所有数据分类
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)

                # 评估分类结果，正确分类为 1，错误分类为 0
                errArr = np.mat(np.ones((n, 1)))
                errArr[predictedVals == labelMat] = 0

                # 计算错误率, D 的初始值是 1/(样本总数)
                weightedError = D.T*errArr
                if weightedError < minError:
                    # 找到更好的决策树，保存结果
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal

    # 返回最佳决策树配置(特征，阀值，操作符), 最佳决策树的错误率, 最佳决策树的分类结果
    return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    """
    基于单层决策树的 adaboost 训练

    dataArr -     样本数据
    classLabels - 样本标签
    numIt -       最大迭代次数
    """

    # 用于保存决策树列表
    # 每次迭代会产生一个决策树, 直到达到最大迭代次数, 或是最终错误率为 0
    weakClassArr = []

    # 样本数
    n = np.shape(dataArr)[0]

    # 初始化样本权值 D，每个数据的权重为 1/(样本数)
    D = np.mat(np.ones((n, 1))/n)

    # 保存累加分类结果
    aggClassEst = np.mat(np.zeros((n, 1)))

    for i in range(numIt):
        # 按样本和权值寻找最佳决策树
        # 返回决策树配置(特征，阀值，操作符), 错误率, 分类结果
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)

        # 计算决策树权值 alpha = 0.5 * ln((1-err)/err)
        # 1e-16 是防止 err 为 0 的情况, ln(1/1e-16) 的结果为 36.8
        # 这里没处理 err > 0.5 导致 alpha < 0 的情况, 照理说也不应该出现这种情况
        alpha = float(0.5 * np.log((1.0 - error)/max(error, 1e-16)))

        # 将决策树权值加入到决策树配置
        bestStump['alpha'] = alpha

        # 将决策树配置加入决策树列表
        weakClassArr.append(bestStump)

        # 计算新的样本权值
        # D(i_new) = (D(i_old) * exp(-alpha * yi * f(xi))) / SUM_j_1_n (D(j_old) * exp(-alpha * yj * f(xj)))
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D/D.sum()

        # 该决策树的分类结果, 按权值算入累加分类结果
        aggClassEst += alpha*classEst

        # 计算累加分类结果的错误率, 如果错误率为 0 则退出迭代
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((n, 1)))
        errorRate = aggErrors.sum()/n
        if errorRate == 0.0:
            break

    # 返回决策树配置列表, 累加分类结果
    return weakClassArr, aggClassEst


def adaClassify(datToClass, classifierArr):
    """
    使用决策树列表进行分类

    weakClassArr -  要分类的数据
    classifierArr - 决策树配置列表
    """

    dataMatrix = np.mat(datToClass)
    n = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((n, 1)))

    # 遍历决策树
    for i in range(len(classifierArr)):
        # 分类
        classEst = stumpClassify(dataMatrix,
                                 classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])

        # 按权值累加分类结果
        aggClassEst += classifierArr[i]['alpha']*classEst

    # sign 函数：大于 0 返回 1，小于 0 返回 -1，等于 0 返回 0
    return np.sign(aggClassEst)









