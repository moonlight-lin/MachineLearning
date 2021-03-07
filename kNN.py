# coding=utf-8
import numpy as np
import operator as op


# KNN 分类算法
def classify(inX, dataSet, labels, k):
    """
    inX     - 要预测的特征值矩阵，是一个 (1, n) 的 numpy 矩阵，n 是特征的数量
    dataSet - 样本的特征值矩阵，是一个 (r, n) 的 numpy 矩阵，r 是样本数量，n 是特征的数量
    labels  - 样本的标签，是一个长度为 r 的列表
    k       - 取最近的 k 个样本
    """

    # 取样本的数量
    dataSetSize = dataSet.shape[0]

    # tile 将 inX 按 (dataSetSize, 1) 平铺，结果是 (r, n) 矩阵，矩阵的每列都是 inX，然后和样本矩阵相减
    # 得到 diffMat 是 inX 的每个特征值和所有样本的差
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet

    # sum(axis=1) 指按第二个下标求和，即
    #     sqDiffMat[0][0] + sqDiffMat[0][1] + sqDiffMat[0][2] + ...... + sqDiffMat[0][n-1]
    #     sqDiffMat[1][0] + sqDiffMat[1][1] + sqDiffMat[0][2] + ...... + sqDiffMat[1][n-1]
    # 得到一个 (r, 1) 矩阵
    # 再开根，得到 inX 和每个样本的距离
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5

    # 对距离按从小到大排序，并返回索引值，即 distances[sortedDistIndex[0]] 是 distances 的最小值
    sortedDistIndex = distances.argsort()

    classCount = {}
    # 取距离最近的 k 个样本
    for i in range(k):
        # 取样本的标签值
        votedLabel = labels[sortedDistIndex[i]]
        # 统计每个标签值的数量
        classCount[votedLabel] = classCount.get(votedLabel, 0) + 1

    # classCount.iteritems 得到为迭代器
    # op.itemgetter(1) 取每个数据的第二个域的值，即 label 的次数
    # 从大到小排序
    sortedClassCount = sorted(classCount.iteritems(), key=op.itemgetter(1), reverse=True)

    # 返回距离最近的 k 个样本中出现次数最多的标签值
    return sortedClassCount[0][0]


# 从文件中取样本数据
def readDataFromFile(filename, featureNumber):
    """
    文件的每一行是一条数据，最后一列是标签，其余列是特征值，共 fieldNumber 个特征
    """
    numberOfLines = 0
    with open(filename) as f:
        # 统计行数，数据量太大时 len(f.readlines()) 可能无法工作，所以采用循环的方式
        while True:
            tempBuffer = f.read(8192 * 1024)
            if not tempBuffer:
                numberOfLines = numberOfLines + 1
                break
            numberOfLines += tempBuffer.count('\n')

    # 特征值矩阵
    featureMat = np.zeros((numberOfLines, featureNumber))
    # 标签列表
    classLabelVector = []

    with open(filename) as f:
        index = 0
        while True:
            line = f.readline()
            if line:
                line = line.strip()
                fields = line.split('\t')
                featureMat[index, :] = fields[0:featureNumber]
                classLabelVector.append(int(fields[featureNumber]))
                index += 1
            else:
                break

    return featureMat, classLabelVector


# 归一化数据
def normalize(dataSet):
    # dataSet.min(0) 按第一个下标求最小值，即求
    #   min(dataSet[0][0], dataSet[1][0], dataSet[2][0], ..., dataSet[m][0])
    #   min(dataSet[0][1], dataSet[1][1], dataSet[2][1], ..., dataSet[m][1])
    # 结果是求每列，即每个特征值的最大和最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)

    # 最大值和最小值的差
    ranges = maxVals - minVals

    # np.tile(minVals, (m, 1)) 复制 m 个 minVals，方便矩阵的计算
    # 归一化 newValue = (oldValue - min)/(max - min) 将所有数据转换到 (0, 1)
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet/np.tile(ranges, (m, 1))

    return normDataSet


# 测试
def classifyTest():
    # 10% 的数据作测试集，90% 的数据作样本集
    ratio = 0.10

    # 读数据
    featureNumber = 10
    featureMat, classLabelVector = readDataFromFile('kNN.txt', featureNumber)

    # 归一化
    normDataSet = normalize(featureMat)

    # 数据量
    m = normDataSet.shape[0]

    # 测试集大小
    numTest = int(m * ratio)

    errorCount = 0
    for i in range(numTest):
        result = classify(normDataSet[i, :], normDataSet[numTest:m, :], classLabelVector[numTest:], featureNumber)
        if result != classLabelVector[i]:
            errorCount += 1

        print "predict result: %d, real answer is: %d" % (result, classLabelVector[i])

    print "error count: %d, total: %d" % (errorCount, numTest)
    print "the total error rate is: %f" % (errorCount / float(numTest))
