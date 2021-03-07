# coding=utf-8

"""
以留言分类为例子
一个一维数组代表一条留言，每个元素代表一个单词
判断留言是否不当内容
"""

import numpy as np


# 从样本数据集创建词典
def createVocabList(dataSet):
    """
    dataSet - 样本数据集，二维数组，每一行是一条留言数据，每个元素是留言的一个单词
    """
    vocabSet = set()

    for document in dataSet:
        # 将样本集中出现过的单词存到一个集合并去重
        vocabSet = vocabSet | set(document)

    # 将结果转成列表，返回作为词典
    return list(vocabSet)


# 将一条留言转换成一个特征向量
def convertDoc2Vec(vocabList, inputDoc):
    """
    vocabList - 词典
    inputDoc  - 留言
    """

    # 创建一个长度和词典一样的特征向量
    # 特征向量的每个值取 1 或 0，代表词典中的这个词在留言中存在，0 代表不存在，全部初始化为 0
    returnVec = [0]*len(vocabList)

    # 遍历留言的每个单词
    for word in inputDoc:
        if word in vocabList:
            # 单词存在词典中，特征向量的对应位置设为 1
            returnVec[vocabList.index(word)] = 1
        else:
            # 不存在与词典中，忽略
            print "the word: %s is not in my Vocabulary!" % word

    return returnVec


# 朴素贝叶斯训练
def trainNB(trainMatrix, trainCategory):
    """
    trainMatrix   - 用于训练的样本，二维 numpy 数组，行数代表留言数，列数是词典收录的词量
                    如果词典的第三个词在第二个样本出现过，则 trainMatrix[1][2] = 1，否则 trainMatrix[1][2] = 0
    trainCategory - 样本数据的分类，一维 numpy 数组，1 代表不当言论，0 代表普通言论
    """

    # 样本数
    numTrainDocs = len(trainMatrix)

    # 词典大小，即特征向量的长度
    numWords = len(trainMatrix[0])

    # 不当留言在所有留言中的概率，即 P(C1)
    # sum 对一维数组求和，由于只有 1 和 0 两种值，求和结果就是不当言论的总数
    p1 = np.sum(trainCategory)/float(numTrainDocs)

    # 每个单词出现在分类 0 的次数，和出现在分类 1 的次数，以及分类 0 的总词数，分类 1 的总词数
    # 目的是为了求出 P(Xn|C0)，P(Xn|C1)
    # 初始化为 1 和 2 是为了防止出现统计为 0 的情况，不然 P(Xn|C0)，P(Xn|C1) 会导致最终结果为 0
    # 选 1 和 2 这样如果完全统计不到，那就默认 50% 的概率
    p0WordNum = np.ones(numWords)
    p1WordNum = np.ones(numWords)
    p0TotalNum = 2.0
    p1TotalNum = 2.0

    # 遍历每个留言
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            # 统计每个单词在类别 1 (既不当言论) 中出现的次数
            p1WordNum += trainMatrix[i]
            # 统计类别 1 的总单词数
            p1TotalNum += np.sum(trainMatrix[i])
        else:
            # 统计类别 0
            p0WordNum += trainMatrix[i]
            p0TotalNum += np.sum(trainMatrix[i])

    # 统计每个单词在类别 0 和 类别 1 中出现的概率，即 P(Xn|C0) 和 P(Xn|C1)
    # 取对数是为了防止下溢出，概率太小有可能被当成 0 处理
    p1Vec = np.log(p1WordNum/p1TotalNum)
    p0Vec = np.log(p0WordNum/p0TotalNum)

    p0 = np.log(1.0 - p1)
    p1 = np.log(p1)

    # 返回 log(P(Xn|C0))，log(P(Xn|C1))，log(P(C0))，log(P(C1))
    # 对于新留言 X，只要比较 log(P(X|C0)*P(C0)) 和 log(P(X|C1)*P(C1))
    # 相当于比较 log(P(X|C0)) + log(P(C0)) 和 log(P(X|C1)) + log(P(C1))
    # 就可以判断属于哪种分类的概率大
    return p0Vec, p1Vec, p0, p1


# 朴素贝叶斯分类
def classifyNB(featureVec, p0Vec, p1Vec, p0, p1):
    """
    featureVec - 要分类的留言，以词典的特征向量表示
    p0Vec - log(P(X|C0))
    p1Vec - log(P(X|C1))
    p0 - log(P(C0))
    p1 - log(P(C1))
    """

    # 比较 log(P(X|C0)) + log(P(C0)) 和 log(P(X|C1)) + log(P(C1))
    # 其中
    #      log(P(X|C0)) = log(P(X1|C0)) + log(P(X2|C0)) + ... + log(P(Xn|C0))
    #      log(P(X|C1)) = log(P(X1|C1)) + log(P(X2|C1)) + ... + log(P(Xn|C1))
    p1 = sum(featureVec * p1Vec) + p1
    p0 = sum(featureVec * p0Vec) + p0

    # 哪个值大，就认为文档属于哪个分类，如果要给出具体概率还要进一步计算
    if p1 > p0:
        return 1
    else:
        return 0


# 测试
def testNB(dataSet, labelSet, testData):
    """
    dataSet  - 样本数据集，二维数组，每一行是一条留言数据，每个元素是留言的一个单词
    labelSet - 样本数据集的分类标签，一维数组
    testData - 要分类的数据，一位数组，每个元素是一个单词
    """
    # 创建词典
    vocabList = createVocabList(dataSet)

    # 转为词典特征向量数组
    trainMat = []
    for doc in dataSet:
        # 将每个文档转换为词典特征向量
        trainMat.append(convertDoc2Vec(vocabList, doc))

    # 训练
    p0Vec, p1Vec, p0, p1 = trainNB(np.array(trainMat), np.array(labelSet))

    # 将要预测的数据转为词典特征向量
    testVec = np.array(convertDoc2Vec(vocabList, testData))

    # 预测
    print classifyNB(testVec, p0Vec, p1Vec, p0, p1)
