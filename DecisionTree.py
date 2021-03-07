# coding=utf-8
import pickle
import operator as op
from math import log


# 计算信息熵（也叫香农熵）
def calcShannonEnt(dataSet):
    """
    dataSet - 二维数组，每一条数据，其最后一个值是标签值，其他值是特征值
    """
    numEntries = len(dataSet)
    labelCount = {}
    for data in dataSet:
        # 最后一个元素是 label，其他元素是特征值
        label = data[-1]
        if label not in labelCount.keys():
            labelCount[label] = 0
        # 统计每种 label 的总数
        labelCount[label] += 1

    shannonEnt = 0.0
    for label in labelCount:
        # 每个 label 所占比例
        prob = float(labelCount[label]) / numEntries
        # 计算熵
        shannonEnt -= prob * log(prob, 2)

    return shannonEnt


# 取子集
def splitDataSet(dataSet, axis, value):
    """
    dataSet - 二维数组，每一条数据，其最后一个值是标签值，其他值是特征值
    axis    - 特征的下标
    value   - 特征的其中一个可能值，按该取子集，并将该特征值从子集中删除
    """
    retDataSet = []
    for data in dataSet:
        if data[axis] == value:
            # 满足条件，将 data[axis] 删除并放入子集
            newData = data[:axis]
            newData.extend(data[axis + 1:])
            retDataSet.append(newData)
    return retDataSet


# 选取最佳特征
def chooseBestFeatureToSplit(dataSet):
    """
    dataSet - 二维数组，每一条数据，其最后一个值是标签值，其他值是特征值
    """

    # 有多少个特征
    numFeatures = len(dataSet[0]) - 1

    # 取未划分时，数据集的熵
    baseEntropy = calcShannonEnt(dataSet)

    # 划分后熵的最大增益
    maxEntropyGain = 0.0
    # 用于划分的特征
    bestFeature = -1

    # 遍历每个特征
    for i in range(numFeatures):
        # 按该特征构建树的节点和子树
        # 即该特征是树的节点，数据集中有相同特征值的组成子集，作为这个特征节点的子树，子集的数据，该特征值被删除
        # 然后计算熵
        # 遍历所有特征构建树，熵最小的那个就是最佳特征

        # 获取数据集在该特征上的所有特征值
        featValueList = [data[i] for data in dataSet]

        # 去掉重复的值
        featValueSet = set(featValueList)

        # 划分后的熵
        newEntropy = 0.0

        # 遍历该特征的每个值
        for featValue in featValueSet:
            # 按该值取子集
            subDataSet = splitDataSet(dataSet, i, featValue)

            # 计算划分后的系统的熵
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)

        # 是否最小的熵（这里有个问题，难道一定存在划分后熵比划分前小的特征？不然岂不是会返回 -1）
        entropyGain = baseEntropy - newEntropy
        if entropyGain > maxEntropyGain:
            maxEntropyGain = entropyGain
            bestFeature = i

    # 返回能得到最小熵的特征的下标
    return bestFeature


# 创建决策树
def createTree(dataSet, featureNames):
    """
    dataSet       - 样本数据集，是一个二维数组，每一条数据，其最后一个值是标签值，其他值是特征值
    featureNames  - 是特征的名字，是一个一维数组
    """

    # 取数据集的所有分类标签
    labelList = [data[-1] for data in dataSet]

    # 全部是同一个标签，返回作为叶子节点
    if labelList.count(labelList[0]) == len(labelList):
        return labelList[0]

    # 只剩最后一个特征了，其他特征都已经被添加到决策树了，返回出现次数最多的标签作为叶子节点
    if len(dataSet[0]) == 1:
        labelCount = {}
        for label in labelList:
            if label not in labelCount.keys():
                labelCount[label] = 0
            labelCount[label] += 1

        # labelCount.iteritems 得到为迭代器
        # op.itemgetter(1) 取每个数据的第二个域的值，即 label 的次数
        # 从大到小排序
        sortedLabelCount = sorted(labelCount.iteritems(), key=op.itemgetter(1), reverse=True)

        # 返回出现次数最多的标签作为叶子节点
        return sortedLabelCount[0][0]

    # 选择最佳特征
    bestFeat = chooseBestFeatureToSplit(dataSet)

    # 以最佳特征的名字创建节点
    bestFeatureName = featureNames[bestFeat]
    tree = {bestFeatureName: {}}

    # 将该特征从 featureNames 中删除
    del (featureNames[bestFeat])

    # 取该特征的所有值
    featValueList = [data[bestFeat] for data in dataSet]
    # 去除重复值
    featValueSet = set(featValueList)

    # 遍历最佳特征的每个值
    for featValue in featValueSet:
        subFeatureNames = featureNames[:]

        # 按该值取子集，子集中该特征值都被删除
        # 递归调用 createTree，以该值作为分叉，连接子集产生的子树
        tree[bestFeatureName][featValue] = createTree(splitDataSet(dataSet, bestFeat, featValue), subFeatureNames)

    """
    返回决策树，类似这样：
    {
        'feature1_name': {
            feature1_value1: {
                'feature2_name': {
                    feature2_value1: label_1,
                    feature2_value2: {
                        'feature3_name': label_2
                    }
                }
            },
            feature1_value2: label_3,
            feature1_value3: {
                'feature2_name': label_2
            }
        }
    }
    """
    return tree


# 使用决策树
def classify(tree, featureNameList, data):
    """
    tree             - 决策树
    featureNameList  - 特征名列表
    data             - 要分类的数据
    """

    # 取根节点的特征名
    featureName = tree.keys()[0]

    # 依据特征名找相应特征的下标
    featIndex = featureNameList.index(featureName)

    # 预测数据对应的特征值
    featureValue = data[featIndex]

    # 依据特征值找子树
    subTree = tree[featureName][featureValue]
    if isinstance(subTree, dict):
        # 不是叶子节点，递归调用
        return classify(subTree, featureNameList, data)
    else:
        # 叶子节点，直接返回分类标签
        return subTree


# 保存决策树
def storeTree(tree, filename):
    fw = open(filename, 'w')
    pickle.dump(tree, fw)
    fw.close()


# 读取决策树
def grabTree(filename):
    fr = open(filename)
    return pickle.load(fr)
