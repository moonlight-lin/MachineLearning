# coding=utf-8
import numpy as np


def binSplitDataSet(dataSet, feature, value):
    """
    给定特征和特征值，通过数组过滤方式将数据集合切分得到两个子集并返回

    dataSet - 样本数据, 每一行的最后一个值是 Y
    feature - 要切分的特征
    value   - 用于切分的特征值
    """

    # nonzero(dataSet[:,feature] > value) 返回两数组
    # 第1个的值表示行数，第2个的值表示列数，就是几行几列 > value，加上[0]取出所有满足条件的行
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]

    return mat0, mat1


def chooseBestSplit(dataSet, leafType, errType, tolS, tolN):
    """
    寻找最佳切分点

    dataSet  - 样本数据, 每一行的最后一个值是 Y
    leafType - 建立叶节点的函数
    errType  - 误差计算函数
    tolS     - 如果切分后的增益小于该值就不要切分
    tolN     - 如果切分后的样本小于该值就不要切分

    返回 (最佳切分特征，最佳切分特征值) 或 (None, 叶子节点)
    """

    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        # 如果数据集的所有值都相同，返回 None 表示不需要继续按特征划分，同时构建并返回叶子节点
        return None, leafType(dataSet)

    m, n = np.shape(dataSet)

    # 计算切分前的误差
    S = errType(dataSet)

    # 最佳的切分特征，切分特征值，切分增益，左子树的样本，右子树的样本
    bestIndex = 0
    bestValue = 0
    bestS = np.inf

    # 遍历每一个特征
    for featIndex in range(n - 1):
        # 遍历每一个特征值
        for splitVal in set(dataSet[:, featIndex]):
            # 切分
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)

            # 切分后的样本数不够
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                continue

            # 切分后的误差
            newS = errType(mat0) + errType(mat1)

            # 如果切分后的误差更小，保存特征、特征值、误差
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS

    # 如果误差的改进太小，就不切分了，构建并返回叶子节点
    if (S - bestS) < tolS:
        return None, leafType(dataSet)

    # 返回用于切分的特征、特征值
    return bestIndex, bestValue


def createTree(dataSet, leafType, errType, tolS, tolN):
    """
    创建 CART 树

    dataSet  - 样本数据, 每一行的最后一个值是 Y
    leafType - 建立叶节点的函数
    errType  - 误差计算函数
    tolS     - 如果切分后的增益小于该值就不要切分
    tolN     - 如果切分后的样本小于该值就不要切分
    """

    # 选择最适合用于切分的特征、特征值
    feat, val = chooseBestSplit(dataSet, leafType, errType, tolS, tolN)

    if feat is None:
        # 无法切分, 返回叶子节点
        return val

    # 切分特征、切分特征值
    retTree = {'spInd': feat, 'spVal': val}

    # 按切分特征、切分特征值将数据集分为两个子集
    lSet, rSet = binSplitDataSet(dataSet, feat, val)

    # 递归生成左子树
    retTree['left'] = createTree(lSet, leafType, errType, tolS, tolN)

    # 递归生成右子树
    retTree['right'] = createTree(rSet, leafType, errType, tolS, tolN)

    return retTree


def regLeaf(dataSet):
    """
    用于构建回归树的叶子节点：所有样本 Y 值的平均值
    dataSet - 样本值，最后一位是 Y 值
    """
    return np.mean(dataSet[:, -1])


def regErr(dataSet):
    """
    用于计算回归树的误差：所有样本 Y 值均方差乘以样本数
    dataSet - 样本值，最后一位是 Y 值
    """
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


def linearSolve(dataSet):
    """
    计算最佳线性回归系数
    dataSet - 样本值，最后一位是 Y 值
    """
    m, n = np.shape(dataSet)

    # 取每一行除最后一位之外的数据，并且第一个值取 1
    X = np.mat(np.ones((m, n)))
    X[:, 1:n - 1] = dataSet[:, 0:n - 2]

    Y = dataSet[:, -1]
    xTx = X.T * X
    if np.linalg.det(xTx) == 0.0:
        # 没有逆矩阵无法计算
        raise ValueError('（X.T * X） 没有逆矩阵，无法求解')

    # 计算最佳回归系统 w
    ws = xTx.I * (X.T * Y)

    return ws, X, Y


def modelLeaf(dataSet):
    """
    用于构建模型树的叶子节点：计算最佳回归系数
    """
    ws, _, _ = linearSolve(dataSet)
    return ws


def modelErr(dataSet):
    """
    用于计算模型树的误差：预测值与实际值的平方差
    """

    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws

    return sum(np.power(Y - yHat, 2))


def createRegTree(dataSet):
    """
    创建 CART 回归树
    dataSet - 样本值，最后一位是 Y 值
    """
    return createTree(dataSet, leafType=regLeaf, errType=regErr, tolS=1, tolN=4)


def createModelTree(dataSet):
    """
    创建 CART 回归树
    dataSet - 样本值，最后一位是 Y 值
    """
    return createTree(dataSet, leafType=modelLeaf, errType=modelErr, tolS=1, tolN=4)


def regTreeEval(model, _):
    """
    根据回归树的叶子节点预测结果
    model - 叶子节点的值 (切分到该叶子节点的样本数据的均值)
    """
    return float(model)


def modelTreeEval(model, inDat):
    """
    根据模型树的叶子节点预测结果
    model - 叶子节点的值 (切分到该叶子节点的样本数据的最佳回归系数)
    inDat - 要预测的数据
    """

    # 将第一个 X0 值取为 1
    n = np.shape(inDat)[1]
    X = np.mat(np.ones((1, n + 1)))
    X[:, 1:n + 1] = inDat

    # 预测 Y = XW
    return float(X * model)


def isTree(obj):
    """
    判断节点是否叶子节点
    """
    return type(obj).__name__ == 'dict'


def predict(tree, inData, modelEval):
    """
    预测

    tree   - CART 树
    inData - 要预测的数据
    modelEval - 用于处理叶子节点的函数
    """
    if not isTree(tree):
        # 已经是叶子节点，处理叶子节点，返回预测结果
        return modelEval(tree, inData)

    if inData[tree['spInd']] > tree['spVal']:
        # 走左子树
        if isTree(tree['left']):
            # 递归左子树
            return predict(tree['left'], inData, modelEval)
        else:
            # 左子树已经是叶子节点
            return modelEval(tree['left'], inData)
    else:
        # 走右子树
        if isTree(tree['right']):
            # 递归右子树
            return predict(tree['right'], inData, modelEval)
        else:
            # 右子树已经是叶子节点
            return modelEval(tree['right'], inData)


def regPredict(tree, testData):
    """
    回归树预测

    tree - CART 回归树
    testData - 要预测的数据
    """
    m = len(testData)
    yHat = np.mat(np.zeros((m, 1)))

    for i in range(m):
        yHat[i, 0] = predict(tree, np.mat(testData[i]), regTreeEval)

    return yHat


def modelPredict(tree, testData):
    """
    模型树预测

    tree - CART 模型树
    testData - 要预测的数据
    """
    m = len(testData)
    yHat = np.mat(np.zeros((m, 1)))

    for i in range(m):
        yHat[i, 0] = predict(tree, np.mat(testData[i]), modelTreeEval)

    return yHat


def getMean(tree):
    """
    对树进行塌陷处理 (即返回树平均值)
    """
    if isTree(tree['right']):
        # 递归右子树
        tree['right'] = getMean(tree['right'])

    if isTree(tree['left']):
        # 递归左子树
        tree['left'] = getMean(tree['left'])

    # 取平均值
    return (tree['left'] + tree['right']) / 2.0


def prune(tree, testData):
    """
    对树进行后剪枝, 这里的树是回归树

    tree - 已经生成的树
    testData - 测试集
    """

    if np.shape(testData)[0] == 0:
        # 已经没有测试数据了，进行塌陷处理
        return getMean(tree)

    if isTree(tree['right']) or isTree(tree['left']):
        # 切分测试数据
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])

        if isTree(tree['left']):
            # 递归剪枝左子树
            tree['left'] = prune(tree['left'], lSet)

        if isTree(tree['right']):
            # 递归剪枝右子树
            tree['right'] = prune(tree['right'], rSet)

    if not isTree(tree['left']) and not isTree(tree['right']):
        # 如果经过剪枝后，左右都是叶子节点

        # 切分测试数据
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])

        # 计算测试数据切分后，和叶子节点之间的方差和
        errorNoMerge = sum(np.power(lSet[:, -1] - tree['left'], 2)) \
                       + sum(np.power(rSet[:, -1] - tree['right'], 2))

        # 合并左右叶子节点
        treeMean = (tree['left'] + tree['right']) / 2.0

        # 用合并后的叶子节点计算切分前测试数据的方差和
        errorMerge = sum(np.power(testData[:, -1] - treeMean, 2))

        if errorMerge < errorNoMerge:
            # 合并叶子节点后的方差和更小，返回合并后的叶子节点
            return treeMean
        else:
            # 否则返回合并前的树
            return tree
    else:
        # 左右子树有一个不是叶子节点，不合并
        return tree

