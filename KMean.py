# coding=utf-8
import numpy as np


def distEclud(vecA, vecB):
    """
    计算两个向量的距离
    """
    return np.sqrt(sum(np.power(vecA - vecB, 2)))


def randCent(dataSet, k):
    """
    随机初始化簇中心点
    """
    n = np.shape(dataSet)[1]

    # 用于存储 k 个簇中心点
    centroids = np.mat(np.zeros((k, n)))

    # 为 k 个簇中心点的每一个特征赋值
    for j in range(n):
        # 随机产生一个 (k,1) 矩阵，值的范围在该特征的最大和最小值之间
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = np.mat(minJ + rangeJ * np.random.rand(k, 1))
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    """
    dataSet - 要进行聚类的数据
    k - 要将数据分成 k 个聚类
    distMeas - 计算向量距离的函数
    createCent - 初始化 K 个簇中心点的函数
    """

    m = np.shape(dataSet)[0]

    # 存储每一个数据属于哪个簇，与簇中心点的距离是多少
    clusterAssment = np.mat(np.zeros((m, 2)))

    # 初始化 K 个簇中心点
    centroids = createCent(dataSet, k)

    # 不断的迭代，直到所有的数据分类不再改变
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False

        # 遍历每一个数据
        for i in range(m):

            # 保存距离最近的簇中心点，及其距离
            minDist = np.inf
            minIndex = -1

            for j in range(k):
                # 计算该数据与不同的簇中心点的距离
                distJI = distMeas(centroids[j, :], dataSet[i, :])

                # 取距离最小的那个簇
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j

            # 只要有一个数据的分类与上次迭代的结果不同，就会继续迭代所有数据
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True

            # 保存该数据所属的簇，以及与簇中心点的距离
            clusterAssment[i, :] = minIndex, minDist ** 2

        for cent in range(k):
            # 获取该簇的所有数据
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]

            # 重新计算该簇的中心点，新的中心点每一个特征的值，是该簇所有数据在该特征的平均值
            centroids[cent, :] = np.mean(ptsInClust, axis=0)

    # 返回 K 个簇中心点，以及所有数据所属簇、与簇中心点的距离
    return centroids, clusterAssment


def biKmeans(dataSet, k, distMeas=distEclud):
    """
    二分 K-Mean 算法

    dataSet - 要进行聚类的数据
    k - 要将数据分成 k 个聚类
    distMeas - 计算向量距离的函数
    """
    m = np.shape(dataSet)[0]

    # 存储每一个数据属于哪个簇，与簇中心点的距离是多少
    clusterAssment = np.mat(np.zeros((m, 2)))

    # 初始化簇中心点，只有一个，每个特征值是所有点的平均值
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]

    for j in range(m):
        # 初始化所有数据距中心点距离
        clusterAssment[j, 1] = distMeas(np.mat(centroid0), dataSet[j, :]) ** 2

    # 每次迭代增加一个簇
    while len(centList) < k:
        lowestSSE = np.inf
        bestClustAss = bestCentToSplit = bestNewCents = None

        # 遍历每一个簇
        for i in range(len(centList)):
            # 取该簇的所有数据
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == i)[0], :]

            # 使用普通的 K-Mean 算法将该簇再分为两个簇
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)

            # 计算新的 2 个簇总的方差和
            sseSplit = sum(splitClustAss[:, 1])

            # 计算剩下的簇总的方差和
            sseNotSplit = sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A != i)[0], 1])

            # 保存使总方差变小的划分
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit

        # 新划分的一部分数据赋予新的簇
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)

        # 另一部分数据维持原来的簇
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit

        # 改变用于划分的簇的中心点
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]

        # 添加新簇的中心点
        centList.append(bestNewCents[1, :].tolist()[0])

        # 改变用于划分的数据的值
        clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss

    # 返回结果
    return np.mat(centList), clusterAssment
