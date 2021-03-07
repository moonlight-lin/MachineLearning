# coding=utf-8
import numpy as np


def pca(dataMat, topNfeat=9999999):
    """
    dataMat - 原数据集
    topNfeat - 压缩为 topNfeat 个特征
    """

    # 所有数据减去平均值
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals

    # 计算协方差矩阵，协方差用于衡量两个变量(特征)的总体误差
    # 正值表示有正相关性，负值表示有负相关性，0 表示两个变量是统计独立的
    # 而方差是协方差的一种特殊情况，即当两个变量是相同的情况
    covMat = np.cov(meanRemoved, rowvar=False)

    # 计算协方差矩阵(n阶矩阵) covMat 的特征值向量 eigVals (维度 n*1) 和特征向量矩阵 eigVects (维度 n*n)
    # eigVals 的每个值是一个特征值，eigVects 的每一列是一个特征向量，所有特征向量之间都是线性无关的
    # 满足 covMat * eigVects[:,j] = eigVals[j] * eigVects[:,j]
    # 注意这里的特征、特征向量是针对协方差矩阵的，不是针对数据集的
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))

    # 对特征值向量进行从小到大的排序，eigValInd 的值是 eigVals 的下标
    eigValInd = np.argsort(eigVals)

    # 步长 -1 所以从最后一个 (既最大的) 开始取，取 topNfeat 个最大的值的下标
    eigValInd = eigValInd[:-(topNfeat + 1):-1]

    # 通过下标取特征值最大的 topNfeat 个特征向量得到 redEigVects (维度 n*topNfeat)
    redEigVects = eigVects[:, eigValInd]

    # 使用新的特征向量矩阵对原始数据进行降维
    lowDDataMat = meanRemoved * redEigVects

    # 使用新的特征向量矩阵将原始数据集矩阵转换到新的空间
    reconMat = (lowDDataMat * redEigVects.T) + meanVals

    # 返回降维后的原始数据，和转换到新空间的数据
    return lowDDataMat, reconMat

