# coding=utf-8
import numpy as np


def ecludSim(inA, inB):
    """
    欧氏距离
    """
    return 1.0 / (1.0 + np.linalg.norm(inA - inB))


def pearsSim(inA, inB):
    """
    皮尔逊相关系数
    """
    if len(inA) < 3:
        return 1.0

    # corrcoef 返回的是矩阵，[0][1] 代表 inA 和 inB 间的相关系数
    return 0.5 + 0.5 * np.corrcoef(inA, inB, rowvar=False)[0][1]


def cosSim(inA, inB):
    """
    余弦相似度
    """
    num = float(inA.T * inB)
    denom = np.linalg.norm(inA) * np.linalg.norm(inB)
    return 0.5 + 0.5 * (num / denom)


def standEst(dataMat, user, simMeas, item):
    """
    预测 user 对 item 的评分 (没有使用 SVD)

    dataMat - 每行是一个用户，每列是该用户对某物品的评分
    user - 要预测的用户
    item - 要预测的物品
    simMeas - 用于计算相似度的函数，ecludSim、pearsSim、cosSim 中的一个
    """
    n = np.shape(dataMat)[1]

    simTotal = 0.0
    ratSimTotal = 0.0

    # 遍历所有物品
    for j in range(n):
        # 取 user 对 j 的评分
        userRating = dataMat[user, j]

        # 没评分则跳过
        if userRating == 0:
            continue

        # 找出所有既对 item 有评分，又对 j 有评分的用户
        overLap = np.nonzero(np.logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:
            # 计算找出来的用户对 item 评分和对 j 评分的相似度
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])

        # 累加相似度
        simTotal += similarity

        # 累加 user 对 j 的评分与相似度的乘积
        ratSimTotal += similarity * userRating

    if simTotal == 0:
        return 0
    else:
        # 预测 user 对 item 的评分，计算方法类似 (a1*x1+a2*x2+...+an*xn)/(x1+x2+...+xn) 其中 xi 是 i 和 item 的相似度，ai 是 user 对 i 的评分
        return ratSimTotal / simTotal


def svdEst(dataMat, user, simMeas, item):
    """
    另一种预测评分的方法：基于 SVD 的评分估计

    dataMat - 每行是一个用户，每列是该用户对某物品的评分
    user - 要预测的用户
    item - 要预测的物品
    simMeas - 用于计算相似度的函数，ecludSim、pearsSim、cosSim 中的一个
    """
    n = np.shape(dataMat)[1]

    simTotal = 0.0
    ratSimTotal = 0.0

    # np.linalg.svd(dataMat) 求解 dataMat 的奇异值矩阵
    U, Sigma, VT = np.linalg.svd(dataMat)

    # 将 Sigma 转换为只有对角线有值的 r 阶矩阵
    # 这里直接取 4 个
    # Sigma 是有排好序的，正常应该取前 r 个，使其平方和大于90%，或固定一个比较大的数值
    # 然后 dataMat(m,n) ≈ U(m,r) * Sig(r,r) * VT(r,n)
    Sig4 = np.mat(np.eye(4) * Sigma[:4])

    # xformedItems 维度是 (n*r)，通过转换只保留 r 个用户的所有评分，实现了降维，减少了数据量
    # 是不是也可以保留所有用户的 r 个评分？是不是可以用 VT 映射到另一个空间
    xformedItems = dataMat.T * U[:, :4] * Sig4.I

    # 遍历所有物品
    for j in range(n):
        # 取 user 对 j 的评分
        userRating = dataMat[user, j]

        # 没评分则跳过
        if userRating == 0 or j == item:
            continue

        # 计算 r 个用户对 j 和 item 的评分的相似度
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)

        # 累加相似度
        simTotal += similarity

        # 累加 user 对 j 的评分与相似度的乘积
        ratSimTotal += similarity * userRating

    if simTotal == 0:
        return 0
    else:
        # 预测 user 对 item 的评分，计算方法类似 (a1*x1+a2*x2+...+an*xn)/(x1+x2+...+xn) 其中 xi 是 i 和 item 的相似度，ai 是 user 对 i 的评分
        return ratSimTotal / simTotal

