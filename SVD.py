# coding=utf-8
import numpy as np


def imgCompress(numSV=3, thresh=0.8):
    """
    numSV - 保留的奇异值数目
    thresh - 阀值，大于这个值当成 1
    """

    myl = []
    # 将图片读入存储在矩阵中，假设是黑白只有 01 值
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)

    myMat = np.mat(myl)
    print "****original matrix******"
    printMat(myMat, thresh)

    # 求解奇异值矩阵
    U, Sigma, VT = np.linalg.svd(myMat)

    SigRecon = np.mat(np.zeros((numSV, numSV)))
    for k in range(numSV):
        # 将 Sigma 转换为 numSV 阶矩阵
        SigRecon[k, k] = Sigma[k]

    # 通过奇异值矩阵降维
    reconMat = U[:, :numSV] * SigRecon * VT[:numSV, :]

    print "****reconstructed matrix using %d singular values******" % numSV
    printMat(reconMat, thresh)


def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                print 1,
            else:
                print 0,
        print ''

