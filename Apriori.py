# coding=utf-8


def createC1(dataSet):
    """
    初始化 C1 集合，C1 每个元素是只有一个项的项集
    每条数据是一个数组类似 [葡萄酒，尿布, 豆奶]，为方便可以用数字代替变成像 [1,2,3] 这样
    """

    C1 = []
    for transaction in dataSet:
        # 遍历每条数据里的每一项
        for item in transaction:
            # 不要重复的
            if not [item] in C1:
                # 每个项集用一个数组表示，长度为 1，后面会根据这个 C1 构建 C2、C3、Cn
                C1.append([item])
    C1.sort()

    # frozenset 使得项集不可改变
    return map(frozenset, C1)


def scanD(D, Ck, minSupport):
    """
    D - 每个元素是一条原始记录(会去掉重复的)
    Ck - 每个元素是有 k 个项的项集
    minSupport - 频繁项集的最小支持度

    基于 D, Ck, minSupport 寻找频繁项集 Lk
    """
    ssCnt = {}
    # 遍历每一条原始记录
    for tid in D:
        # 遍历每一个项集
        for can in Ck:
            # 如果该原始记录包含该项集，统计该项集出现的次数
            if can.issubset(tid):
                if can not in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1

    numItems = float(len(D))
    retList = []
    supportData = {}

    # 遍历每一个项集
    for key in ssCnt:
        # 支持度 = 项集出现次数/原始记录总数
        support = ssCnt[key] / numItems
        if support >= minSupport:
            # 支持度大于 minSupport 的是频繁项集
            retList.insert(0, key)
        # 记录所有项集的支持度
        supportData[key] = support

    # 返回频繁以及所有项集的支持度
    return retList, supportData


def aprioriGen(L, k):
    """
    L - 频繁项集 L(k-1)

    通过频繁项集 L(k-1) 计算项集 C(k)
    """

    # 用于存 Ck，而 Ck 的每个元素是有 k 个项的项集
    retList = []
    length = len(L)
    for i in range(length):
        for j in range(i + 1, length):
            # list(L[i])[:k-2] 是项集 i 除最后一个数据外的所有数据，比如项集 [1,2,3] 取到的就是 [1,2]
            L1 = list(L[i])[:k - 2]
            L2 = list(L[j])[:k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                # 比如 [1,2,3] 和 [1,2,4] 取到的 L1 和 L2 就是相等的，并且会合并成 [1,2,3,4]
                # 这里的原理是这样
                # 如果 [1,2,3,4] 是频繁项集，那它的子集也必然是频繁项集
                # 也就是说必然存在频繁项集 [1,2,3] 和 [1,2,4]
                # 按照这种方法就可以找出所有 L(k) 的潜在组合，也就是 C(k)，而不用对所有的组合进行计算
                # 看该章节的图会更清晰
                retList.append(L[i] | L[j])

    # 返回 Ck
    return retList


def apriori(dataSet, minSupport):
    """
    寻找 dataSet 中支持度大于 minSupport 的频繁项集
    """

    # 初始化 C1
    C1 = createC1(dataSet)

    # 原始记录去掉重复值
    D = map(set, dataSet)

    # 取 C1 中支持度大于 minSupport 的频繁项集 L1，以及 C1 中所有项集的支持度
    L1, supportData = scanD(D, C1, minSupport)

    # 添加频繁项集 L1
    L = [L1]

    k = 2
    # L[k - 2] 是频繁项集 L(k-1)
    while len(L[k - 2]) > 0:
        # 通过频繁项集 L(k-1) 计算 Ck
        # 注意：这里 L[k-2] 其实是数组 L 的最后一个值，该值是频繁项集 L(k-1)
        Ck = aprioriGen(L[k - 2], k)

        # 取 Ck 中支持度大于 minSupport 的项集 Lk，以及 Ck 中所有项集的支持度
        Lk, supK = scanD(D, Ck, minSupport)

        # ====== 字典的 update：
        # ====== 如果 supK 的 key 在 supportData 也存在则用 supK 值替换 supportData 中的值，
        # ====== 不存在则添加到 supportData
        supportData.update(supK)

        # 添加频繁项集 Lk，Lk 的每个元素是一个项集，每个项集有 k 个元素
        L.append(Lk)

        # 继续寻找下一个频繁项集 L(k+1)
        k += 1

    # 返回所有的频繁项集 L，和所有项集的支持度
    return L, supportData


def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    """
    freqSet - 频繁项集
    H - 每个元素是 freqSet 的子集比如 {1,2}，每个元素具有相同长度
    supportData - 每个项集的支持度
    brl - 已找到的关联规则
    minConf - 最小可信度

    把 H 的每个元素作为推导结果，freqSet 减去推导结果作为前提，计算可信度
    """
    prunedH = []
    for conseq in H:
        # 关联规则 {freqSet-conseq} -> {conseq} 的可信度 = 支持度(freqSet)/支持度(freqSet-conseq)
        # 既出现 {freqSet-conseq} 时会出现 {conseq} 的概率，可信度最大是 1
        conf = supportData[freqSet] / supportData[freqSet - conseq]

        # 只取可信度 > minConf 的规则
        if conf >= minConf:
            # 添加关联规则(前提、推导、可信度)
            brl.append((freqSet - conseq, conseq, conf))

            # 添加关联规则的推导
            prunedH.append(conseq)

    return prunedH


def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    """
    freqSet - 是频繁项集
    H - 每个元素是频繁项集的子集比如 {1,2}，每个元素具有相同长度，每个元素都可作为推导结果
    brl - 存储关联规则
    supportData - 存储所有项集的支持度
    minConf - 最小可信度
    """

    # 取 H 每个元素的长度 m，在 generateRules 开始调用时是 1
    m = len(H[0])

    # 能进入这个函数的 len(freqSet) 至少是 3，在 generateRules 开始调用时必然大于 m+1，意味着可以合并推导结果
    if len(freqSet) > (m + 1):
        # 比如 freqSet 是 {1,2,3,4}
        # 一开始 H 是 [{1},{2},{3},{4}]
        # 则可以将 H 进一步合并为 [{1,2},{1,3},{1,4},{2,3},{2,4},{3,4}] 产生类似 {3,4}->{1,2} 的结果

        # 产生长度为 m+1 的 H
        # 比如产生 [{1,2},{1,3},{1,4},{2,3},{2,4},{3,4}]
        Hmp1 = aprioriGen(H, m + 1)

        # 产生推导结果长度为 m+1 的关联规则
        # 比如产生 {3,4}->{1,2}
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)

        # 长度为 m+1 的推导结果有多个，可以继续尝试合并为 m+2
        # 比如进一步推导出 {3} -> {1,2,4}
        if len(Hmp1) > 1:
            # 迭代
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


def generateRules(L, supportData, minConf=0.7):
    """
    寻找关联规则

    如果某条规则并不满足最小可信度要求，那么该规则的所有子集也不会满足最小可信度要求
    比如 0,1,2->3 不满足最小可信度要求，那么任何左部为 {0,1,2} 子集的规则也不会满足最小可信度要求

    L - 频繁项集
    supportData - 项集支持度
    minConf - 最小可信度

    L 和 supportData 是 apriori 的返回值
    """
    bigRuleList = []

    # 从 L[1] (即 L2) 开始，因为至少有 2 个元素才会有关联
    for i in range(1, len(L)):
        # 遍历 Li 的每个频繁项集
        for freqSet in L[i]:
            # H1 获取频繁项集的每个元素
            H1 = [frozenset([item]) for item in freqSet]

            if i > 1:
                # 项集大于两个元素，可以有多种推导，比如 {0}->{1,2}，{0,2}->{1}
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                # 项集只有两个元素，只有一种推导
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)

    # 返回所有可信度大于 minConf 的关联规则
    return bigRuleList
