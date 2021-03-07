# coding=utf-8


class treeNode:
    """
    FP 树的节点

    name - 元素
    count - 出现次数
    nodeLink - 链接相同元素节点
    parent - 链接父节点
    children - 链接子节点 (可以有多个)
    """
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}

    def inc(self, numOccur):
        self.count += numOccur


def createTree(dataSet, minSup=1):
    """
    创建 FP 树

    dataSet - 字典，每个 key 是一条记录，比如 frozenset(['x','y'])，对应的 value 是该条记录出现的次数
    minSup - 最小可信度
    """

    headerTable = {}

    # 每条记录
    for trans in dataSet:
        # 每个元素
        for item in trans:
            # 计算每个元素出现的次数
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]

    for k in headerTable.keys():
        # 只取至少出现 minSup 次的元素
        if headerTable[k] < minSup:
            del (headerTable[k])

    # 得到单个元素的频繁项
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:
        # 所有元素都不是频繁项
        return None, None

    for k in headerTable:
        # headerTable 记录每个元素出现的次数、第一个链接节点(初始化为 None)
        headerTable[k] = [headerTable[k], None]

    # 创建树的根节点
    retTree = treeNode('Null Set', 1, None)

    # 第二次遍历，使用 items() 函数得到每个记录，以及该记录出现的次数
    for tranSet, count in dataSet.items():
        localD = {}
        # 遍历记录的每个元素
        for item in tranSet:
            # 该元素是频繁项，记下该条记录出现的次数
            if item in freqItemSet:
                localD[item] = headerTable[item][0]

        # 该记录至少有一个频繁项
        if len(localD) > 0:
            # 将该记录的频繁元素按出现次数降序排序
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]

            # 更新节点或添加节点
            updateTree(orderedItems, retTree, headerTable, count)

    # 返回 FP 树、元素表
    return retTree, headerTable


def updateTree(items, inTree, headerTable, count):
    # items[0] 是出现次数最多的元素
    if items[0] in inTree.children:
        # 子节点存在，增加计数
        inTree.children[items[0]].inc(count)
    else:
        # 子节点不存在，创建节点
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] is None:
            # 表头该元素的链接节点为空，链接到该元素
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            # 表头该元素的链接节点不为空，将该元素添加到链接的最后面
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])

    if len(items) > 1:
        # 对剩下的元素进行迭代
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)


def updateHeader(nodeToTest, targetNode):
    while nodeToTest.nodeLink is not None:
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    """
    通过 FP 树发现频繁项集

    inTree - createTree 的返回值 (这里没显式用到，而是通过 headerTable 隐式用到)
    headerTable 是 createTree 的返回值
    minSup - 要求最少要出现的次数
    preFix - 前缀频繁项，该函数在此基础上构建更大的频繁项，第一次进来为空集
    freqItemList - 存储所有的频繁项
    """

    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1])]

    # 迭代每个元素，从次数少的开始，既树的低层往上走
    for basePat in bigL:
        # 在前缀频繁项的基础上添加元素，得到新的更大的频繁项
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)

        # 添加频繁项
        freqItemList.append(newFreqSet)

        # 该元素的每个路径，从该元素的父节点到根节点，产生新的记录集合
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])

        # 通过新的记录集合产生新的 FP 树
        myCondTree, myHead = createTree(condPattBases, minSup)
        if myHead is not None:
            # 新的频繁项作前缀，通过新 FP 树，继续迭代
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)


def findPrefixPath(basePat, treeNode):
    """
    basePat - 元素
    treeNode - 元素的第一个链接节点
    """
    condPats = {}
    while treeNode is not None:
        prefixPath = []

        # 从该元素到根节点，组成一个新的项集
        ascendTree(treeNode, prefixPath)

        if len(prefixPath) > 1:
            # 产生键值对 (新项集:新项集出现次数)
            # 新项集不包括该元素 basePat
            # 新项集的次数取 basePat 在该路径上的次数
            condPats[frozenset(prefixPath[1:])] = treeNode.count

        # 该元素的下一个链接节点
        treeNode = treeNode.nodeLink

    return condPats


def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)

