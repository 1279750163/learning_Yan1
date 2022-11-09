import operator
from math import log

def calcShannonEnt(dataSet):
    """
    计算给定数据集的香农熵
    :param dataSet: 给定的数据集
    :return: 返回香农熵
    """
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for label in labelCounts.keys():
        prob = float(labelCounts[label]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    """按照给定特征划分数据集"""
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    """选择最好的数据集划分方式"""
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bsetInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bsetInfoGain:
            bsetInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classlist):
    """获取出现次数最好的分类名称"""
    classCount = {}
    for vote in classCount:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classlist[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def create_Tree(dataset, labels):
    """ 创建树 """
    classlist = [example[-1] for example in dataset]
    if classlist.count(classlist[0]) == len(classlist):
        return classlist[0]
    if len(dataset[0]) == 1:
        return majorityCnt(classlist)
    bestFeat = chooseBestFeatureToSplit(dataSet=dataset)
    bestFeatLabel = labels[bestFeat]
    myTree = { bestFeatLabel: {}}
    del labels[bestFeat]
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        sublabel = labels[:]
        myTree[bestFeatLabel][value] = create_Tree(splitDataSet(dataSet=dataset, axis=bestFeat, value=value), sublabel)
    return myTree

if __name__ == '__main__':
    f = open('ID3.txt')
    lenses = [inst.strip().split(' ') for inst in f.readlines()]
    lensesLabel = ['age', 'prescript', 'astigamtic', 'tearRate']
    lensesTree = create_Tree(lenses, lensesLabel)
    print(lensesTree)