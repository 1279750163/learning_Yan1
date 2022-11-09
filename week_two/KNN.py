import numpy as np
from numpy import *
import operator


# 定义KNN算法分类器函数
# 函数参数包括：（测试数据， 训练数据， 分类， K值）
def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffmat = diffMat ** 2
    sqDistances = sqDiffmat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()  #排序并返回index
    # 选择距离最近的K个值
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        # print(voteIlabel)
        """
        .get(key[, value])
        key 字典中的键 value 可选 若键不存在，返回该默认值 
        """
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createDataSet():
    group = array([[1, 1.1], [1, 1], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# 将文本记录转为numpy
def file2matrix(filepath):
    f = open(filepath)
    arraylines = f.readlines()
    numberOfLines = len(arraylines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arraylines:
        line = line.strip()  # 删除首尾空格
        listFromLine = line.split('\t')  # 以\t进行拆分
        # print(listFromLine)
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(listFromLine[-1])
        index += 1
    return returnMat, classLabelVector


# 归一化处理函数
def autoNorm(dataSet):
    """
    min(0)返回该矩阵中每一列的最小值
    min(1)返回该矩阵中每一行的最小值
    max(0)返回该矩阵中每一列的最大值
    max(1)返回该矩阵中每一行的最大值
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))  #tile复制minval成dataset形状
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


if __name__ == '__main__':
    # group, labels = createDataSet()
    # target = classify([0, 0], group, labels, 3)
    # print(target)
    dateData, dateLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(dateData)
    print(ranges)