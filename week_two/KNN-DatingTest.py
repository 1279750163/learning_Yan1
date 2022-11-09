import numpy as np
from KNN import *
import matplotlib.pyplot as plt

# 定义datingTestSet.txt测试算法的函数
def datingClassTest(dateData, dateLabels, ratio):
    datingData, datingLabels = dateData, dateLabels
    normMat, ranges, minVals = autoNorm(datingData)
    m = normMat.shape[0]
    numTestVecs = int(m * ratio) # 测试数据行数
    errorCount = 0 # 定义变量来存储错误分类数
    rand = int((m - numTestVecs) * np.random.random()) # 定义在所有数据中取numTestVecs个连续随机数
    print(rand)
    for i in range(numTestVecs):
        classifierResult = classify(normMat[rand+i, :], append(normMat[:rand, :], normMat[rand+numTestVecs:m, :], axis=0), (datingLabels[:rand]+dateLabels[rand+numTestVecs:m]), 5)
        print('the classifier came back with : %d, the real answer is : %d' % (classifierResult, int(datingLabels[rand+i])))
        if classifierResult != datingLabels[rand+i]:
            errorCount += 1
    print('the total error is : %f' % (errorCount/float(numTestVecs)))


def classifypersion():
    reslutList = ['not at all', 'in small doses', 'in large doses']
    ffMiles = float(input('frequent flier miles earned per year :'))
    percentTats = float(input('percentage of time spent playing video games :'))
    iceCream = float(input('liters of ice creamconsued per year :'))
    datingData, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingData)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify(((inArr - minVals) / ranges), normMat, datingLabels, 5)
    print('You will probably like this persion : %s' % reslutList[int(classifierResult) - 1])



if __name__ == '__main__':
    classifypersion()
    # dateData, dateLabels = file2matrix('datingTestSet.txt')
    # dateLabels = list(map(int, dateLabels))
    # datingClassTest(dateData=dateData, dateLabels=dateLabels, ratio=0.1)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # 对2，3列数据分析
    # ax.scatter(dateData[:, 1], dateData[:, 2], c=dateLabels)
    # plt.xlabel('Percentage of Time Spent Playing Video Games')
    # plt.ylabel('Liters of Ice Cream Consumed Per Week')
    # fig.show()
    # 对1，2列数据分析
    # ax.scatter(dateData[:, 0], dateData[:, 1], c=dateLabels)
    # plt.xlabel('Miles of plane Per year')
    # plt.ylabel('Percentage of Time Spent Video Games')
    # fig.show()