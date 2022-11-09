import csv
import numpy as np
from os import listdir
import KNN

def loadCSV(filename):
    csv_reader = (csv.reader(open(filename, encoding='utf-8')))
    i = 0
    dataSet = []
    for row in csv_reader:
        if i != 0:
            row = list(map(int, row))
            dataSet.append(row)
        i += 1
    dataSet = np.array(dataSet)
    # dataSet = dataSet[:, 1:]
    # print(dataSet[1:4])
    return dataSet


def handwritingClassTest():
    hwLabels = []
    ratio = 0.01
    allData = loadCSV('mnist_train_70000.csv')
    n = allData.shape[0]
    numtest = int(n * ratio)
    dataSet = allData[:, 1:]
    trianData = dataSet[:(n - numtest)]
    testData = dataSet[(n - numtest):]
    Labels = allData[:, :1].reshape(1, -1).flatten()
    trianLabels = Labels[:(n - numtest)]
    testLabels = Labels[(n - numtest):]
    errorCount = 0
    for i in range(numtest):
        classNumber = testLabels[i]
        classifierResult = KNN.classify(testData[i], trianData, trianLabels, 10)
        print('the classifier came back with :%d, the real answer is: %d' % (classifierResult, classNumber))
        if (classifierResult != classNumber): errorCount += 1
    print('\n the total number of errors is : %d' % errorCount)
    print('\n total error rate is %f' % (errorCount/float(numtest)))
if __name__ == '__main__':
    handwritingClassTest()

