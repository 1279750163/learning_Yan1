import csv
import sys

import numpy as np

import cv2


# img = cv2.imread('img/250.png', 0)
# h, w = img.shape
# dst = np.float32([[0, 0],
#                  [0, h/2],
#                  [w/2, h - 1],
#                  [w - 1, 0]]).reshape(-1, 1, 2)
# img = cv2.polylines(img, [np.int32(dst)], True, 0, 1, cv2.LINE_AA)
# cv2.imshow('a', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# x = np.array([[1, 3, 5], [3, 2, 4]])
# print(np.argsort(x, axis=1))
# y = np.array(([[2, 3, 4]]))
# print(np.append(x, y, axis=0))
# a = [1, 2, 3]
# b = [4, 5, 6, 7]
# print(type(a))
# a.extend(b)
# print(a)
# l = ['Python', 'C++', 'Java']
# # 追加元素
# l.append('PHP')
# print(l)
def input_matrix():
    # 第一行输入两个数 n、 m，表示输入输入数据是 n 行 m 列的二维数组
    matrix = list()
    input1 = sys.stdin.readline().strip().split(' ')
    m, n = input1[0], input1[1]
    for i in range(int(m)):
        value = list(map(int, sys.stdin.readline().strip().split(' ')))
        matrix.append(value)
    print("打印保存的输入数据：")
    print(matrix)


if __name__ == "__main__":
    # input_matrix()
    # data = np.loadtxt('mnist_train_70000.csv', delimiter=',', userows=(6, 7))
    # print(data)
    # csv_reader = (csv.reader(open('mnist_train_70000.csv', encoding='utf-8')))
    # i = 0
    # dataSet = []
    # for row in csv_reader:
    #     if i != 0:
    #         row = list(map(int, row))
    #         dataSet.append(row)
    #     i += 1
    # dataSet = np.array(dataSet)
    # dataSet = dataSet[:, 1:]
    # print(dataSet[1:4])
    a = np.array([1, 3, 4, 5, 6])
    print(a[:(2-1)])
