import operator

import numpy as np
import cv2
# r, c = np.mgrid[0:5:1, 0:5:1]
# r -= int((5-1)/2)
# c -= int((5-1)/2)
# print(np.sqrt(4))
# sigma2 = pow(1.5, 2.0)
# sigma4 = pow(1.5, 4.0)
# print('sigma2:', sigma2)
# print('sigma4:', sigma4)
# norm2 = np.power(r, 2.0) + np.power(c, 2.0)
# print(norm2)
# kernel = (norm2 / sigma4 - 2 / sigma2) * np.exp(-norm2 / (2 * sigma2))
# print(kernel)
# import scipy.signal
#
# image = [[1, 2, 3, 4, 5, 6, 7],
#          [8, 9, 10, 11, 12, 13, 14],
#          [15, 16, 17, 18, 19, 20, 21],
#          [22, 23, 24, 25, 26, 27, 28],
#          [29, 30, 31, 32, 33, 34, 35],
#          [36, 37, 38, 39, 40, 41, 42],
#          [43, 44, 45, 46, 47, 48, 49]]
#
# filter_kernel = [[-1, 1, -1],
#                  [-2, 3, 1],
#                  [2, -6, 0]]
#
# res = scipy.signal.convolve2d(image, filter_kernel,
#                               mode='same', boundary='fill', fillvalue=0)
# print(res)

# a = np.ones((7, 7))
# kernel = np.ones((5, 5))
# b = cv2.filter2D(a, -1, kernel=kernel, borderType=cv2.BORDER_CONSTANT)
# print(b)

# a = np.ones((100, 50))
# print(a.shape)
# for i in range(0, 100):
#     for j in range(0, 50):
#         print(a[i][j])
#
# a = np.ones((7, 7))
# b = np.zeros((5, 5))
# print(b.shape)
# b = a[1:6, 1: 6]
# print(b.shape)
# print(b)
# classCount = {'a': 2, 'b': 3, 'c': 1}
# sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(0), reverse=True)
# print(sortedClassCount[0][0])
# fr = open('ID3.txt')
# lenses = [inst.strip().split(' ') for inst in fr.readlines()]
# print(len(lenses[0]))
# label = [1, 2, 3]
# b = label[:]
# b[1] = 3
# print(label, b)
# a = [[1, 2],
#      [2, 3]]
# print(int(2<1))
# def a():
#      return [False, None]
# print(a()[1])
a = 'asdf'
print(a[:-1])