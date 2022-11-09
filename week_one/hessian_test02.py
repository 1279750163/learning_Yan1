import cv2
import sys

import numpy as np

if __name__ == '__main__':
    img = cv2.imread('dongman2.jpg')
    if img is None:
        print('Invalid image:')
        sys.exit()
    if img.shape != 1:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #
    # cv2.imshow('aaa', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    height, width = img.shape[0], img.shape[1]
    outImg = np.zeros((height, width), np.float64)
    filter = 2
    sigma = 0.5
    xxGaukernel = np.zeros((2 * filter + 1, 2 * filter + 1), np.float64)
    xyGaukernel = np.zeros((2 * filter + 1, 2 * filter + 1), np.float64)
    yyGaukernel = np.zeros((2 * filter + 1, 2 * filter + 1), np.float64)
    print(np.pi)
    # 构建高斯二阶偏导数模板
    for i in range(-filter, filter+1):
        for j in range(-filter, filter+1):
            yyGaukernel[i + filter][j + filter] = (1 - (i * i) / (sigma * sigma)) * np.exp(-1 * (i * i + j * j) / (2 * sigma * sigma)) * (-1 / (2 * np.pi * np.power(sigma, 4)))

            xxGaukernel[i + filter][j + filter] = (1 - (j * j) / (sigma * sigma)) * np.exp(-1 * (i * i + j * j) / (2 * sigma * sigma)) * (-1 / (2 * np.pi * np.power(sigma, 4)))

            xyGaukernel[i + filter][j + filter] = ((i * j) / (2 * np.pi * np.power(sigma, 6))) * np.exp(-1 * (i * i + j * j) / (2 * sigma * sigma))
    # count = 0
    # for i in range(0, 2 * filter + 1):
    #     for j in range(0, 2 * filter + 1):
    #         print(xxGaukernel[i][j])
    #         count += 1
    # print(count)
    print(height, width)
    xxDerivate = np.zeros((height, width), np.int64)
    yyDerivate = np.zeros((height, width), np.int64)
    xyDerivate = np.zeros((height, width), np.int64)
    # print(xxDerivate.shape)
    xxDerivate = cv2.filter2D(img, -1, kernel=xxGaukernel, borderType=cv2.BORDER_CONSTANT)
    xyDerivate = cv2.filter2D(img, -1, kernel=xyGaukernel, borderType=cv2.BORDER_CONSTANT)
    yyDerivate = cv2.filter2D(img, -1, kernel=yyGaukernel, borderType=cv2.BORDER_CONSTANT)
    print(xxDerivate.shape)
    # for i in range(0, height):
    #     for j in range(0, width):
    #             print(xxDerivate[i][j])
    dst = np.hstack((img, xxDerivate, yyDerivate, xyDerivate))
    cv2.imshow('aaa', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()