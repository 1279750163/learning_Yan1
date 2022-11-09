import operator

import cv2
import matplotlib.pyplot as plt
import numpy as np


def calculate_instance(des1, des2):
    inst = 0.0
    for i in range(0, 128):
        inst += (des1[i] - des2[i]) ** 2
    return inst


def calculate_Allinstance(des, descriptors):
    insts = {}
    for i, des2 in enumerate(descriptors):
        inst = calculate_instance(des, des2)
        insts[i] = inst
    sortinsts = sorted(insts.items(), key=operator.itemgetter(1), reverse=False)
    return sortinsts[0], sortinsts[1]


if __name__ == '__main__':
    img1 = cv2.imread('img/250.png', 0)
    img2 = cv2.imread('img/366.png', 0)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    print(img1.shape)
    print(img2.shape)
    print(len(kp1), len(des1))
    print(len(kp2), len(des2))
    print(kp1[0].pt[1])
    matches = {}
    for i, des in enumerate(des1):
        min1, min2 = calculate_Allinstance(des, des2)
        matches[i] = [min1, min2]
    good = {}
    for i in range(0, len(kp1)):
        if matches[i][0][1] < matches[i][1][1] * 1:
            good[i] = matches[i][0]
    print(len(good))
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    nwidth = w1 + w2
    nheight = max(h2, h1)
    newimg = np.zeros((nheight, nwidth, 3), np.uint8)
    for i in range(3):
        newimg[:h1, : w1, i] = img1
        newimg[:h2, w1 : w1 + w2, i] = img2

    if len(good) > 10:
        for i in good.keys():
            pt1 = (int(kp1[i].pt[0]), int(kp1[i].pt[1]))
            pt2 = (int(kp2[good[i][0]].pt[0] + w1), int(kp2[good[i][0]].pt[1]))
            cv2.line(newimg, pt1, pt2, (0, 255, 0))
    plt.imshow(newimg)
    plt.show()
    # implot = plt.imshow(img1, cmap='gray')
    # for i in range(0, len(kp1)):
    #     plt.scatter(kp1[i].pt[0], kp1[i].pt[1], s=10)
    # plt.show()
    # cv2.imshow('a', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # a, b = calculate_Allinstance(des1[3], des2)
    # print(a, b)
