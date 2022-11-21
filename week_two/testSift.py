import random
import cv2
import numpy as np
from matplotlib import pyplot as plt

# f = open('pointdata_0.4.txt')
# for line in f.readlines():
#     points = list(map(float, line.strip().split(' ')))
#     print(points[0])


if __name__ == '__main__':
    img1 = cv2.imread('img/1.ppm', 0)
    img2 = cv2.imread('img/2.ppm', 0)
    # f = open('data_point/3Dpoint/pointdata_0.4.txt')
    # colors = ((0, 0, 0), (125, 0, 0), (255, 0, 0), (0, 125, 0), (0, 255, 0), (0, 0, 125), (0, 0, 255), (125, 125, 0),
    #           (0, 125, 125), (125, 255, 255))
    # print(img1.shape)
    # h1, w1 = img1.shape
    # h2, w2 = img2.shape
    # nwidth = w1 + w2
    # nheight = max(h2, h1)
    # newimg = np.zeros((nheight, nwidth, 3), np.uint8)
    # for i in range(3):
    #     newimg[:h1, : w1, i] = img1
    #     newimg[:h2, w1: w1 + w2, i] = img2
    # for line in f.readlines():
    #     points = list(map(float, line.strip().split(' ')))
    #     point01 = (round(points[0]), round(points[1]))
    #     point02 = (round(points[2] + w1), round(points[3]))
    #     print(point01)
    #     cv2.line(newimg, point01, point02, colors[random.randint(0, 9)])
    plt.imshow(img2)
    plt.show()