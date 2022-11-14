import numpy as np
import cv2
import matplotlib.pyplot as plt

# 定义卷积的函数
def conv2D(image, weight):
    height, width = image.shape
    h, w = weight.shape
    # 经滑动卷积操作后得到的新的图像的尺寸
    new_h = height - h + 1
    new_w = width - w + 1
    new_image = np.zeros((new_h, new_w), dtype=np.float64)
    # 进行卷积操作
    for i in range(new_w):
        for j in range(new_h):
            new_image[i, j] = np.sum(image[i: i+h, j: j+w] * weight)

    # 去掉矩阵乘法后小于0和大于255的原值，重置为0和255
    new_image = new_image.clip(0, 255)
    new_image = np.rint(new_image).astype('uint8')
    return new_image

if __name__ == '__main__':
    a = np.array([[1, 1, 1, 1, 1],
                  [1, 2, 3, 4, 5],
                  [2, 2, 2, 2, 2],
                  [3, 2, 1, 1, 1],
                  [1, 1, 1, 1, 1]])
    b = np.ones([3, 3])
    a_ = conv2D(a, b)
    print(a_)
