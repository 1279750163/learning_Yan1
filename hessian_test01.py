"""
    测试先图像二阶导再高斯模糊
"""

import cv2
import numpy as np
def readIamge(filename):
    img = cv2.imread(filename, 0)
    if img is None:
        print('Invalid image:' + filename)
        return None
    else:
        print('Image successful read...')
        return img

def Get_hessian(img):
    height, width = img.shape[0], img.shape[1]
    xxDerivate = np.zeros((height, width), np.float64)
    yyDerivate = np.zeros((height, width), np.float64)
    xyDerivate = np.zeros((height, width), np.float64)
    for i in range(2, height-1):
        for j in range(2, width-1):
            xxDerivate[i][j] = img[i][j+1] - 2 * img[i][j] + img[i][j-1]
            yyDerivate[i][j] = img[i+1][j] - 2 * img[i][j] + img[i-1][j]
            xyDerivate[i][j] = int(img[i+1][j+1]) - int(img[i][j+1]) - int(img[i+1][j]) + int(img[i][j])
    print(xxDerivate.shape)
    return xxDerivate, yyDerivate, xyDerivate

def Get_kernel(sigma, size):
    h, w = size
    r, c = np.mgrid[0:h:1, 0:w:1]
    r -= int((h-1)/2)
    c -= int((w-1)/2)
    kernel = 1/(np.sqrt(2 * np.pi * (sigma * sigma))) * np.exp(-((np.power(r, 2.0) + np.power(c, 2.0)) / (2 * sigma * sigma)))
    return kernel

if __name__ == '__main__':
    img = readIamge('dongman2.jpg')
    print(type(img))
    # a = np.ones((3, 3))
    img2 = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
    print(img.shape)
    print(img2.shape)
    xxDerivate, yyDerivate, xyDerivate = Get_hessian(img2)
    h, w = xxDerivate.shape[0] - 4, xxDerivate.shape[1] - 4
    print(h, w)
    # x = np.zeros((h, w), np.int64)
    # y = np.zeros((h, w), np.int64)
    # xy = np.zeros((h, w), np.int64)
    # print(x.shape)
    kernel = Get_kernel(0.5, (5, 5))
    # print(kernel)
    xxDerivate = cv2.filter2D(xxDerivate, -1, kernel, borderType=cv2.BORDER_CONSTANT).astype(int)
    xyDerivate = cv2.filter2D(xyDerivate, -1, kernel, borderType=cv2.BORDER_CONSTANT).astype(int)
    yyDerivate = cv2.filter2D(yyDerivate, -1, kernel, borderType=cv2.BORDER_CONSTANT).astype(int)
    x = xxDerivate[2:h+2, 2:w+2]
    y = yyDerivate[2:h+2, 2:w+2]
    xy = xyDerivate[2:h+2, 2:w+2]
    print(x.shape)
    print(xy.shape)
    print(y.shape)
    xy[xy < 0] = 0
    x[x < 0] = 0
    y[y < 0] = 0
    xy = np.uint8(xy)
    x = np.uint8(x)
    y = np.uint8(y)
    dst = np.hstack((img, x, y, xy))

    cv2.imshow('z', dst)
    # print(img)
    # cv2.imshow('aa', img)
    # cv2.imshow('aa', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()