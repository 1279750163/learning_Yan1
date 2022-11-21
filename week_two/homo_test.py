"""
 对sift的匹配特征点求homography, 并测试Image Matching 和 Homography Estimation
 Image Matching 是计算模型输出的匹配点的 MMA（mean matching accuracy），Homography Estimation 是根据匹配点计算单应性矩阵再计算精度
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt


def MMATest(Hfile, pointfile, threshold):
    """
    计算模型输出的匹配点的 MMA（mean matching accuracy）
    :param Hfile: 真实单应矩阵文件
    :param pointfile: 匹配点的文件
    :param threshold: 经过单应变换后的点与my_sift匹配点的允许误差阈值
    :return:
    """
    # read in homography
    H = np.loadtxt(Hfile)
    matches = np.loadtxt(pointfile)
    point_nums = matches.shape[0]
    point01 = matches[:, :2]
    point02 = matches[:, 2: 4]
    point01_ = np.concatenate((point01, np.ones((point_nums, 1))), axis=1)
    # print(point02)
    # 计算point01经过单应变换后在第二张图的点的位置
    projection = np.matmul(H, point01_.T).T
    projection = projection / projection[:, 2: 3]
    projection = projection[:, : 2]
    # print(projection)
    # 评估匹配点的偏差
    result = np.linalg.norm(point02 - projection, axis=1)
    result_nums = np.sum(result < threshold)
    print(result_nums)
    print("threshold : {} , mean matching accuracy : {} ".format(threshold, float(result_nums/point_nums)))



def HETest(Hfile, pointfile, img1, img2):
    """
    根据匹配点计算单应性矩阵再计算精度
    利用两个单应性矩阵对图像的4个角点做变换，然后计算像素误差
    其实就是把计算匹配点的 MMA 变成了计算4个角点的 MMA
    :param Hfile: 单应矩阵的文件
    :param pointfile: 匹配点的文件
    :param img1, img2: 原图与目标图像
    :return:
    """
    H_ground = np.loadtxt(Hfile)
    matches = np.loadtxt(pointfile)
    point01 = matches[:, :2].reshape(-1, 1, 2)
    point02 = matches[:, 2: 4].reshape(-1, 1, 2)
    H_pred, _ = cv2.findHomography(point01, point02, cv2.RANSAC)
    print(H_pred)
    img3 = img2.copy()
    height, width = img1.shape
    centers = np.array([[width/4, height/4, 1],
                        [3*width/4, height/4, 1],
                        [3*width/4, 3*height/4, 1],
                        [width/4, 3*height/4, 1]])

    real_centers = np.dot(centers, np.transpose(H_ground))
    real_centers = real_centers[:, :2] / real_centers[:, 2:]
    print(real_centers)
    pred_centers = np.dot(centers, np.transpose(H_pred))
    pred_centers = pred_centers[:, :2] / pred_centers[:, 2:]
    # 下面方法与上面等价
    centers2 = np.array([[width/4, height/4],
                        [3*width/4, height/4],
                        [3*width/4, 3*height/4],
                        [width/4, 3*height/4]]).reshape(-1, 1, 2)
    # dst = cv2.perspectiveTransform(centers2, H_pred)
    # print(dst)
    mean_dist = np.mean(np.linalg.norm(real_centers - pred_centers, axis=1))
    print('mean distation :', mean_dist)
    img1 = cv2.polylines(img1, [np.int64(centers2)], True, 255, 3, cv2.LINE_AA)
    img2 = cv2.polylines(img2, [np.int64(real_centers)], True, 255, 3, cv2.LINE_AA)

    img3 = cv2.polylines(img3, [np.int64(pred_centers)], True, 255, 3, cv2.LINE_AA)
    newimg = np.zeros((height, width*3, 3), np.uint8)
    for i in range(3):
        newimg[:height, : width, i] = img1
        newimg[:height, width: width*2, i] = img2
        newimg[:height, width*2: width*3, i] = img3
    plt.imshow(newimg)
    plt.show()

if __name__ == '__main__':
    Hfile = 'data_point/2Dpoint/H_1_5'
    pointfile = 'data_point/2Dpoint/pointdata1_5_0.4.txt'
    pointfile2 = 'data_point/2Dpoint/pointdata1_5_0.7.txt'
    img1 = cv2.imread('img/1.jpg', 0)
    img2 = cv2.imread('img/5.jpg', 0)


    # print('近邻次邻的阈值为0.4：')
    # MMATest(Hfile, pointfile, 0.7)
    # print('近邻次邻的阈值为0.7：')
    # MMATest(Hfile, pointfile2, 0.7)
    HETest(Hfile, pointfile, img1, img2)