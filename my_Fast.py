import cv2
import numpy as np
import matplotlib.pyplot as plt

def circle(row, col):
    point1 = (row-3, col)
    point2 = (row-3, col+1)
    point3 = (row-2, col+2)
    point4 = (row-1, col+3)
    point5 = (row, col+3)
    point6 = (row+1, col+3)
    point7 = (row+2, col+2)
    point8 = (row+3, col+1)
    point9 = (row+3, col)
    point10 = (row+3, col-1)
    point11 = (row+2, col-2)
    point12 = (row+1, col-3)
    point13 = (row, col-3)
    point14 = (row-1, col-3)
    point15 = (row-2, col-2)
    point16 = (row-3, col-1)
    return [point1, point2, point3, point4, point5, point6, point7, point8, point9, point10, point11, point12, point13, point14, point15, point16];

def is_corner(img, row, col, ROI, threshold):
    diff = np.zeros(17)
    count = 0  # 统计符合阀值的点个数
    intensity = int(img[row][col])
    score = 0
    diff[1] = abs(img[ROI[0][0]][ROI[0][1]] - intensity)
    diff[9] = abs(img[ROI[8][0]][ROI[8][1]] - intensity)
    count = int(diff[1] > threshold) + int(diff[9] > threshold)
    if count != 2:
        return None

    diff[5] = abs(img[ROI[4][0]][ROI[4][1]] - intensity)
    diff[13] = abs(img[ROI[12][0]][ROI[12][1]] - intensity)
    count = count + int(diff[5] > threshold) + int(diff[13] > threshold)
    if count < 3:
        return None

    diff[2] = abs(img[ROI[1][0]][ROI[1][1]] - intensity)
    diff[3] = abs(img[ROI[2][0]][ROI[2][1]] - intensity)
    diff[4] = abs(img[ROI[3][0]][ROI[3][1]] - intensity)

    diff[6] = abs(img[ROI[5][0]][ROI[5][1]] - intensity)
    diff[7] = abs(img[ROI[6][0]][ROI[6][1]] - intensity)
    diff[8] = abs(img[ROI[7][0]][ROI[7][1]] - intensity)

    diff[10] = abs(img[ROI[9][0]][ROI[9][1]] - intensity)
    diff[11] = abs(img[ROI[10][0]][ROI[10][1]] - intensity)
    diff[12] = abs(img[ROI[11][0]][ROI[11][1]] - intensity)

    diff[14] = abs(img[ROI[13][0]][ROI[13][1]] - intensity)
    diff[15] = abs(img[ROI[14][0]][ROI[14][1]] - intensity)
    diff[16] = abs(img[ROI[15][0]][ROI[15][1]] - intensity)

    count = count + int(diff[2] > threshold) + int(diff[3] > threshold) + int(diff[4] > threshold) \
                    + int(diff[6] > threshold) + int(diff[7] > threshold) + int(diff[8] > threshold) \
                    + int(diff[10] > threshold) + int(diff[11] > threshold) + int(diff[12] > threshold) \
                    + int(diff[14] > threshold) + int(diff[15] > threshold) + int(diff[16] > threshold) \

    if count >= 12:
        return True

def areAdjacent(point1, point2):
    row1, col1 = point1
    row2, col2 = point2
    xDist = row1 - row2
    yDist = col1 - col2
    return (xDist ** 2 + yDist ** 2) ** 0.5 <= 4

def calculateScore(img, point, ROI):
    col, row = point
    diff = np.zeros(17)
    intensity = int(image[row][col])
    score = 0.0
    diff[1] = abs(img[ROI[0][0]][ROI[0][1]] - intensity)
    diff[2] = abs(img[ROI[1][0]][ROI[1][1]] - intensity)
    diff[3] = abs(img[ROI[2][0]][ROI[2][1]] - intensity)
    diff[4] = abs(img[ROI[3][0]][ROI[3][1]] - intensity)
    diff[5] = abs(img[ROI[4][0]][ROI[4][1]] - intensity)
    diff[6] = abs(img[ROI[5][0]][ROI[5][1]] - intensity)
    diff[7] = abs(img[ROI[6][0]][ROI[6][1]] - intensity)
    diff[8] = abs(img[ROI[7][0]][ROI[7][1]] - intensity)
    diff[9] = abs(img[ROI[8][0]][ROI[8][1]] - intensity)
    diff[10] = abs(img[ROI[9][0]][ROI[9][1]] - intensity)
    diff[11] = abs(img[ROI[10][0]][ROI[10][1]] - intensity)
    diff[12] = abs(img[ROI[11][0]][ROI[11][1]] - intensity)
    diff[13] = abs(img[ROI[12][0]][ROI[12][1]] - intensity)
    diff[14] = abs(img[ROI[13][0]][ROI[13][1]] - intensity)
    diff[15] = abs(img[ROI[14][0]][ROI[14][1]] - intensity)
    diff[16] = abs(img[ROI[15][0]][ROI[15][1]] - intensity)
    for i in range(0, 17):
        score = score + diff[i]
    return score

def suppress(image, corners, ROI, threshold):
    i = 1
    while i < len(corners):
        currPoint = corners[i]
        prevPoint = corners[i-1]
        if areAdjacent(prevPoint, currPoint):
            currScore = calculateScore(image, currPoint, ROI)
            prevScore = calculateScore(image, currPoint, ROI)
            if (currScore > prevScore):
                del(corners[i-1])
            else:
                del(corners[i])
        else:
            i += 1
            continue
    return

def detect(image, threshold):
    corners = []
    rows, cols = image.shape
    for row in range(3, rows - 3):
        for col in range(3, cols - 3):
            ROI = circle(row, col)
            if is_corner(image, row, col, ROI, threshold):
                corners.append((col, row))
        if len(corners) > 2:
            suppress(image, corners, ROI, threshold)
    return corners

if __name__ == '__main__':
    image = cv2.imread('towel.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.medianBlur(image, 5)
    corners = detect(image, threshold=50)
    print(len(corners))
    implot = plt.imshow(image, cmap='gray')
    for point in corners:
        plt.scatter(point[0], point[1], s=10)
    plt.show()
