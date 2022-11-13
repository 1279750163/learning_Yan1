import numpy as np
import matplotlib.pyplot as plt

class K_Means():
    def __init__(self, k, tolerance, max_iter):
        self.k = k
        self.tolerance_ = tolerance
        self.max_iter = max_iter

    def fit(self, data):
        self.centers_ = {}
        for i in range(self.k):
            self.centers_[i] = data[i]

        for iter in range(self.max_iter):
            self.clf_ = {}
            for i in range(self.k):
                self.clf_[i] = []

            for feature in data:
                distances = []
                for i in self.centers_:
                    # 欧式距离
                    # np.sqrt(np.sum((feature - self.centers_[i]) ** 2 ))
                    distances.append(np.linalg.norm(feature - self.centers_[i]))
                classification = distances.index(min(distances))
                self.clf_[classification].append(feature)

            prev_centers = dict(self.centers_)
            for c in self.clf_:
                self.centers_[c] = np.average(self.clf_[c], axis=0)

            optimized = True
            for center in self.centers_:
                org_centers = prev_centers[center]
                cur_centers = self.centers_[center]
                if np.sum((cur_centers - org_centers) / org_centers * 100.0) > self.tolerance_:
                    optimized = False

            if optimized:
                break

    def predict(self, testData):
        distances = [np.linalg.norm(testData - self.centers_[center]) for center in self.centers_]
        index = distances.index(min(distances))
        return index


if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    k_means = K_Means(k=2, tolerance=0.01, max_iter=200)
    k_means.fit(x)
    print(k_means.centers_)
    colors = ['r', 'g', 'b']
    for i in k_means.centers_:
        plt.scatter(k_means.centers_[i][0], k_means.centers_[i][1], marker='*', s=150)

    for i in k_means.clf_:
        for point in k_means.clf_[i]:
            plt.scatter(point[0], point[1], c=colors[i])

    predict = [[2, 1], [6, 9]]
    for point in predict:
        label = k_means.predict(point)
        plt.scatter(point[0], point[1], marker='x', c=colors[label])

    plt.show()
