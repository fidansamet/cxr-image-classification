import numpy as np


class NearestNeighbors:
    def __init__(self, opt, train_samples, ground_truths):
        """
        :param train_samples: 2d train samples matrix of size NxD
        :param ground_truths: 1d ground truth array of size N
        """
        self.opt = opt
        self.train_samples = train_samples
        self.ground_truths = ground_truths

    def calculate_weights(self, dists):
        dists[dists == 0.0] = 0.00001  # division by 0 handle for floats
        dists[dists == 0] = 1  # division by 0 handle for integers
        weights = 1.0 / dists
        return weights / np.sum(weights)

    def calculate_dist(self, x, y):
        if self.opt.dist_measure == 'euclidean':
            return np.sqrt(np.sum((x - y) ** 2))
        elif self.opt.dist_measure == 'manhattan':
            return np.sum(np.abs(x - y))
        elif self.opt.dist_measure == 'hamming':
            return np.sum(np.logical_xor(x, y))
        else:  # minkowski distance
            # return np.sum(np.abs(x - y)**5)**(1/5)
            return np.abs(np.sum(np.abs(x - y) ** 3)) ** (1 / 3)

    def find_neighbors(self, x, y):
        return np.asarray([self.calculate_dist(i, y) for i in x])

    def predict(self, test_sample):
        """
        :param test_sample: test sample to classify
        :return: test sample class prediction
        """
        dists = self.find_neighbors(self.train_samples, test_sample)
        min_dist_indices = dists.argsort()[:self.opt.neighbor_num]  # get smallest k distance indices
        y_preds = self.ground_truths[min_dist_indices]  # get ground truths of closest samples

        if self.opt.weighted_knn:
            votes = np.zeros(3, dtype=np.float32)  # votes for 3 classes
            nearest_neighbors = dists[min_dist_indices]
            weights = self.calculate_weights(nearest_neighbors)
            for i, j in zip(weights, y_preds):  # predicted class and its weight
                votes[j] += i * 1.0
            return np.where(votes == np.amax(votes))[0][0]  # return class with highest vote

        else:
            return np.argmax(np.bincount(y_preds))  # return most frequent ground truth(closest neighbor on equality)
