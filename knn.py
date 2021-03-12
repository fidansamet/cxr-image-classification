import numpy as np


class NearestNeighbors:
    def __init__(self, train_samples, ground_truths, k, dist_metric='euclidean'):
        """
        :param train_samples: 2d train samples matrix of size NxD
        :param ground_truths: 1d ground truth array of size N
        :param k: number of neighbors to consider
        :return:
        """
        self.train_samples = train_samples
        self.ground_truths = ground_truths
        self.k = k
        self.dist_metric = dist_metric

    def calculate_dist(self, x, y):
        if self.dist_metric == 'euclidean':
            return np.sqrt(np.sum(np.diff(x, y) ** 2))
        else:  # manhattan dist
            return np.sum(np.abs(np.diff(x, y)))
        # else:   # TODO: hamming dist

    def find_neighbors(self, x, y):
        return np.asarray([self.calculate_dist(i, y) for i in x])

    def predict(self, test_sample):
        """
        :param test_sample: test sample to classify
        :return:
        """
        dists = self.find_neighbors(self.train_samples, test_sample)
        min_dist_indices = dists.argsort()[:self.k]  # get smallest k distance indices
        y_preds = self.ground_truths[min_dist_indices]  # get ground truths of closest samples

        return np.argmax(np.bincount(y_preds))  # return most frequent ground truth
