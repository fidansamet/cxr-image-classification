import numpy as np
from options import Options
from data_loader import DataLoader
from knn import NearestNeighbors
import time


def train(sample_folds, label_folds):
    time_start = time.clock()
    accuracies = []
    for i in range(len(sample_folds)):
        # create train samples by removing current sample fold from sample folds
        train_samples = list(sample_folds)
        del train_samples[i]
        # concat train sample folds
        train_samples = np.concatenate(train_samples)

        # create train labels by removing current label fold from label folds
        train_labels = list(label_folds)
        del train_labels[i]
        # concat train label folds
        train_labels = np.concatenate(train_labels)

        # create test samples and labels from current sample and label folds
        test_samples = list(sample_folds[i])
        test_labels = list(label_folds[i])

        knn = NearestNeighbors(opt, train_samples, train_labels)
        correct_classified = 0

        for test_sample, test_label in zip(test_samples, test_labels):
            pred = knn.predict(test_sample)
            if pred == test_label:
                correct_classified += 1

        accuracies.append(100 * (correct_classified / len(test_samples)))
        print("End of fold %d - %d/%d samples are correctly classified - Accuracy: %0.2f" %
              (i, correct_classified, len(test_samples), accuracies[-1]))

    print("Mean accuracy: %0.2f - Computation time: %0.2f second(s)" %
          (sum(accuracies) / len(accuracies), time.clock() - time_start))
    print("-------------------")


if __name__ == '__main__':
    opt = Options().parse()
    data_loader = DataLoader(opt)
    sample_folds, label_folds = data_loader.split_cross_valid()
    train(sample_folds, label_folds)
