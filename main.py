import numpy as np
from options import Options
from data_loader import DataLoader
from knn import NearestNeighbors
import time
import csv

CATEGORIES = ['COVID', 'NORMAL', 'VIRAL']


def classify_images(train_samples, train_labels, test_samples, test_labels, write_file=False):
    knn = NearestNeighbors(opt, train_samples, train_labels)
    correct_classified = 0
    row_list = []

    for test_sample, test_label in zip(test_samples, test_labels):
        pred = knn.predict(test_sample)
        if pred == test_label:
            correct_classified += 1
        if write_file is True:
            row_list.append([len(row_list) + 1, CATEGORIES[pred]])

    acc = 100 * (correct_classified / len(test_samples))
    print("%d/%d samples are correctly classified - Accuracy: %0.2f" % (correct_classified, len(test_samples), acc))
    return acc, row_list


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

        acc, _ = classify_images(train_samples, train_labels, test_samples, test_labels)
        accuracies.append(acc)

    print("Mean accuracy: %0.2f - Computation time: %0.2f second(s)" %
          (sum(accuracies) / len(accuracies), time.clock() - time_start))
    print("-------------------")


def test(train_samples, train_labels, test_samples, test_labels):
    acc, row_list = classify_images(train_samples, train_labels, test_samples, test_labels, write_file=True)
    with open('predictions.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Id", "Category"])
        writer.writerows(row_list)


if __name__ == '__main__':
    opt = Options().parse()
    data_loader = DataLoader(opt)
    if opt.phase == 'train':
        img_folds, gt_folds = data_loader.split_cross_valid()
        train(img_folds, gt_folds)
    else:  # test phase
        train_imgs, train_gts, test_imgs, test_gts = data_loader.get_train_test_data()
        test(train_imgs, train_gts, test_imgs, test_gts)
