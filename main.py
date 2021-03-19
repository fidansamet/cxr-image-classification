import numpy as np
from options import Options
from data_loader import DataLoader
from knn import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import seaborn as sn
import time
import csv

CATEGORIES = ['COVID', 'NORMAL', 'VIRAL']


def classify_images(train_samples, train_labels, test_samples, test_labels, write_file=False):
    knn = NearestNeighbors(opt, train_samples, train_labels)
    correct_classified = 0
    row_list, trues, preds = [], [], []

    for test_sample, test_label in zip(test_samples, test_labels):
        pred = knn.predict(test_sample)
        trues.append(test_label)
        preds.append(pred)
        if pred == test_label:
            correct_classified += 1
        if write_file is True:
            row_list.append([len(row_list) + 1, CATEGORIES[pred]])

    acc = 100 * (correct_classified / len(test_samples))
    print("%d/%d samples are correctly classified - Accuracy: %0.2f" % (correct_classified, len(test_samples), acc))
    return acc, row_list, trues, preds


def train(sample_folds, label_folds):
    print("Train started")
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

        acc, _, _, _ = classify_images(train_samples, train_labels, test_samples, test_labels)
        accuracies.append(acc)

    mean_acc = sum(accuracies) / len(accuracies)
    comp_time = time.clock() - time_start
    print("Mean accuracy: %0.2f - Computation time: %0.2f second(s)" % (mean_acc, comp_time))
    print("-------------------")
    return mean_acc, comp_time


def test(train_samples, train_labels, test_samples, test_labels):
    print("Test started")
    time_start = time.clock()
    acc, row_list, trues, preds = classify_images(train_samples, train_labels, test_samples, test_labels,
                                                  write_file=True)
    print("Computation time: %0.2f second(s)" % (time.clock() - time_start))
    print("-------------------")
    plot_conf_matrix(trues, preds)

    with open('predictions.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Id", "Category"])
        writer.writerows(row_list)


def plot_conf_matrix(true, pred):
    score = metrics.accuracy_score(true, pred)
    # cls_report = metrics.classification_report(ground_truth, test_result)
    conf_mat = metrics.confusion_matrix(true, pred)

    print('Accuracy: {:.3f}'.format(score))
    # print(cls_report)
    print(conf_mat)

    df_cm = pd.DataFrame(conf_mat, CATEGORIES, CATEGORIES)
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt='d', cmap=plt.get_cmap('jet'))  # font size
    plt.title("Confusion Matrix")
    plt.show()


def plot_graph(title, x, y, y_label):
    plt.title(title)
    plt.plot(x, y['euclidean'])
    plt.plot(x, y['manhattan'])
    plt.plot(x, y['hamming'])
    plt.plot(x, y['minkowski'])

    plt.legend(['Euclidean', 'Manhattan', 'Hamming', 'Minkowski'])

    plt.xticks(np.arange(1, 11, 1))
    plt.xlabel('Neighbor Number')
    plt.ylabel(y_label)
    plt.show()


def experiment(opt, img_folds, gt_folds, temp):
    x = []
    y_acc = {'euclidean': [], 'manhattan': [], 'hamming': [], 'minkowski': []}
    y_time = {'euclidean': [], 'manhattan': [], 'hamming': [], 'minkowski': []}
    dist_metrics = ['euclidean', 'manhattan', 'hamming', 'minkowski']

    for i in range(10):
        print("Starting k = %d" % (i + 1))
        opt.neighbor_num = i + 1
        x.append(i + 1)
        for j in dist_metrics:
            print("Starting " + j)
            opt.dist_measure = j
            acc, t = train(img_folds, gt_folds)
            y_acc[j].append(acc)
            y_time[j].append(t)

    f = open(temp + "_%d-fold.txt" % opt.fold_num, "a")
    f.write("Neighbors\n")
    f.write(str(x) + "\n")
    f.write("Accuracy\n")
    f.write(str(y_acc) + "\n")
    f.write("Time\n")
    f.write(str(y_time) + "\n")
    f.close()

    plot_graph("Accuracies for Different Neighbor Numbers (%d-fold)" % opt.fold_num, x, y_acc,
               "Classification Accuracy")
    plot_graph("Computation Times for Different Neighbor Numbers (%d-fold)" % opt.fold_num, x, y_time,
               "Computation Time (sec.)")


if __name__ == '__main__':
    opt = Options().parse()
    data_loader = DataLoader(opt)
    if opt.phase == 'train':
        for i in range(5, 7):
            opt.fold_num = i
            img_folds, gt_folds = data_loader.split_cross_valid()
            experiment(opt, img_folds, gt_folds, 'wknn_vgg19_hog')

        # train(img_folds, gt_folds)
        # experiment(opt, img_folds, gt_folds)

    # else:  # test phase
    # train_imgs, train_gts, test_imgs, test_gts = data_loader.get_train_test_data()
    # test(train_imgs, train_gts, test_imgs, test_gts)
