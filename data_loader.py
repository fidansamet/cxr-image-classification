import os
import cv2
import random
import numpy as np

SUBSET_DIR_NAMES = ['COVID', 'NORMAL', 'Viral Pneumonia']
RANDOM_SEED = 42


class DataLoader:
    def __init__(self, opt):
        """
        :param opt:
        """
        self.opt = opt
        self.train_samples, self.train_labels = [], []
        self.load_train_data()
        if opt.phase != 'train':  # test phase
            self.test_samples, self.test_labels = [], []
            self.load_test_data()

    def extract_feature(self, path, img_size=(32, 32)):
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        # TODO
        return cv2.resize(image, img_size).flatten()

    def load_train_data(self):
        dataset = []
        for subset_dir_name in SUBSET_DIR_NAMES:
            subset_img_names = sorted(os.listdir(self.opt.dataroot + '/train/' + subset_dir_name))
            for img_name in subset_img_names:
                sample = {
                    'sample': self.extract_feature(self.opt.dataroot + '/train/' + subset_dir_name + '/' + img_name),
                    'label': SUBSET_DIR_NAMES.index(subset_dir_name)}
                dataset.append(sample)
            random.Random(RANDOM_SEED).shuffle(dataset)

        for data in dataset:
            self.train_samples.append(data['sample'])
            self.train_labels.append(data['label'])

    def load_test_data(self):
        img_names = sorted(os.listdir(self.opt.dataroot + '/test'))
        for img_name in img_names:
            self.test_samples.append(self.extract_feature(self.opt.dataroot + '/test/' + img_name))
            # find label of the test image
            for i in range(len(SUBSET_DIR_NAMES)):
                if SUBSET_DIR_NAMES[i] in img_name:
                    self.test_labels.append(i)
                    break

    def split_cross_valid(self):
        # get split data for k-fold cross validation
        # TODO type
        return np.array_split(np.array(self.train_samples, dtype=np.int32), self.opt.fold_num), \
               np.array_split(np.array(self.train_labels, dtype=np.int32), self.opt.fold_num)

    def get_train_test_data(self):
        # get all data for test phase
        # TODO type
        return np.array(self.train_samples, dtype=np.int32), np.array(self.train_labels, dtype=np.int32), \
               np.array(self.test_samples, dtype=np.int32), np.array(self.test_labels, dtype=np.int32)
