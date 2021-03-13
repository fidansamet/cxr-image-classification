import os
import cv2
import random
import numpy as np


class DataLoader:
    def __init__(self, opt):
        """
        :param opt:
        """
        self.opt = opt
        self.subset_dir_names = ['COVID', 'NORMAL', 'Viral Pneumonia']
        self.samples = []
        self.labels = []
        self.RANDOM_SEED = 42
        self.load_data()

    def extract_feature(self, path, img_size=(32, 32)):
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        # TODO
        return cv2.resize(image, img_size).flatten()

    def load_data(self):
        dataset = []
        for subset_dir_name in self.subset_dir_names:
            subset_img_names = sorted(os.listdir(self.opt.dataroot + '/' + subset_dir_name))
            for img_name in subset_img_names:
                sample = {'sample': self.extract_feature(self.opt.dataroot + '/' + subset_dir_name + '/' + img_name),
                          'label': self.subset_dir_names.index(subset_dir_name)}
                dataset.append(sample)
            random.Random(self.RANDOM_SEED).shuffle(dataset)

        for data in dataset:
            self.samples.append(data['sample'])
            self.labels.append(data['label'])

    # def __getitem__(self, idx):
    #     return self.dataset[idx]

    def split_cross_valid(self):
        # folds = []
        # fold_size = int(len(self.dataset) / self.opt.fold_num)
        # for i in range(self.opt.fold_num):
        #     cur_fold = []
        #     for j in range(fold_size):
        #         cur_fold.append()
        #
        #     folds.append(cur_fold)

        # TODO type
        return np.array_split(np.array(self.samples, dtype=np.int32), self.opt.fold_num), \
               np.array_split(np.array(self.labels, dtype=np.int32), self.opt.fold_num)
