import os
import cv2
import random
import numpy as np
from PIL import Image
import filters

SUBSET_DIR_NAMES = ['COVID', 'NORMAL', 'Viral Pneumonia']
RANDOM_SEED = 42


class DataLoader:
    def __init__(self, opt):
        """
        :param opt:
        """
        self.opt = opt
        self.train_samples, self.train_labels = [], []
        self.get_features("train")
        self.load_train_data()
        if opt.phase != "train":  # test phase
            self.test_samples, self.test_labels = [], []
            self.get_features("test")
            self.load_test_data()
        print("Data loaded")

    def get_features(self, folder_name):
        # get saved features or create files to save features
        if self.opt.canny:
            self.canny_path = self.opt.features_path + "/" + folder_name + "/canny.txt"
            if os.path.exists(self.canny_path):
                self.canny_read = True
                self.canny_iter = iter(np.loadtxt(self.canny_path, dtype=int))
            else:
                self.canny_read = False
                open(self.canny_path, 'wb')
                self.canny_features = []

        if self.opt.gabor:
            self.gabor_path = self.opt.features_path + "/" + folder_name + "/gabor.txt"
            if os.path.exists(self.gabor_path):
                self.gabor_read = True
                self.gabor_iter = iter(np.loadtxt(self.gabor_path, dtype=int))
            else:
                self.gabor_read = False
                open(self.gabor_path, 'wb')
                self.gabor_features = []

        if self.opt.sift:
            self.sift_path = self.opt.features_path + "/" + folder_name + "/sift.txt"
            if os.path.exists(self.sift_path):
                # TODO type
                self.sift_read = True
                self.sift_iter = iter(np.loadtxt(self.sift_path, dtype=int))
            else:
                self.sift_read = False
                open(self.sift_path, 'wb')
                self.sift_features = []

        if self.opt.vgg19:
            self.vgg19_path = self.opt.features_path + "/" + folder_name + "/vgg19.txt"
            if os.path.exists(self.vgg19_path):
                self.vgg19_read = True
                self.vgg19_iter = iter(np.loadtxt(self.vgg19_path, dtype=np.float32))
            else:
                self.vgg19 = filters.VGG19()
                self.vgg19_read = False
                open(self.vgg19_path, 'wb')
                self.vgg19_features = []

    def extract_features(self, path, img_size=(64, 64), tiny_img_size=(32, 32)):
        print("Extracting features from " + path)
        image = cv2.imread(path)
        image_features = np.array([], dtype=np.float32)
        if self.opt.canny:
            if self.canny_read:
                image_features = np.concatenate((image_features, next(self.canny_iter)), axis=0)
            else:
                canny = filters.canny_edge(image)
                canny.resize(img_size, refcheck=False)
                canny_flatten = canny.flatten()
                self.canny_features.append(canny_flatten)
                image_features = np.concatenate((image_features, canny_flatten), axis=0)

        if self.opt.gabor:
            if self.gabor_read:
                image_features = np.concatenate((image_features, next(self.gabor_iter)), axis=0)
            else:
                gabor = filters.gabor_process(image)
                gabor.resize(img_size)
                gabor_flatten = gabor.flatten()
                self.gabor_features.append(gabor_flatten)
                image_features = np.concatenate((image_features, gabor_flatten), axis=0)

        # TODO: SIFT or HoG

        if self.opt.vgg19:
            if self.vgg19_read:
                image_features = np.concatenate((image_features, next(self.vgg19_iter)), axis=0)
            else:
                pil_image = Image.open(path).convert("RGB")
                vgg19_extracted = self.vgg19.forward(pil_image)
                vgg19_extracted = vgg19_extracted.numpy()[0]
                self.vgg19_features.append(vgg19_extracted)
                image_features = np.concatenate((image_features, vgg19_extracted), axis=0)

        if self.opt.tiny_img:
            tiny_flatten = cv2.cvtColor(cv2.resize(image, tiny_img_size), cv2.COLOR_BGR2GRAY).flatten()
            image_features = np.concatenate((image_features, tiny_flatten), axis=0)

        # if no image feature specified, extract 64x64 feature
        if image_features.size == 0:
            small_flatten = cv2.cvtColor(cv2.resize(image, img_size), cv2.COLOR_BGR2GRAY).flatten()
            image_features = np.concatenate((image_features, small_flatten), axis=0)

        return image_features

    def save_features(self):
        # save features for reuse
        if self.opt.canny:
            if not self.canny_read:
                np.savetxt(self.canny_path, self.canny_features, fmt='%d')

        if self.opt.gabor:
            if not self.gabor_read:
                np.savetxt(self.gabor_path, self.gabor_features, fmt='%d')

        if self.opt.sift:
            if not self.sift_read:
                # TODO: type float
                np.savetxt(self.sift_path, self.sift_features, fmt='%d')

        if self.opt.vgg19:
            if not self.vgg19_read:
                np.savetxt(self.vgg19_path, self.vgg19_features, fmt='%f')

    def load_train_data(self):
        dataset = []
        for subset_dir_name in SUBSET_DIR_NAMES:
            subset_img_names = sorted(os.listdir(self.opt.dataroot + '/train/' + subset_dir_name))
            for img_name in subset_img_names:
                sample = {
                    'sample': self.extract_features(self.opt.dataroot + '/train/' + subset_dir_name + '/' + img_name),
                    'label': SUBSET_DIR_NAMES.index(subset_dir_name)}
                dataset.append(sample)
            random.Random(RANDOM_SEED).shuffle(dataset)
        self.save_features()

        for data in dataset:
            self.train_samples.append(data['sample'])
            self.train_labels.append(data['label'])

    def load_test_data(self):
        img_names = sorted(os.listdir(self.opt.dataroot + '/test'))
        for img_name in img_names:
            self.test_samples.append(self.extract_features(self.opt.dataroot + '/test/' + img_name))
            # find label of the test image
            for i in range(len(SUBSET_DIR_NAMES)):
                if SUBSET_DIR_NAMES[i] in img_name:
                    self.test_labels.append(i)
                    break
        self.save_features()

    def split_cross_valid(self):
        # get split data for k-fold cross validation
        return np.array_split(np.array(self.train_samples, dtype=np.float32), self.opt.fold_num), \
               np.array_split(np.array(self.train_labels, dtype=np.int32), self.opt.fold_num)

    def get_train_test_data(self):
        # get all data for test phase
        return np.array(self.train_samples, dtype=np.float32), np.array(self.train_labels, dtype=np.int32), \
               np.array(self.test_samples, dtype=np.float32), np.array(self.test_labels, dtype=np.int32)
