#!/usr/bin/python3
import os
import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image
from random import random
from collections import defaultdict


def save_mean_std_dev(mean, std_dev):
    with open('mean_std_dev.csv', 'w') as f:
        f.write('{},{}\n'.format(mean, std_dev))
    print('saved mean = {}, std_dev = {}'.format(mean, std_dev))

def load_mean_std_dev():
    with open('mean_std_dev.csv', 'r') as f:
        mean, std_dev = [float(x) for x in f.readlines()[0].strip().split(',')]
    print('loaded mean = {}, std_dev = {}'.format(mean, std_dev))
    return ((mean,), (std_dev,))


class TrainDataPreprocessor():
    VERSIONS = [
        ('main0', 1),
        ('main1', 1),
    ]

    def __init__(self, percent_training=0.8):
        print('preprocessing data...')
        x_total = []
        y_total = []

        counts = defaultdict(int)
        for version, multiplier in self.VERSIONS:
            LABELED_PATH = '/home/calvin/storage/cv-alarm-clock-data/{}/labeled'.format(version)
            FACE_PATH = '/home/calvin/storage/cv-alarm-clock-data/{}/face.csv'.format(version)
            NON_FACE_PATH = '/home/calvin/storage/cv-alarm-clock-data/{}/non-face.csv'.format(version)

            open_file = ''
            open_img = None
            with open(FACE_PATH, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    filename = parts[0]
                    x1, x2, y1, y2 = [int(val) for val in parts[1:]]
                    if open_file != filename:
                        open_img = cv2.imread(os.path.join(LABELED_PATH, filename), cv2.IMREAD_GRAYSCALE)
                        open_file = filename
                    if multiplier < 1:
                        if random() < multiplier:
                            counts[version + 'face'] += 1
                            x_total.append(open_img[y1:y2, x1:x2])
                            y_total.append(1)
                    else:
                        for _ in range(int(multiplier)):
                            counts[version + 'face'] += 1
                            x_total.append(open_img[y1:y2, x1:x2])
                            y_total.append(1)

            with open(NON_FACE_PATH, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    filename = parts[0]
                    x1, x2, y1, y2 = [int(val) for val in parts[1:]]
                    if open_file != filename:
                        open_img = cv2.imread(os.path.join(LABELED_PATH, filename), cv2.IMREAD_GRAYSCALE)
                        open_file = filename
                    if multiplier < 1:
                        if random() < multiplier:
                            counts[version + 'nonface'] += 1
                            x_total.append(open_img[y1:y2, x1:x2])
                            y_total.append(0)
                    else:
                        for _ in range(int(multiplier)):
                            counts[version + 'nonface'] += 1
                            x_total.append(open_img[y1:y2, x1:x2])
                            y_total.append(0)

        print(counts)

        n = len(x_total)
        x_total = np.array(x_total, dtype=np.uint8)
        # x_total = np.expand_dims(x_total, axis=4) # channels dimension (tensorflow requires this)
        y_total = np.array(y_total, dtype=int)

        save_mean_std_dev(np.average(x_total) / 255., np.std(x_total) / 255.)
        # print(np.average(x_total) / 255.)
        # print(np.std(x_total) / 255.)

        # split into training and validation sets
        indices = np.random.permutation(n)
        split = int(percent_training * n)
        train_idx, val_idx = indices[:split], indices[split:]
        self.x_train, self.x_val = x_total[train_idx,:], x_total[val_idx,:]
        self.y_train, self.y_val = y_total[train_idx], y_total[val_idx]

        print('done.')


class TestDataPreprocessor():
    LABELED_PATH = '/home/calvin/storage/cv-alarm-clock-data/test0/labeled'
    FACE_PATH = '/home/calvin/storage/cv-alarm-clock-data/test0/face.csv'
    NON_FACE_PATH = '/home/calvin/storage/cv-alarm-clock-data/test0/non-face.csv'

    def __init__(self):
        print('preprocessing data...')
        x_total = []
        y_total = []

        open_file = ''
        open_img = None
        with open(self.FACE_PATH, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                filename = parts[0]
                x1, x2, y1, y2 = [int(val) for val in parts[1:]]
                if open_file != filename:
                    open_img = cv2.imread(os.path.join(self.LABELED_PATH, filename), cv2.IMREAD_GRAYSCALE)
                    open_file = filename
                x_total.append(open_img[y1:y2, x1:x2])
                y_total.append(1)

        with open(self.NON_FACE_PATH, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                filename = parts[0]
                x1, x2, y1, y2 = [int(val) for val in parts[1:]]
                if open_file != filename:
                    open_img = cv2.imread(os.path.join(self.LABELED_PATH, filename), cv2.IMREAD_GRAYSCALE)
                    open_file = filename
                x_total.append(open_img[y1:y2, x1:x2])
                y_total.append(0)

        self.n = len(x_total)
        self.x_test = np.array(x_total, dtype=np.uint8)
        # x_total = np.expand_dims(x_total, axis=4) # channels dimension (tensorflow requires this)
        self.y_test = np.array(y_total, dtype=int)

        print('done.')


class TorchDataset(data.Dataset):
    def __init__(self, x, y, transform):
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.x[index], self.y[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, 'L')

        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.x)
