import os
import os.path
import errno
import numpy as np
import sys
import torch.utils.data as data
import csv
import cv2
import random
import deepdish as dd

from PIL import Image


class camel(data.Dataset):
    def __init__(self, transform=None, mode='train'):
        super(camel, self).__init__()
        self.transform = transform
        self.mode = mode

        self.data = []
        self.labels = []

        if self.mode == 'train':
            csv_file = open('train_csv.csv', 'r', encoding='utf-8')
            csv_reader = csv.reader(csv_file)
            self.img_name_list1 = []
            for img, label in csv_reader:
                array = cv2.imread(img, cv2.IMREAD_COLOR)
                self.data.append(array)
                self.labels.append(label)
                self.img_name_list1.append(img)
        elif self.mode == 'test':
            csv_file = open('test_csv.csv', 'r', encoding='utf-8')
            csv_reader = csv.reader(csv_file)
            self.img_name_list2 = []
            for img, label in csv_reader:
                array = cv2.imread(img, cv2.IMREAD_COLOR)
                self.data.append(array)
                self.labels.append(label)
                self.img_name_list2.append(img)
        else:
            csv_file = open('mining_csv.csv', 'r', encoding='utf-8')
            csv_reader = csv.reader(csv_file)
            self.img_name_list3 = []
            for img, label in csv_reader:
                array = cv2.imread(img, cv2.IMREAD_COLOR)
                self.data.append(array)
                self.labels.append(label)
                self.img_name_list3.append(img)
                self.data.append(array)
                self.labels.append(label)
                self.img_name_list3.append(img)
            csv_file2 = open('train_csv.csv', 'r', encoding='utf-8')
            csv_reader2 = csv.reader(csv_file2)
            for img, label in csv_reader2:
                if random.randint(0, 100) == 50:
                    array = cv2.imread(img, cv2.IMREAD_COLOR)
                    self.data.append(array)
                    self.labels.append(label)
                    self.img_name_list3.append(img)

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)

        if self.mode == 'train':
            target, filename = self.labels[index], self.img_name_list1[index]
        elif self.mode == 'test':
            target, filename = self.labels[index], self.img_name_list2[index]
        else:
            target, filename = self.labels[index], self.img_name_list3[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target, filename

    def __len__(self):
        return len(self.data)


def get_dataset(train_transform, test_transform, mining_mode=False):
    if mining_mode:
        mining_dataset = camel(transform=train_transform, mode='mining')
        return mining_dataset
    else:
        train_dataset = camel(transform=train_transform, mode='train')
        test_dataset = camel(transform=test_transform, mode='test')
        return train_dataset, test_dataset