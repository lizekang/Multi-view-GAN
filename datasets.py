import glob
import random
import os
import numpy as np
import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class MultiViewDataset(torch.utils.data.Dataset):  # 继承的torch.utils.data.Dataset
    def __init__(self, path, batch_size, train_step, transform=None):  # 初始化一些需要传入的参数
        f = open(path, 'r')
        self.img_class_dict = {}
        for line in f:
            words = line.split(' ')
            if int(words[1]) not in self.img_class_dict.keys():
                self.img_class_dict[int(words[1])] = [(words[0], int(words[-1]))]
            else:
                self.img_class_dict[int(words[1])].append((words[0], int(words[-1])))

        self.batch_size = batch_size
        self.train_step = train_step
        self.transform = transforms.Compose(transform)

    def __getitem__(self, index):
        class_index = random.choice(list(self.img_class_dict.keys()))

        img_list = random.sample(self.img_class_dict[class_index], 6)
        if transforms is not None:
            img1, img2, img3, img4, img5, real_img = [self.transform(Image.open(i[0]))[:3,:,:] for i in img_list]
        else:
            img1, img2, img3, img4, img5, real_img = [np.array(Image.open(i[0]))[:3,:,:] for i in img_list]
        label = img_list[-1][1]

        return img1, img2, img3, img4, img5, real_img, label

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return self.batch_size * self.train_step
