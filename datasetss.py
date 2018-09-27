# coding: utf-8
import glob
import random
import os
import numpy as np
import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

vlist = ['-50','0','50']
hlist = ['0','120','240']

class MultiViewDataset(torch.utils.data.Dataset):  # 继承的torch.utils.data.Dataset
    def __init__(self, path, batch_size, train_step, transform=None):  # 初始化一些需要传入的参数
        self.img_input_dict = {}
        self.img_real_image = {}
        f = open(path, 'r')
        for line in f:
            words = line[:-1].split(' ')
            if words[2] in hlist:
                if words[3] in vlist:
                    if int(words[1]) not in self.img_input_dict.keys():
                        self.img_input_dict[int(words[1])] = [words[0]]
                    else:
                        self.img_input_dict[int(words[1])].append(words[0])

            if int(words[1]) not in self.img_real_image.keys():
                self.img_real_image[int(words[1])] = [(words[0], int(words[-2]), int(words[-1]))]
            else:
                self.img_real_image[int(words[1])].append((words[0],int(words[-2]), int(words[-1])))
        self.batch_size = batch_size
        self.train_step = train_step
        self.transform = transforms.Compose(transform)

    def __getitem__(self, index):
        class_index = random.choice(list(self.img_input_dict.keys()))
        img_list = sorted(self.img_input_dict[class_index][:]) + [random.choice(self.img_real_image[class_index])]
        if transforms is not None:
            input_img = [self.transform(Image.open(i))[:3, :, :] for i in img_list[:9]]
            input_img = torch.cat(input_img, 0)
            real_img = self.transform(Image.open(img_list[9][0]))[:3, :, :]
        label = np.asarray([img_list[9][1], img_list[9][2]], dtype=np.float64)
        return input_img, real_img, label

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return self.batch_size * self.train_step

