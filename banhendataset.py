from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import PIL.ImageOps
import pandas as pd
import os
import torchvision.transforms as transforms
import torch


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomRotation(40),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'val': transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'test': transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
}


class Dataset(Dataset):
    def __init__(self, datatxt, transform=None, should_invert=True):
        fh = open(datatxt, 'r')
        imgs = []
        scores = []
        cls = []
        for line in fh:
            line = line.rstrip()
            words = line.split(' ')
            imgs.append("/media/E_4TB/WW/dataset/AAA【已整理数据】瘢痕/【评分用】瘢痕/ScoreDataset/"+words[0]+'.jpg')
            scores.append([float(words[1]), float(words[2]), float(words[3]), float(words[4])])
            if words[0][0:3] == "114":
                cls.append(0)
            elif words[0][0:3] == "943":
                cls.append(1)
            elif words[0][0:3] == "650":
                cls.append(1)
            elif words[0][0:3] == "235":
                cls.append(2)
            elif words[0][0:3] == "208":
                cls.append(3)
        self.imgs = imgs
        self.scores = scores
        self.cls = cls
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB')
        # scores = self.scores[index]
        scores = torch.tensor(self.scores[index], dtype=torch.float32)
        cls = self.cls[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, scores, cls

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    from tqdm import tqdm
    import torch
    import sys

    train_dataset = Dataset(datatxt='/media/E_4TB/WW/train.txt', transform=data_transforms['train'])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=16, shuffle=False,
                                                  num_workers=2)
    train_bar = tqdm(train_loader, file=sys.stdout)
    for step, data in enumerate(train_bar):
        img, scores, cls = data
        print(img, scores, cls)
