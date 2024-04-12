import os
import sys
import json
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from torchsummary import summary
from sklearn import metrics
import torchvision.models as models
import banhendataset as ds
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from PIL import Image
from PIL import ImageFile


class Config():
    def __init__(self,):
        os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
        self.model_name = 'resnet50'   # 'resnet50' 'resnet18'
        self.optimizer_select = "adam"                       # adam sgd
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_ids = [0, 1]
        self.train_batch_size = 16
        self.test_batch_size = 100
        self.max_train_epochs = 100
        # self.c_score_weight = [3, 4, 3, 5]
        self.initial_lr = 0.0001                            # 0.01 0.001 0.0001
        self.flag_if_poly_adjust_lr = False                  # True False
        self.num_score_tasks = 4        # 4个回归任务：sz hd xg rr
        self.num_classify_objects = 4   # 4个分类目标：0-瘢痕癌，1-瘢痕疙瘩，2-萎缩性瘢痕， 3-增生性瘢痕

        self.train_txt = "/media/D_4TB/SUGURS/Banhen_multi/train.txt"
        self.test_txt = "/media/D_4TB/SUGURS/Banhen_multi/test.txt"
        # self.data_transform = {
        #     "train": transforms.Compose([
        #         transforms.Resize(256),
        #         transforms.RandomResizedCrop(244, scale=(0.6, 1.0), ratio=(0.8, 1.0)),
        #         transforms.RandomHorizontalFlip(),
        #         # torchvision.transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0),
        #         # torchvision.transforms.ColorJitter(brightness=0, contrast=0.5, saturation=0, hue=0),
        #         transforms.transforms.ToTensor(),
        #         transforms.transforms.Normalize(mean=[0.4848, 0.4435, 0.4023], std=[0.2744, 0.2688, 0.2757])]),
        #     "test": transforms.Compose([
        #         transforms.Resize((224, 224)),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.4848, 0.4435, 0.4023], std=[0.2744, 0.2688, 0.2757])])
        # }

        self.data_transform = {
            "train": transforms.Compose([
                transforms.RandomResizedCrop([224, 224]),
                transforms.RandomHorizontalFlip(),
                transforms.transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            "test": transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        }

        self.data_num_workers = 8
        self.using_pretrained_weights = True
        # self.test_res_txt = '/media/E_4TB/WW/test_res.txt'
        # self.pretrained_model_path = "/media/E_4TB/WW/deep-learning-for-image-processing-master/pytorch_classification/Banhen_Scoring/resnet34-pre.pth"
        # self.pretrained_model_path = "/media/E_4TB/WW/code/banhen/Banhen_Scoring/resnet50-0676ba61.pth"
