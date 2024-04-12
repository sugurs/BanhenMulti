import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from torchsummary import summary
from sklearn import metrics
import torchvision.models as models
import banhendataset as ds
# import test_dataset as tds
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from PIL import Image
from PIL import ImageFile
from config import Config
import timm
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# import PIL.ImageOps
# from tqdm import tqdm


# dingyi config
cfg = Config()


def adjust_lr_poly(optimizer, epoch, max_epochs, base_lr=0.1, power=0.9):
    lr = base_lr * (1 - epoch / max_epochs) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 112 * 112, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv(x)
        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ScoringModel(nn.Module):
    def __init__(self, num_classes=4):
        super(ScoringModel, self).__init__()
        self.backbone = timm.create_model(cfg.model_name, pretrained=True, features_only=True, out_indices=[4])

        for param in self.backbone.parameters():
            param.requires_grad = True

        if cfg.model_name == 'resnet50':
            self.mlp = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, num_classes)
            )
        elif cfg.model_name == 'resnet18':
            self.mlp = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        features = self.backbone(x)[0]
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        output = self.mlp(features)
        return output


def train_test():
    train_dataset = ds.Dataset(datatxt=cfg.train_txt, transform=cfg.data_transform['train'])
    test_dataset = ds.Dataset(datatxt=cfg.test_txt, transform=cfg.data_transform['test'])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True,
                                                num_workers=cfg.data_num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.test_batch_size, shuffle=False,
                                                num_workers=cfg.data_num_workers)

    print("using {} images for training, {} images for validation.".format(len(train_dataset),
                                                                           len(test_dataset)))

    # net = timm.create_model(cfg.model_name, pretrained=True, num_classes=cfg.num_score_tasks)
    net = ScoringModel(num_classes=cfg.num_score_tasks)
    # net = SimpleNet()

    if len(cfg.device_ids) > 1:
        net = torch.nn.DataParallel(net, device_ids=cfg.device_ids)
    net.to(cfg.device)

    loss_function = nn.MSELoss()

    optimizer = None
    if cfg.optimizer_select == "adam":
        optimizer = optim.Adam(net.parameters(), lr=cfg.initial_lr)
    elif cfg.optimizer_select == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=cfg.initial_lr, momentum=0.9, weight_decay=5e-4)

    epochs = cfg.max_train_epochs

    best_avg_mae = 100.0
    best_mae_sz = 100.0
    best_mae_hd = 100.0
    best_mae_xg = 100.0
    best_mae_rr = 100.0

    train_steps = len(train_loader)
    for epoch in range(epochs):
        if cfg.flag_if_poly_adjust_lr:
            adjust_lr_poly(optimizer, epoch, cfg.max_train_epochs, base_lr=cfg.initial_lr, power=0.9)
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for _, data in enumerate(train_bar):
            imgs, scores, cls = data
            optimizer.zero_grad()

            # weight = torch.tensor(cfg.c_score_weight).to(cfg.device)
            # predictions = net(imgs.to(cfg.device)) * weight
            predictions = net(imgs.to(cfg.device))

            # scores = torch.cat((scores[0].view(1, len(imgs)), scores[1].view(1, len(imgs)), scores[2].view(1, len(imgs)), scores[3].view(1, len(imgs))), 0).t()
            # loss = loss_function(predictions, scores.to(torch.float32).to(cfg.device))

            loss = loss_function(predictions, scores.to(cfg.device))

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)
        # validate
        net.eval()
        # test_batch_num = 0

        temp = []

        with torch.no_grad():
            test_bar = tqdm(test_loader, file=sys.stdout)
            for test_data in test_bar:
                # test_batch_num += 1
                test_images, test_labels_scores, test_lables_cls = test_data
                outputs = net(test_images.to(cfg.device))
                bbb = mean_absolute_error(outputs.cpu(), test_labels_scores.cpu(), multioutput='raw_values')
                # print(bbb)
                temp.append(bbb * len(test_images))
        # print(temp)
        aaa = np.sum(temp, axis=0)
        mae_sz = (aaa/len(test_dataset)).tolist()[0]
        mae_hd = (aaa/len(test_dataset)).tolist()[1]
        mae_xg = (aaa/len(test_dataset)).tolist()[2]
        mae_rr = (aaa/len(test_dataset)).tolist()[3]

        avg_mae = (1/3*mae_sz+1/4*mae_hd+1/3*mae_xg+1/5*mae_rr)/(1/3+1/4+1/3+1/5)

        if avg_mae < best_avg_mae:
            best_avg_mae = avg_mae
            best_mae_sz = mae_sz
            best_mae_hd = mae_hd
            best_mae_xg = mae_xg
            best_mae_rr = mae_rr

            save_path = './save_models_score_only/model_resnet50_epoch_%d_mae_sz_%0.3f_mae_hd_%0.3f_mae_xg_%0.3f_mae_rr_%0.3f_avg_mae_%0.3f.pth'\
                        % (epoch + 1, best_mae_sz, best_mae_hd, best_mae_xg, best_mae_rr, best_avg_mae)
            torch.save(net, save_path)

        print('[epoch %d] train_loss: %.3f  test_mae_sz: %.3f test_mae_hd: %.3f test_mae_xg: %.3f test_mae_rr: %.3f test_avg_mae: %.3f' %
              (epoch + 1, running_loss / train_steps, mae_sz, mae_hd, mae_xg, mae_rr, avg_mae))
        print('[epoch %d] train_loss: %.3f  best_mae_sz: %.3f best_mae_hd: %.3f best_mae_xg: %.3f best_mae_rr: %.3f best_avg_mae: %.3f' %
            (epoch + 1, running_loss / train_steps, best_mae_sz, best_mae_hd, best_mae_xg, best_mae_rr, best_avg_mae))
        print()

    print('Finished Training')


def test():
    net = torch.load('/media/E_4TB/WW/deep-learning-for-image-processing-master/pytorch_classification/Banhen_Scoring/save_models_score_only/model_swin_small_patch4_window7_224_epoch_12_mae_sz_0.460_mae_hd_0.444_mae_xg_0.469_mae_rrd_0.504_maeaver_0.469.pth')
    test_dataset = ds.Dataset(datatxt=cfg.test_txt, transform=cfg.test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.test_batch_size, shuffle=False, num_workers=cfg.data_num_workers)

    print("using {} images for test.".format(len(test_dataset)))

    net.eval()
    mae_sz = 0.0
    mae_hd = 0.0
    mae_xg = 0.0
    mae_rr = 0.0
    test_batch_num = 0

    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout)
        for test_data in test_bar:
            test_batch_num += 1
            test_images, test_labels_scores, test_lables_cls = test_data
            outputs = net(test_images.to(cfg.device))
            weight = torch.tensor(cfg.c_score_weight).to(cfg.device)
            outputs = outputs * weight

            test_labels_scores_sz = test_labels_scores[0]
            test_labels_scores_hd = test_labels_scores[1]
            test_labels_scores_xg = test_labels_scores[2]
            test_labels_scores_rr = test_labels_scores[3]

            test_predict_scores_sz = outputs.t()[0].cpu()
            test_predict_scores_hd = outputs.t()[1].cpu()
            test_predict_scores_xg = outputs.t()[2].cpu()
            test_predict_scores_rr = outputs.t()[3].cpu()

            mae_sz += mean_absolute_error(test_labels_scores_sz, test_predict_scores_sz)
            mae_hd += mean_absolute_error(test_labels_scores_hd, test_predict_scores_hd)
            mae_xg += mean_absolute_error(test_labels_scores_xg, test_predict_scores_xg)
            mae_rr += mean_absolute_error(test_labels_scores_rr, test_predict_scores_rr)

        mae_sz = mae_sz/test_batch_num
        mae_hd = mae_hd/test_batch_num
        mae_xg = mae_xg/test_batch_num
        mae_rr = mae_rr/test_batch_num
        test_mae_all = (mae_sz + mae_hd + mae_xg + mae_rr) / 4
        print(test_mae_all)
        print('test_mae_sz: %.3f test_mae_hd: %.3f test_mae_xg: %.3f test_mae_rrd: %.3f test_mae_all: %.3f' %
                (mae_sz, mae_hd, mae_xg, mae_rr, test_mae_all))


if __name__ == '__main__':

    train_test()
