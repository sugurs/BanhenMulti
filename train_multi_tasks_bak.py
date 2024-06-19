import os
import sys
import json
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
import timm
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
from config import Config
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# import PIL.ImageOps
# from tqdm import tqdm


# dingyi config
cfg = Config()


def calc_tpfn(gt, pred, pos_label):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(len(gt)):
        if gt[i] == pos_label and pred[i] == pos_label:
            TP += 1
        elif gt[i] != pos_label and pred[i] != pos_label:
            TN += 1
        elif gt[i] != pos_label and pred[i] == pos_label:
            FP += 1
        elif gt[i] == pos_label and pred[i] != pos_label:
            FN += 1
    return TP, TN, FP, FN


def adjust_lr_poly(optimizer, epoch, max_epochs, base_lr=0.1, power=0.9):
    lr = base_lr * (1 - epoch / max_epochs) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class ScoringModel(nn.Module):
    def __init__(self, num_cls_objects=4, num_reg_tasks=4):
        super(ScoringModel, self).__init__()
        self.backbone = timm.create_model(cfg.backbone_name, pretrained=True, features_only=True, out_indices=[4])

        if cfg.freeze_feature_extractor_weights:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if cfg.backbone_name == 'resnet50':
            self.mlp_classify = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_cls_objects)

                # nn.Linear(2048, 128),
                # nn.ReLU(),
                # nn.Dropout(0.5),
                # nn.Linear(128, num_cls_objects)

                # nn.Linear(2048, num_cls_objects)
            )
            self.mlp_regress = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_reg_tasks)

                # nn.Linear(2048, 128),
                # nn.ReLU(),
                # nn.Dropout(0.5),
                # nn.Linear(128, num_reg_tasks)

                # nn.Linear(2048, num_reg_tasks)
            )
        elif cfg.backbone_name == 'resnet18':
            self.mlp_classify = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_cls_objects)

                # nn.Linear(512, 1024),
                # nn.ReLU(),
                # nn.Dropout(0.5),
                # nn.Linear(1024, num_cls_objects)

                # nn.Linear(512, num_cls_objects)
            )
            self.mlp_regress = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_reg_tasks)

                # nn.Linear(512, 1024),
                # nn.ReLU(),
                # nn.Dropout(0.5),
                # nn.Linear(1024, num_reg_tasks)

                # nn.Linear(512, num_reg_tasks)
            )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        features = self.backbone(x)[0]
        
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        
        output_cls = self.mlp_classify(features)
        
        output_reg = self.mlp_regress(features)
        
        return output_cls, output_reg


def train_test():
    train_dataset = ds.Dataset(datatxt=cfg.train_txt, transform=cfg.data_transform['train'])
    test_dataset = ds.Dataset(datatxt=cfg.test_txt, transform=cfg.data_transform['test'])

    # print(train_dataset.cls)
    class_counts = torch.unique(torch.tensor(train_dataset.cls), return_counts=True)[1]
    class_weights = 1. / class_counts.float()
    sample_weights = [class_weights[cls] for cls in train_dataset.cls]
    train_sampler = WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)
    # exit(0)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train_batch_size, sampler=train_sampler, num_workers=cfg.data_num_workers)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True, num_workers=cfg.data_num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.test_batch_size, shuffle=False, num_workers=cfg.data_num_workers)

    print("using {} images for training, {} images for validation.".format(len(train_dataset), len(test_dataset)))

    # net = initialize_model(cfg.num_classify_objects, feature_extracting)
    # net = timm.create_model('resnet50', pretrained=True, num_classes=cfg.num_classify_objects)
    
    net = ScoringModel(cfg.num_classify_objects, cfg.num_score_tasks)

    if len(cfg.device_ids) > 1:
        net = torch.nn.DataParallel(net, device_ids=cfg.device_ids)
    net.to(cfg.device)

    loss_function_cls = nn.CrossEntropyLoss()
    loss_function_reg = nn.MSELoss()

    optimizer = None
    if cfg.optimizer_select == "adam":
        optimizer = optim.Adam(net.parameters(), lr=cfg.initial_lr)
    elif cfg.optimizer_select == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=cfg.initial_lr, momentum=0.9, weight_decay=5e-4)

    best_acc = 0.0
    best_acc_list = [0.0, 0.0, 0.0, 0.0]

    best_bacc = 0.0
    best_bacc_list = [0.0, 0.0, 0.0, 0.0]
    
    best_avg_mae = 100.0
    best_mae_sz = 100.0
    best_mae_hd = 100.0
    best_mae_xg = 100.0
    best_mae_rr = 100.0

    train_steps = len(train_loader)
    for epoch in range(cfg.max_train_epochs):
        if cfg.flag_if_poly_adjust_lr:
            adjust_lr_poly(optimizer, epoch, cfg.max_train_epochs, base_lr=cfg.initial_lr, power=0.9)
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for _, data in enumerate(train_bar):
            imgs, scores, cls = data
            optimizer.zero_grad()

            out_cls, out_reg = net(imgs.to(cfg.device))

            loss_cls = loss_function_cls(out_cls, cls.to(cfg.device))
            loss_reg = loss_function_reg(out_reg, scores.to(cfg.device))
            
            loss = loss_cls + loss_reg

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}], loss:{:.3f}, loss_cls:{:.3f}, loss_reg:{:.3f}".format(epoch + 1, cfg.max_train_epochs, loss, loss_cls, loss_reg)
        # validate
        net.eval()
        acc = 0.0
        
        temp = []

        tp_list = np.zeros(cfg.num_classify_objects).tolist()
        tn_list = np.zeros(cfg.num_classify_objects).tolist()
        fp_list = np.zeros(cfg.num_classify_objects).tolist()
        fn_list = np.zeros(cfg.num_classify_objects).tolist()

        with torch.no_grad():
            test_bar = tqdm(test_loader, file=sys.stdout)
            for test_data in test_bar:
                test_images, test_labels_scores, test_lables_cls = test_data
                outputs_cls, outputs_reg = net(test_images.to(cfg.device))
                
                bbb = mean_absolute_error(outputs_reg.cpu(), test_labels_scores.cpu(), multioutput='raw_values')
                temp.append(bbb * len(test_images))

                predict_y = torch.max(outputs_cls, dim=1)[1]
                acc += torch.eq(predict_y, test_lables_cls.to(cfg.device)).sum().item()

                labels_cls = test_lables_cls.cpu().numpy().tolist()
                predicted_cls = predict_y.cpu().numpy().tolist()
                
                for kkk in range(cfg.num_classify_objects):
                    temp_tp, temp_tn, temp_fp, temp_fn = calc_tpfn(labels_cls, predicted_cls, kkk)
                    tp_list[kkk] += temp_tp
                    tn_list[kkk] += temp_tn
                    fp_list[kkk] += temp_fp
                    fn_list[kkk] += temp_fn
                # print(tp_list)
                  
        # print(tp_list)
        # print(fn_list)
        acc_banhenai = tp_list[0] / (tp_list[0] + fn_list[0])
        acc_banhengeda = tp_list[1] / (tp_list[1] + fn_list[1])
        acc_weisuoxing = tp_list[2] / (tp_list[2] + fn_list[2])
        acc_zengshengxing = tp_list[3] / (tp_list[3] + fn_list[3])
        
        test_acc = acc / len(test_dataset)

        test_bacc = (acc_banhenai + acc_banhengeda + acc_weisuoxing + acc_zengshengxing) / 4
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_acc_list = [acc_banhenai, acc_banhengeda, acc_weisuoxing, acc_zengshengxing]

        if test_bacc > best_bacc:
            best_bacc = test_bacc
            best_bacc_list = [acc_banhenai, acc_banhengeda, acc_weisuoxing, acc_zengshengxing]
        
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

        print('[epoch %d] train_loss: %.3f, best_bacc: %.3f, best_bacc_list: [%.3f, %.3f, %.3f, %.3f], '
              'best_acc: %.3f, best_acc_list: [%.3f, %.3f, %.3f, %.3f]' %
              (epoch + 1, running_loss / train_steps, best_bacc, best_bacc_list[0], best_bacc_list[1], best_bacc_list[2], best_bacc_list[3],
               best_acc, best_acc_list[0], best_acc_list[1], best_acc_list[2], best_acc_list[3]))

        print('[epoch %d] train_loss: %.3f  best_mae_sz: %.3f best_mae_hd: %.3f best_mae_xg: %.3f best_mae_rr: %.3f best_avg_mae: %.3f' %
            (epoch + 1, running_loss / train_steps, best_mae_sz, best_mae_hd, best_mae_xg, best_mae_rr, best_avg_mae))
              
        for param_group in optimizer.param_groups:
            print(f"Epoch {epoch + 1}, Current Learning Rate: {param_group['lr']}")
        print()

    print('Finished Training')


def test():
    net = torch.load('/media/E_4TB/WW/deep-learning-for-image-processing-master/pytorch_classification/Banhen_Scoring/save_models_classify_only/model_epoch_13_acc_0.543.pth')
    test_dataset = ds.Dataset(datatxt=cfg.test_txt, transform=cfg.test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.test_batch_size, shuffle=False, num_workers=cfg.data_num_workers)

    print("using {} images for test.".format(len(test_dataset)))

    net.eval()
    acc = 0.0

    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout)
        for test_data in test_bar:
            test_images, test_labels_scores, test_lables_cls = test_data
            outputs = net(test_images.to(cfg.device))

            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, test_lables_cls.to(cfg.device)).sum().item()

        test_acc = acc / len(test_dataset)
        print('test_mae_sz: %.3f test_acc: %.3f' % (test_acc))


if __name__ == '__main__':

    train_test()
    # test_one_img("/media/E_4TB/WW/dataset/AAA【已整理数据】瘢痕/【评分用】瘢痕/ScoreDataset/%s.jpg" % "943-257")
    # test()
