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


feature_extracting = False


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(num_classes, feature_extracting, use_pretrained=True):
    model_ft = models.resnet18(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extracting)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft


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
    net = timm.create_model('resnet50', pretrained=True, num_classes=cfg.num_classify_objects)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if len(cfg.device_ids) > 1:
        net = torch.nn.DataParallel(net, device_ids=cfg.device_ids)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()

    optimizer = None
    if cfg.optimizer_select == "adam":
        optimizer = optim.Adam(net.parameters(), lr=cfg.initial_lr)
    elif cfg.optimizer_select == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=cfg.initial_lr, momentum=0.9, weight_decay=5e-4)

    best_acc = 0.0
    best_acc_list = [0.0, 0.0, 0.0, 0.0]

    best_bacc = 0.0
    best_bacc_list = [0.0, 0.0, 0.0, 0.0]

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

            predictions = net(imgs.to(cfg.device))

            loss = loss_function(predictions, cls.to(cfg.device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, cfg.max_train_epochs, loss)
        # validate
        net.eval()
        acc = 0.0
        test_batch_num = 0

        tp_list = np.zeros(cfg.num_classify_objects).tolist()
        tn_list = np.zeros(cfg.num_classify_objects).tolist()
        fp_list = np.zeros(cfg.num_classify_objects).tolist()
        fn_list = np.zeros(cfg.num_classify_objects).tolist()

        with torch.no_grad():
            test_bar = tqdm(test_loader, file=sys.stdout)
            for test_data in test_bar:
                test_batch_num += 1
                test_images, test_labels_scores, test_lables_cls = test_data
                outputs = net(test_images.to(cfg.device))

                predict_y = torch.max(outputs, dim=1)[1]
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
            save_path = '/media/D_4TB/SUGURS/Banhen_multi/save_models_clssify_only/model_epoch_%d_acc_%0.3f.pth' % (epoch + 1, best_acc)
            torch.save(net, save_path)

        if test_bacc > best_bacc:
            best_bacc = test_bacc
            best_bacc_list = [acc_banhenai, acc_banhengeda, acc_weisuoxing, acc_zengshengxing]
            save_path = '/media/D_4TB/SUGURS/Banhen_multi/save_models_clssify_only/model_epoch_%d_bacc_%0.3f.pth' % (epoch + 1, best_bacc)
            torch.save(net, save_path)

        print('[epoch %d] train_loss: %.3f,  best_bacc: %.3f, best_bacc_list: [%.3f, %.3f, %.3f, %.3f], '
              'best_acc: %.3f, best_acc_list: [%.3f, %.3f, %.3f, %.3f], '
              'current_test_acc: %.3f, current_acc_list: [%.3f, %.3f, %.3f, %.3f]' %
              (epoch + 1, running_loss / train_steps, best_bacc, best_bacc_list[0], best_bacc_list[1], best_bacc_list[2], best_bacc_list[3],
               best_acc, best_acc_list[0], best_acc_list[1], best_acc_list[2], best_acc_list[3],
               test_acc, acc_banhenai, acc_banhengeda, acc_weisuoxing, acc_zengshengxing))
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
    test_batch_num = 0

    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout)
        for test_data in test_bar:
            test_batch_num += 1
            test_images, test_labels_scores, test_lables_cls = test_data
            outputs = net(test_images.to(cfg.device))

            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, test_lables_cls.to(cfg.device)).sum().item()

        test_acc = acc / len(test_dataset)
        print('test_mae_sz: %.3f test_acc: %.3f' % (test_acc))


if __name__ == '__main__':

    cfg = Config()
    train_test()
    # test_one_img("/media/E_4TB/WW/dataset/AAA【已整理数据】瘢痕/【评分用】瘢痕/ScoreDataset/%s.jpg" % "943-257")
    # test()
