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
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from PIL import Image
from PIL import ImageFile
from config import Config
from scnet import scnet34, scnet50, scnet101
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# import PIL.ImageOps
# from tqdm import tqdm


def calc_tfpn(gt, pred, pos_label):
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



def train_test():
    train_dataset = ds.Dataset(datatxt=cfg.train_txt, transform=cfg.data_transform['train'])
    test_dataset = ds.Dataset(datatxt=cfg.test_txt, transform=cfg.data_transform['test'])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=cfg.train_batch_size, shuffle=True,
                                                num_workers=cfg.data_num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=cfg.test_batch_size, shuffle=False,
                                                num_workers=cfg.data_num_workers)

    print("using {} images for training, {} images for validation.".format(len(train_dataset),
                                                                           len(test_dataset)))
    
    net = scnet50(num_scores=cfg.num_score_tasks, num_classes=cfg.num_classify_objects)
    
    pretext_model = torch.load(cfg.pretrained_model_path)
    model_dict = net.state_dict()
    state_dict = {k:v for k,v in pretext_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    net.load_state_dict(model_dict)

    net = torch.nn.DataParallel(net, device_ids=cfg.device_ids)
    net.to(cfg.device)
    print("using {} device.".format(cfg.device_ids))

    criterion_mse = nn.MSELoss()
    criterion_ce = nn.CrossEntropyLoss()

    params = [p for p in net.parameters() if p.requires_grad]

    optimizer = optim.Adam(params, lr=0.0001)
    # optimizer = optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=5e-4)

    epochs = cfg.max_train_epochs

    best_acc = 0.0
    best_mae = 100.0
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for _, data in enumerate(train_bar):
            imgs, scores, cls = data
            optimizer.zero_grad()

            predictions_scores, predictions_classify = net(imgs.to(cfg.device))

            weight = torch.tensor(cfg.c_score_weight).to(cfg.device)
            predictions_scores = predictions_scores*weight
            scores = torch.cat((scores[0].view(1, len(imgs)), scores[1].view(1, len(imgs)), scores[2].view(1, len(imgs)), scores[3].view(1, len(imgs))), 0).t()

            loss_mse = criterion_mse(predictions_scores, scores.to(torch.float32).to(cfg.device))
            loss_ce = criterion_ce(predictions_classify, cls.to(cfg.device))

            loss = loss_mse + loss_ce

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f} loss_mse:{:.3f} loss_ce:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss, loss_mse, loss_ce)
        # validate
        net.eval()
        mae_sz = 0.0
        mae_hd = 0.0
        mae_xg = 0.0
        mae_rr = 0.0
        
        tp_list = np.zeros(cfg.num_classify_objects).tolist()
        tn_list = np.zeros(cfg.num_classify_objects).tolist()
        fp_list = np.zeros(cfg.num_classify_objects).tolist()
        fn_list = np.zeros(cfg.num_classify_objects).tolist()
        acc = 0.0
        test_batch_num = 0

        with torch.no_grad():
            test_bar = tqdm(test_loader, file=sys.stdout)
            for test_data in test_bar:
                test_batch_num += 1
                test_images, test_labels_scores, test_lables_cls = test_data
                outputs_scores, outputs_classify = net(test_images.to(cfg.device))

                #---------------------------回归---------------------------------
                weight = torch.tensor(cfg.c_score_weight).to(cfg.device)
                outputs_scores = outputs_scores*weight

                test_labels_scores_sz = test_labels_scores[0]
                test_labels_scores_hd = test_labels_scores[1]
                test_labels_scores_xg = test_labels_scores[2]
                test_labels_scores_rr = test_labels_scores[3]

                test_predict_scores_sz = outputs_scores.t()[0].cpu()
                test_predict_scores_hd = outputs_scores.t()[1].cpu()
                test_predict_scores_xg = outputs_scores.t()[2].cpu()
                test_predict_scores_rr = outputs_scores.t()[3].cpu()

                mae_sz += mean_absolute_error(test_labels_scores_sz, test_predict_scores_sz)
                mae_hd += mean_absolute_error(test_labels_scores_hd, test_predict_scores_hd)
                mae_xg += mean_absolute_error(test_labels_scores_xg, test_predict_scores_xg)
                mae_rr += mean_absolute_error(test_labels_scores_rr, test_predict_scores_rr)

                #---------------------------分类---------------------------------
                predict_y = torch.max(outputs_classify, dim=1)[1]
                acc += torch.eq(predict_y, test_lables_cls.to(cfg.device)).sum().item()
                
                
                labels_cls = test_lables_cls.cpu().numpy().tolist()
                predicted_cls = predict_y.cpu().numpy().tolist()
                
                for kkk in range(cfg.num_classify_objects):
                    temp_tp, temp_tn, temp_fp, temp_fn = calc_tfpn(labels_cls, predicted_cls, kkk)
                    tp_list[kkk] += temp_tp
                    tn_list[kkk] += temp_tn
                    fp_list[kkk] += temp_fp
                    fn_list[kkk] += temp_fn
                # print(tp_list)
                
        
        tp_list = np.array(tp_list)
        tn_list = np.array(tn_list)
        fp_list = np.array(fp_list)
        fn_list = np.array(fn_list)   
        print(tp_list)

        print("xxxsas1x")

        mae_sz = mae_sz/test_batch_num
        mae_hd = mae_hd/test_batch_num
        mae_xg = mae_xg/test_batch_num
        mae_rr = mae_rr/test_batch_num
        test_mae_all = (mae_sz + mae_hd + mae_xg + mae_rr) / 4

        # acc_banhenai = acc_banhenai/test_batch_num
        # acc_banhengeda = acc_banhengeda/test_batch_num
        # acc_weisuoxing = acc_weisuoxing/test_batch_num
        # acc_zengshengxing = acc_zengshengxing/test_batch_num

        print(tp_list)
        print(fn_list)
        acc_banhenai = tp_list[0] / (tp_list[0] + fn_list[0])
        acc_banhengeda = tp_list[1] / (tp_list[1] + fn_list[1])
        acc_weisuoxing = tp_list[2] / (tp_list[2] + fn_list[2])
        acc_zengshengxing = tp_list[3] / (tp_list[3] + fn_list[3])

        test_acc = acc / len(test_dataset)

        print('[epoch %d] train_loss: %.3f  test_mae_sz: %.3f test_mae_hd: %.3f test_mae_xg: %.3f test_mae_rrd: %.3f test_acc_banhenai: %.3f test_acc_banhengeda: %.3f test_acc_weisuoxing: %.3f test_acc_zengshengxing: %.3f test_mae_all: %.3f test_acc: %.3f' %
              (epoch + 1, running_loss / train_steps, mae_sz, mae_hd, mae_xg, mae_rr, acc_banhenai, acc_banhengeda, acc_weisuoxing, acc_zengshengxing, test_mae_all, test_acc))
        print()
            
        if test_mae_all < best_mae:
            best_mae = test_mae_all
            save_path = '/media/E_4TB/WW/code/banhen/Banhen_Scoring/Banhen_multi/save_models/model_epoch_%d_mae_%0.3f_acc_%0.3f.pth'%(epoch + 1, test_mae_all, test_acc)
            torch.save(net, save_path)
        elif test_acc > best_acc:
            best_acc = test_acc
            save_path = '/media/E_4TB/WW/code/banhen/Banhen_Scoring/Banhen_multi/save_models/model_epoch_%d_mae_%0.3f_acc_%0.3f.pth'%(epoch + 1, test_mae_all, test_acc)
            torch.save(net, save_path)

    print('Finished Training')


# def test_one_img(img_name):
#     net = torch.load('/media/E_4TB/WW/deep-learning-for-image-processing-master/pytorch_classification/Banhen_Scoring/save_models/model_epoch_44_mse_0.547.pth')
#     net.eval()
#     print("using {} device.".format(cfg.device_ids))

#     # summary(net, input_size=[(3, 224, 224)], batch_size=1)

#     img = Image.open(img_name).convert('RGB')
#     img = cfg.test_transform(img).unsqueeze(0)

#     weight = torch.tensor(cfg.c_score_weight).to(cfg.device)
#     predict = net(img.to(cfg.device))*weight
#     predict = predict.cpu().detach().numpy()[0]
    
#     print(predict)


# def test_imgs_1b1():
#     net = torch.load('/media/E_4TB/WW/deep-learning-for-image-processing-master/pytorch_classification/Banhen_Scoring/save_models/model_epoch_1_mae_2.651.pth')
#     net.eval()
#     print("using {} device.".format(cfg.device_ids))

#     fh = open(cfg.test_txt, 'r')
#     test_res_file = open(cfg.test_res_txt,mode='w')

#     mae_list = []

#     for line in tqdm(fh):
#         line = line.rstrip()
#         words = line.split(' ')
#         img_name = ("/media/E_4TB/WW/dataset/AAA【已整理数据】瘢痕/【评分用】瘢痕/ScoreDataset/"+words[0]+'.jpg')
#         score = [float(words[1]), float(words[2]), float(words[3]), float(words[4])]
#         img = Image.open(img_name).convert('RGB')

#         test_transform = cfg.test_transform

#         img = test_transform(img).unsqueeze(0)

#         weight = torch.tensor(cfg.c_score_weight).to(cfg.device)

#         predict = net(img.to(cfg.device))*weight

#         predict = predict.cpu().detach().numpy()[0]

#         # test_res_file.writelines("img_name %s\n" % img_name)
#         # test_res_file.writelines("label %s\n" % str(np.round(score,2)))
#         # test_res_file.writelines("prdct %s\n" % str(np.round(predict,2)))
#         # test_res_file.writelines("\n")


#         mae_list.append(mean_absolute_error(score, predict))

#         # print("img_name", img_name)
#         # print("label", np.round(score,2))
#         # print("prdct", np.round(predict,2))
#         # print("mae", mean_absolute_error(score, predict))
#         # print("\n")
#     print("mae", np.array(mae_list).sum()/len(mae_list))


def test():
    net = torch.load('/media/E_4TB/WW/code/banhen/Banhen_Scoring/Banhen_multi/save_models1024+PA(dim2)/2/2)/model_epoch_53_mae_0.568_acc_0.858.pth')
    test_dataset = ds.Dataset(datatxt=cfg.test_txt, transform=cfg.data_transform['test'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.test_batch_size, shuffle=False, num_workers=cfg.data_num_workers)

    print("using {} images for test.".format(len(test_dataset)))

    net.eval()
    mae_sz = 0.0
    mae_hd = 0.0
    mae_xg = 0.0
    mae_rr = 0.0
    
    
    tp_list = np.zeros(cfg.num_classify_objects).tolist()
    tn_list = np.zeros(cfg.num_classify_objects).tolist()
    fp_list = np.zeros(cfg.num_classify_objects).tolist()
    fn_list = np.zeros(cfg.num_classify_objects).tolist()
    
    acc = 0.0
    test_batch_num = 0

    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout)
        for test_data in test_bar:
            test_batch_num += 1
            test_images, test_labels_scores, test_lables_cls = test_data
            outputs_scores, outputs_classify = net(test_images.to(cfg.device))

            #---------------------------回归---------------------------------
            weight = torch.tensor(cfg.c_score_weight).to(cfg.device)
            outputs_scores = outputs_scores*weight

            test_labels_scores_sz = test_labels_scores[0]
            test_labels_scores_hd = test_labels_scores[1]
            test_labels_scores_xg = test_labels_scores[2]
            test_labels_scores_rr = test_labels_scores[3]

            test_predict_scores_sz = outputs_scores.t()[0].cpu()
            test_predict_scores_hd = outputs_scores.t()[1].cpu()
            test_predict_scores_xg = outputs_scores.t()[2].cpu()
            test_predict_scores_rr = outputs_scores.t()[3].cpu()

            mae_sz += mean_absolute_error(test_labels_scores_sz, test_predict_scores_sz)
            mae_hd += mean_absolute_error(test_labels_scores_hd, test_predict_scores_hd)
            mae_xg += mean_absolute_error(test_labels_scores_xg, test_predict_scores_xg)
            mae_rr += mean_absolute_error(test_labels_scores_rr, test_predict_scores_rr)

            #---------------------------分类---------------------------------
            predict_y = torch.max(outputs_classify, dim=1)[1]
            acc += torch.eq(predict_y, test_lables_cls.to(cfg.device)).sum().item()
            
            labels_cls = test_lables_cls.cpu().numpy().tolist()
            predicted_cls = predict_y.cpu().numpy().tolist()
            
            
            for kkk in range(cfg.num_classify_objects):
                temp_tp, temp_tn, temp_fp, temp_fn = calc_tfpn(labels_cls, predicted_cls, kkk)
                tp_list[kkk] += temp_tp
                tn_list[kkk] += temp_tn
                fp_list[kkk] += temp_fp
                fn_list[kkk] += temp_fn
                
        
        # acc_banhenai = (tp_list[0] + tn_list[0]) / len(test_dataset)
        # acc_banhengeda = (tp_list[1] + tn_list[1]) / len(test_dataset)
        # acc_weisuoxing = (tp_list[2] + tn_list[2]) / len(test_dataset)
        # acc_zengshengxing = (tp_list[3] + tn_list[3]) / len(test_dataset)
        
        print(tp_list)
        print(fn_list)
        acc_banhenai = tp_list[0] / (tp_list[0] + fn_list[0])
        acc_banhengeda = tp_list[1] / (tp_list[1] + fn_list[1])
        acc_weisuoxing = tp_list[2] / (tp_list[2] + fn_list[2])
        acc_zengshengxing = tp_list[3] / (tp_list[3] + fn_list[3])
        
        
        
        tp_list = np.array(tp_list)
        tn_list = np.array(tn_list)
        fp_list = np.array(fp_list)
        fn_list = np.array(fn_list)   
        
        
        current_sen = sum((tp_list / (tp_list + fn_list)).tolist())/cfg.num_classify_objects
        current_spe = sum((tn_list / (fp_list + tn_list)).tolist())/cfg.num_classify_objects


        mae_sz = mae_sz/test_batch_num
        mae_hd = mae_hd/test_batch_num
        mae_xg = mae_xg/test_batch_num
        mae_rr = mae_rr/test_batch_num
        test_mae_all = (mae_sz + mae_hd + mae_xg + mae_rr) / 4

        test_acc = acc / len(test_dataset)

        # print(test_mae_all)
        # print('test_mae_sz: %.3f test_mae_hd: %.3f test_mae_xg: %.3f test_mae_rrd: %.3f test_mae_all: %.3f test_acc: %.3f' %
        #         (mae_sz, mae_hd, mae_xg, mae_rr, test_mae_all, test_acc))
        
        print('test_mae_sz: %.3f test_mae_hd: %.3f test_mae_xg: %.3f test_mae_rrd: %.3f test_acc_banhenai: %.3f test_acc_banhengeda: %.3f test_acc_weisuoxing: %.3f test_acc_zengshengxing: %.3f test_mae_all: %.3f test_acc: %.3f' %
              (mae_sz, mae_hd, mae_xg, mae_rr, acc_banhenai, acc_banhengeda, acc_weisuoxing, acc_zengshengxing, test_mae_all, test_acc))

if __name__ == '__main__':

    cfg = Config()

    # train_test()
    # test_one_img("/media/E_4TB/WW/dataset/AAA【已整理数据】瘢痕/【评分用】瘢痕/ScoreDataset/%s.jpg" % "943-257")
    test()
