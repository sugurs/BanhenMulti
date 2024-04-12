import numpy as np
from sklearn.model_selection import StratifiedKFold
from numpy import array
import numpy as np
import xlrd2
import random


all_key_list = []
all_class_list = []
all_value_list = []
for f in [114, 943, 650, 235, 208]:
    file = xlrd2.open_workbook(r"/media/E_4TB/WW/dataset/AAA【已整理数据】瘢痕/【评分用】瘢痕/ScoreExcel/%d.xlsx" % f)
    table = file.sheet_by_name('Sheet1')    # 用工作表的名称来调取需要读取的数据，这里需要Sheet1里的数据
    key_list = []
    class_list = []
    value_list = []
    for i in range (0, table.nrows):
        if i <= 2:
            continue
        img_name = table.cell_value(i, 0)
        key_list.append(img_name)
        
        if f == 114:
            class_list.append(0)
        elif f == 943:
            class_list.append(1)
        elif f == 650:
            class_list.append(1)
        elif f == 235:
            class_list.append(2)
        elif f == 208:
            class_list.append(3)
    
        sz = table.cell_value(i, 13)
        hd = table.cell_value(i, 14)
        xg = table.cell_value(i, 15)
        rrd = table.cell_value(i, 16)
        value_list.append([sz, hd,xg, rrd])
    
    all_key_list += key_list
    all_class_list += class_list
    all_value_list += value_list

array_img = np.array(all_key_list)
array_label = np.array(all_class_list)
array_scores = np.array(all_value_list)

skf = StratifiedKFold(n_splits=5).split(array_img, array_label)

cnt = 1
for train_index, test_index in skf:
    # print(len(train_index), len(test_index))

    x_train, y_train = array_img[train_index].tolist(), array_scores[train_index].tolist()
    x_test, y_test = array_img[test_index].tolist(), array_scores[test_index].tolist()
    
    # print(x_train, y_train)
    # print(x_test, y_test)

    train_file=open('./train_split_%d.txt' % (cnt),mode='w')
    test_file=open('./test_split_%d.txt' % (cnt),mode='w')
    
    for kkk in range(len(x_train)):
        train_file.write('%s %0.2f %0.2f %0.2f %0.2f \n' % (x_train[kkk], y_train[kkk][0], y_train[kkk][1], y_train[kkk][2], y_train[kkk][3]))
    for kkk in range(len(x_test)):
        test_file.write('%s %0.2f %0.2f %0.2f %0.2f \n' % (x_test[kkk], y_test[kkk][0], y_test[kkk][1], y_test[kkk][2], y_test[kkk][3]))

    cnt += 1
