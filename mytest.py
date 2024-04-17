# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image
# import numpy as np
# from config import Config
# import timm
# # ImageFile.LOAD_TRUNCATED_IMAGES = True
# import PIL.ImageOps
# from tqdm import tqdm


#
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(16 * 64 * 64, 256),
#             nn.ReLU(),
#             nn.Linear(256, 4)
#         )
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x
#
#
# class CustomDataset(Dataset):
#     def __init__(self, data_path, transform=None):
#         self.data_path = data_path
#         self.transform = transform
#
#         fh = open("train.txt", 'r')
#         imgs = []
#         scores = []
#         for line in fh:
#             line = line.rstrip()
#             words = line.split(' ')
#             imgs.append("/media/E_4TB/WW/dataset/AAA【已整理数据】瘢痕/【评分用】瘢痕/ScoreDataset/" + words[0] + '.jpg')
#             scores.append([float(words[1]), float(words[2]), float(words[3]), float(words[4])])
#         self.images = imgs
#         self.targets = scores
#
#     def __len__(self):
#         return len(self.images)
#
#     def __getitem__(self, idx):
#         img = Image.open(self.images[idx]).convert('RGB')
#
#         if self.transform:
#             img = self.transform(img)
#
#         target = torch.tensor(self.targets[idx], dtype=torch.float32)
#         return img, target
#
#
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor()
# ])
#
#
# train_dataset = CustomDataset(data_path='path_to_train_data', transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
#
#
# model = Net().to(device)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#
# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     for batch_images, batch_targets in train_loader:
#         optimizer.zero_grad()
#         outputs = model(batch_images.to(device))
#
#         # print(outputs.size())
#         # print(batch_targets.size())
#
#         loss = criterion(outputs, batch_targets.to(device))
#         loss.backward()
#         optimizer.step()
#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
#
#
# torch.save(model.state_dict(), '/media/D_4TB/SUGURS/Banhen_multi/save_models_clssify_only/model.pth')
#

#
# class CustomModel(nn.Module):
#     def __init__(self, num_classes=4):
#         super(CustomModel, self).__init__()
#         self.base_model = timm.create_model('resnet18', pretrained=True)
#
#         for param in self.base_model.parameters():
#             param.requires_grad = False
#
#         self.mlp = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, num_classes)
#         )
#
#     def forward(self, x):
#         x = self.base_model(x)
#         x = self.mlp(x)
#         return x
#
#
# model = CustomModel(4)

import torch
import torch.nn as nn


mse_loss = nn.MSELoss()

predictions = torch.tensor([
    [-1.0433,  0.7545,  0.0914, -0.4057],
    [0.4255,  0.2729,  1.1568,  0.3180],
    [-0.9085, -0.4700, -1.1146, -0.6535]
])

p1 = torch.tensor([-1.0433, 0.4255, -0.9085])
p2 = torch.tensor([0.7545, 0.2729, -0.4700])
p3 = torch.tensor([0.0914, 1.1568, -1.1146])
p4 = torch.tensor([-0.4057, 0.3180, -0.6535])

targets = torch.tensor([
    [0.0194, -0.1611, -2.4982,  1.1459],
    [-0.0056, -0.6082,  0.9976, -0.7496],
    [0.6966, -0.4037, -1.7588, -0.6110]
])

t1 =torch.tensor([0.0194, -0.0056, 0.6966])
t2 =torch.tensor([-0.1611, -0.6082, -0.4037])
t3 =torch.tensor([-2.4982, 0.9976, -1.7588])
t4 =torch.tensor([1.1459, -0.7496, -0.6110])

loss = mse_loss(predictions, targets)
print(loss)

loss1 = mse_loss(p1, t1)
loss2 = mse_loss(p2, t2)
loss3 = mse_loss(p3, t3)
loss4 = mse_loss(p4, t4)

print(loss1, loss2, loss3, loss4)
print((loss1+loss2+loss3+loss4)/4.0)
