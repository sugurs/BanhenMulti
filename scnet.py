import torch.nn as nn
import torch
from torchsummary import summary


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class GAM_Module(nn.Module):
    """ Gate attention module"""
    def __init__(self, in_dim):
        super(GAM_Module, self).__init__()
        
        # self.in_dim = 2048
        # self.chanel_in = in_dim   #这句是干什么的？
        
        
        # self.score_conv = nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=3, padding=1, bias=False),
        #                                nn.BatchNorm2d(in_dim//4),
        #                                nn.Conv2d(in_channels=in_dim//4, out_channels=in_dim//8, kernel_size=1, stride=1, bias=False))
        # self.classify_conv = nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=3, padding=1, bias=False),
        #                                nn.BatchNorm2d(in_dim//4),
        #                                nn.Conv2d(in_channels=in_dim//4, out_channels=in_dim//8, kernel_size=1, stride=1, bias=False))
        
        
        
        self.score_conv = nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=3, padding=1, bias=False),
                                       nn.BatchNorm2d(in_dim//4),
                                       nn.Conv2d(in_channels=in_dim//4, out_channels=in_dim//8, kernel_size=1, stride=1, bias=False))
        self.classify_conv = nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=3, padding=1, bias=False),
                                       nn.BatchNorm2d(in_dim//4),
                                       nn.Conv2d(in_channels=in_dim//4, out_channels=in_dim//8, kernel_size=1, stride=1, bias=False))
        
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x):
        # 回归子网和分类子网特征
        score_x = self.score_conv(x)
        classify_x = self.classify_conv(x)
        
        # print(score_feat.shape)
        
        # 交互注意力
        score_feat = score_x + self.sigmoid(classify_x) * score_x
        # print(self.sigmoid(score_x).shape)
        # print(score_feat.shape)
        
        classify_feat = classify_x + self.sigmoid(score_x) * classify_x
        return score_feat, classify_feat


class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        
        # self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        # self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        # self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # self.gamma = nn.Parameter(torch.zeros(1))



        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out



class SCNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_scores=4,
                 num_classes=4,
                 groups=1,
                 width_per_group=64):
        super(SCNet, self).__init__()
        self.in_channel = 64
        
        self.groups = groups
        self.width_per_group = width_per_group
        
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        
        # self.conv_reduce = nn.Conv2d(2048, 256, 1, 1, bias=False)
        # 如果resnet换了，这里维度需要改一下
        # self.score_conv = nn.Sequential(nn.Conv2d(2048, 512, 3, padding=1, bias=False),
        #                                nn.BatchNorm2d(512),
        #                                nn.Conv2d(512, 256, 1, 1, bias=False))
        # self.classify_conv = nn.Sequential(nn.Conv2d(2048, 512, 3, padding=1, bias=False),
        #                                nn.BatchNorm2d(512),
        #                                nn.Conv2d(512, 256, 1, 1, bias=False))
        
        # self.sigmoid = nn.Sigmoid()
        self.ganet = GAM_Module(2048)
        
        self.panet_score = PAM_Module(256)
        self.panet_classify = PAM_Module(256)
        

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
        
        # self.fc_score = nn.Linear(512 * block.expansion, num_scores)
        # self.fc_classify = nn.Linear(512 * block.expansion, num_classes)
        self.fc_score = nn.Linear(256, num_scores)
        self.fc_classify = nn.Linear(256, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.shape)    #resnet输出的size是7*7，需不需要变大一点
        
        # 回归子网和分类子网特征
        # score_x = self.score_conv(x)
        # classify_x = self.classify_conv(x)
        
        # print(score_feat.shape)
        
        
        # 子网
        
        
        
        # 交互注意力
        # 门注意力
        score_GA_feat, classify_GA_feat = self.ganet(x)
        # score_feat = score_x + self.sigmoid(classify_x) * score_x
        # # print(self.sigmoid(score_x).shape)
        # print(score_GA_feat.shape)
        
        # classify_feat = classify_x + self.sigmoid(score_x) * classify_x
        
        # 位置注意力
        score_PA_feat = self.panet_score(score_GA_feat)
        classify_PA_feat = self.panet_classify(classify_GA_feat)
        # print(score_PA_feat.shape)


        score_out = self.avgpool(score_PA_feat)
        score_out = torch.flatten(score_out, 1)
        
        classify_out = self.avgpool(classify_PA_feat)
        classify_out = torch.flatten(classify_out, 1)
        
        
        y_socre = self.fc_score(score_out)
        y_classify = self.fc_classify(classify_out)

        return y_socre, y_classify


def scnet34(num_scores=4, num_classes=4):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return SCNet(BasicBlock, [3, 4, 6, 3], num_scores=num_scores, num_classes=num_classes)


def scnet50(num_scores=4, num_classes=4):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return SCNet(Bottleneck, [3, 4, 6, 3], num_scores=num_scores, num_classes=num_classes)


def scnet101(num_scores=4, num_classes=4):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return SCNet(Bottleneck, [3, 4, 23, 3], num_scores=num_scores, num_classes=num_classes)


# def resnext50_32x4d(num_classes=1000, include_top=True):
#     # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
#     groups = 32
#     width_per_group = 4
#     return SCNet(Bottleneck, [3, 4, 6, 3],
#                   num_classes=num_classes,
#                   include_top=include_top,
#                   groups=groups,
#                   width_per_group=width_per_group)


# def resnext101_32x8d(num_classes=1000, include_top=True):
#     # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
#     groups = 32
#     width_per_group = 8
#     return SCNet(Bottleneck, [3, 4, 23, 3],
#                   num_classes=num_classes,
#                   include_top=include_top,
#                   groups=groups,
#                   width_per_group=width_per_group)
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = scnet50()

    # pretext_model = torch.load("/media/E_4TB/WW/deep-learning-for-image-processing-master/pytorch_classification/Banhen_Scoring/resnet34-pre.pth")
    
    pretext_model = torch.load("/media/E_4TB/WW/code/banhen/Banhen_Scoring/resnet50-0676ba61.pth")
    model_dict = model.state_dict()
    state_dict = {k:v for k,v in pretext_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)


    # model = model.to(device)
    # summary(model, input_size=[(3, 224, 224)])
    
    
    img = torch.randn([64, 3, 224, 224])
    out = model(img)

