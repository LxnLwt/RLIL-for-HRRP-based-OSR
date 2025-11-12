import torch.nn as nn
import torch
from loss import gcpl_loss, KPFLoss, ARPLoss, SLCPLoss, RingLoss, RPLoss, CACLoss, LPFLossPlus, LogitNormLoss
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channel)
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
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm1d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv1d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm1d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv1d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm1d(out_channel*self.expansion)
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


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 loss_f='gcpl',
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64,
                 dimension_reduction=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.dimension_reduction = dimension_reduction
        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv1d(1, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool1d(1)  # output size = (1, 1)
            # self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.apply(self.weigth_init)
        self.loss_f = loss_f
        if loss_f == 'softmax':
            self.fc = nn.Linear(512 * block.expansion, num_classes)
            self.loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        if loss_f == 'cac':
            self.fc = nn.Linear(512 * block.expansion, num_classes)
            self.loss = CACLoss(n_classes=num_classes, alpha=10)
        if loss_f == 'gcpl':
            self.fc = nn.Linear(512 * block.expansion, 12)
            self.loss = gcpl_loss(n_classes=num_classes, feat_dim=12)
        if loss_f == 'kpf':
            self.fc = nn.Linear(512 * block.expansion, 12)
            self.loss = KPFLoss(n_classes=num_classes, feat_dim=12)
        if loss_f == 'rpl':
            self.fc = nn.Linear(512 * block.expansion, 12)
            self.loss = RPLoss(n_classes=num_classes, feat_dim=12)
        if loss_f == 'arpl':
            self.fc = nn.Linear(512 * block.expansion, 12)
            self.loss = ARPLoss(n_classes=num_classes, feat_dim=12)
        if loss_f == 'logitnorm':
            self.fc = nn.Linear(512 * block.expansion, num_classes)
            self.loss = LogitNormLoss(t=0.05)
        if loss_f == 'slcpl':
            self.fc = nn.Linear(512 * block.expansion, 12)
            self.loss = SLCPLoss(n_classes=num_classes, feat_dim=12)
        if loss_f == 'ring':
            self.fc = nn.Linear(512 * block.expansion, 12)
            self.fc1 = nn.Linear(12, num_classes)
            self.loss = RingLoss()
        if loss_f == 'ecapl':
            self.fc = nn.Linear(512 * block.expansion, 12)
            opt = {
                'weight_pl': 0.1,
                'weight_s': 0.05,
                'weight_d': 0.2,
                'weight_c': 0.1,
                'weight_pl2': 0.05,
                'temp': 0.5,
                'num_classes': num_classes,
                'feat_dim': 12
            }
            self.loss = LPFLossPlus(**opt)

    @staticmethod
    def weigth_init(m):
        if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight.data)
        elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, std=1e-3)
                nn.init.constant_(m.bias.data, 0)

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(channel * block.expansion))

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

    def forward(self, x, labels=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        if self.loss_f == 'softmax':
            if labels is None:
                return {'logits': x}
            loss = self.loss(x, labels)
            return {'logits': x,
                    'loss': loss}

        if self.loss_f == 'ring':
            y = self.fc1(x)
            out = self.loss(x, y, labels)
            return out

        out = self.loss(x, labels)
        return out


def resnet18(loss_f='gcpl', num_classes=1000, include_top=True,  dimension_reduction=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [2, 2, 2, 2], loss_f=loss_f, num_classes=num_classes, include_top=include_top, dimension_reduction=dimension_reduction)


def resnet34(loss_f='gcpl', num_classes=1000, include_top=True,  dimension_reduction=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], loss_f=loss_f, num_classes=num_classes, include_top=include_top, dimension_reduction=dimension_reduction)


def resnet50(num_classes=1000, include_top=True, dimension_reduction=False):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top, dimension_reduction=dimension_reduction)


def resnet101(num_classes=1000, include_top=True, dimension_reduction=False):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top, dimension_reduction=dimension_reduction)


# AlexNet
def AlexNet_Activation():
    return nn.Sequential(nn.Conv1d(1, 96, kernel_size=128, padding=1),
                         nn.BatchNorm1d(96),
                         nn.ReLU(),
                         nn.MaxPool1d(kernel_size=3, stride=2),
                         nn.Conv1d(96, 256, kernel_size=64, padding=2),
                         nn.BatchNorm1d(256),
                         nn.ReLU(),
                         nn.MaxPool1d(kernel_size=3, stride=2),
                         nn.Conv1d(256, 384, kernel_size=32, padding=1),
                         nn.BatchNorm1d(384),
                         nn.ReLU(),
                         nn.Conv1d(384, 384, kernel_size=16, padding=1),
                         nn.BatchNorm1d(384),
                         nn.ReLU(),
                         nn.Conv1d(384, 256, kernel_size=8, padding=1),
                         nn.BatchNorm1d(256),
                         nn.ReLU(),
                         nn.MaxPool1d(kernel_size=3, stride=2),
                         nn.Flatten()
                         )


def AlexNet_output(input=2304, output=4):
    return nn.Sequential(nn.Linear(input, output))


class AlexNet(nn.Module):
    def __init__(self, loss_f='gcpl', num_classes=4, embd_dim=12):
        super(AlexNet, self).__init__()
        self.layer1 = AlexNet_Activation()
        self.layer2 = nn.Linear(2304, 64)
        self.loss_f = loss_f
        if loss_f == 'softmax':
            self.layer3 = nn.Linear(64, num_classes)
            self.loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        if loss_f == 'cac':
            self.layer3 = nn.Linear(64, num_classes)
            self.loss = CACLoss(n_classes=num_classes, alpha=10)
        if loss_f == 'gcpl':
            self.layer3 = AlexNet_output(input=64, output=embd_dim)
            self.loss = gcpl_loss(n_classes=num_classes, feat_dim=embd_dim)
        if loss_f == 'kpf':
            self.layer3 = AlexNet_output(input=64, output=embd_dim)
            self.loss = KPFLoss(n_classes=num_classes, feat_dim=embd_dim)
        if loss_f == 'rpl':
            self.layer3 = AlexNet_output(input=64, output=embd_dim)
            self.loss = RPLoss(n_classes=num_classes, feat_dim=embd_dim)
        if loss_f == 'arpl':
            self.layer3 = AlexNet_output(input=64, output=embd_dim)
            self.loss = ARPLoss(n_classes=num_classes, feat_dim=embd_dim)
        if loss_f == 'logitnorm':
            self.layer3 = AlexNet_output(input=64, output=embd_dim)
            self.loss = LogitNormLoss(t=0.05)
        if loss_f == 'slcpl':
            self.layer3 = AlexNet_output(input=64, output=embd_dim)
            self.loss = SLCPLoss(n_classes=num_classes, feat_dim=embd_dim)
        if loss_f == 'ring':
            self.layer3 = AlexNet_output(input=64, output=embd_dim)
            self.fc1 = nn.Linear(embd_dim, num_classes)
            self.loss = RingLoss()
        if loss_f == 'ecapl':
            self.layer3 = AlexNet_output(input=64, output=embd_dim)
            opt = {
                'weight_pl': 0.1,
                'weight_s': 0.05,
                'weight_d': 0.2,
                'weight_c': 0.1,
                'weight_pl2': 0.05,
                'temp': 0.5,
                'num_classes': num_classes,
                'feat_dim': 12
            }
            self.loss = LPFLossPlus(**opt)

    def forward(self, X, labels=None):
        x = self.layer1(X)
        x = self.layer2(x)
        x = self.layer3(x)

        if self.loss_f == 'softmax':
            if labels is None:
                return {'logits': x}
            loss = self.loss(x, labels)
            return {'logits': x,
                    'loss': loss}

        if self.loss_f == 'ring':
            y = self.fc1(x)
            out = self.loss(x, y, labels)
            return out

        out = self.loss(x, labels)
        return out




