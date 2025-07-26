import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from FSFE import OS_block
from loss import gcpl_loss, KPFLoss, ARPLoss, SLCPLoss, RingLoss, RPLoss, CACLoss
from DWT_layer import DWT_1D, IDWT_1D


def calculate_mask_index(kernel_length_now, largest_kernel_lenght):
    right_zero_mast_length = math.ceil((largest_kernel_lenght - 1) / 2) - math.ceil((kernel_length_now - 1) / 2)
    left_zero_mask_length = largest_kernel_lenght - kernel_length_now - right_zero_mast_length
    return left_zero_mask_length, left_zero_mask_length + kernel_length_now


def creat_mask(number_of_input_channel, number_of_output_channel, kernel_length_now, largest_kernel_lenght):
    ind_left, ind_right = calculate_mask_index(kernel_length_now, largest_kernel_lenght)
    mask = np.ones((number_of_input_channel, number_of_output_channel, largest_kernel_lenght))
    mask[:, :, 0:ind_left] = 0
    mask[:, :, ind_right:] = 0
    return mask


def creak_layer_mask(layer_parameter_list):
    largest_kernel_lenght = layer_parameter_list[-1][-1]
    mask_list = []
    init_weight_list = []
    bias_list = []
    for i in layer_parameter_list:
        conv = torch.nn.Conv1d(in_channels=i[0], out_channels=i[1], kernel_size=i[2], dilation=2)
        ind_l, ind_r = calculate_mask_index(i[2], largest_kernel_lenght)
        big_weight = np.zeros((i[1], i[0], largest_kernel_lenght))
        big_weight[:, :, ind_l:ind_r] = conv.weight.detach().numpy()

        bias_list.append(conv.bias.detach().numpy())
        init_weight_list.append(big_weight)

        mask = creat_mask(i[1], i[0], i[2], largest_kernel_lenght)
        mask_list.append(mask)

    mask = np.concatenate(mask_list, axis=0)
    init_weight = np.concatenate(init_weight_list, axis=0)
    init_bias = np.concatenate(bias_list, axis=0)
    return mask.astype(np.float32), init_weight.astype(np.float32), init_bias.astype(np.float32)


def layer_parameter_list_input_change(layer_parameter_list, input_channel):
    
    new_layer_parameter_list = []
    for i, i_th_layer_parameter in enumerate(layer_parameter_list):
        if i == 0:
            first_layer_parameter = []
            for cov_parameter in i_th_layer_parameter:
                first_layer_parameter.append((input_channel,cov_parameter[1],cov_parameter[2]))
            new_layer_parameter_list.append(first_layer_parameter)
        else:
            new_layer_parameter_list.append(i_th_layer_parameter)
    return new_layer_parameter_list


class SampaddingConv1D_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SampaddingConv1D_BN, self).__init__()
        self.padding = nn.ConstantPad1d((int((kernel_size-1)/2), int(kernel_size/2)), 0)
        self.conv1d = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.bn = nn.BatchNorm1d(num_features=out_channels)
        
    def forward(self, X):
        X = self.padding(X)
        X = self.conv1d(X)
        X = self.bn(X)
        return X

    
class Res_OS_layer(nn.Module):
    def __init__(self, layer_parameter_list, out_put_channel_numebr):
        super(Res_OS_layer, self).__init__()  
        self.layer_parameter_list = layer_parameter_list
        self.net = OS_block(layer_parameter_list, False)
        self.res = SampaddingConv1D_BN(layer_parameter_list[0][0][0], out_put_channel_numebr, 1)
        
    def forward(self, X):
        temp = self.net(X)
        shot_cut = self.res(X)
        block = F.relu(torch.add(shot_cut, temp))
        return block


class RSBU_CW(torch.nn.Module):
    def __init__(self, layer_parameters):
        super().__init__()
        self.dwt = DWT_1D(wavename='sym2')
        self.idwt = IDWT_1D(wavename='sym2')
        os_mask, init_weight, init_bias = creak_layer_mask(layer_parameters)
        out_channels = os_mask.shape[0]
        kernel_size = int(abs((math.log(out_channels, 2)+1)/2))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # kernel_size = kernel_size + 4
        padding = int((kernel_size - 1) / 2)

        self.global_average_pool = nn.AdaptiveAvgPool1d(1)
        self.FC = nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding),
            nn.Sigmoid()
        )
        self.flatten = nn.Flatten()
        self.average_pool = nn.AvgPool1d(kernel_size=1, stride=2)

    def forward(self, x):
        x_l, x_h = self.dwt(x)
        x_abs = torch.abs(x_h)
        gap = self.global_average_pool(x_abs)
        gap1 = gap.permute(0, 2, 1)
        alpha = self.FC(gap1)
        alpha = alpha.permute(0, 2, 1)
        alpha = self.flatten(alpha)
        gap = self.flatten(gap)

        threshold = torch.mul(gap, alpha)
        threshold = torch.unsqueeze(threshold, 2)
        # 软阈值化
        sub = x_abs - threshold
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x_h = torch.mul(torch.sign(x_h), n_sub)
        x = self.idwt(x_l, x_h)
        x = F.relu(x)
        return x


class FSBN(nn.Module):
    def __init__(self, layer_parameter_list, loss_f='gcpl', n_class=9, n_layers=3, embd_dim=12, few_shot=False):
        super(FSBN, self).__init__()
        self.few_shot = few_shot
        self.layer_parameter_list = layer_parameter_list
        self.n_layers = n_layers
        self.rsbu = RSBU_CW(layer_parameter_list[3][len(layer_parameter_list[3])-1])

        self.pool = nn.MaxPool1d(kernel_size=3, stride=2)
        out_put_channel_numebr = 0
        new_layer_parameter_list = []
        for final_layer_parameters in self.layer_parameter_list[0][-1]:
            out_put_channel_numebr = out_put_channel_numebr + final_layer_parameters[1]
        for i in range(0, 4):
            new_layer_parameter_list.append(layer_parameter_list_input_change(layer_parameter_list[i], out_put_channel_numebr))
        
        self.net_1 = Res_OS_layer(layer_parameter_list[0], out_put_channel_numebr)
        
        self.net_list = []
        for i in range(self.n_layers-1):
            temp_layer = Res_OS_layer(new_layer_parameter_list[i+1], out_put_channel_numebr)
            pool = nn.MaxPool1d(kernel_size=3, stride=2)
            self.net_list.append(temp_layer)
            self.net_list.append(pool)

        self.net = nn.Sequential(*self.net_list)
        
        self.averagepool = nn.AdaptiveAvgPool1d(1)
        self.hidden = nn.Linear(out_put_channel_numebr, embd_dim)

        self.loss_f = loss_f
        if loss_f == 'softmax':
            self.hidden = nn.Linear(out_put_channel_numebr, n_class)
            self.loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        if loss_f == 'cac':
            self.layer3 = nn.Linear(out_put_channel_numebr, n_class)
            self.loss = CACLoss(n_classes=n_class, alpha=10)
        if loss_f == 'gcpl':
            self.hidden = nn.Linear(out_put_channel_numebr, embd_dim)
            self.loss = gcpl_loss(n_classes=n_class, feat_dim=embd_dim)
        if loss_f == 'kpf':
            self.hidden = nn.Linear(out_put_channel_numebr, embd_dim)
            self.loss = KPFLoss(n_classes=n_class, feat_dim=embd_dim)
        if loss_f == 'rpl':
            self.hidden = nn.Linear(out_put_channel_numebr, embd_dim)
            self.loss = RPLoss(n_classes=n_class, feat_dim=embd_dim)
        if loss_f == 'arpl':
            self.hidden = nn.Linear(out_put_channel_numebr, embd_dim)
            self.loss = ARPLoss(n_classes=n_class, feat_dim=embd_dim)
        if loss_f == 'slcpl':
            self.hidden = nn.Linear(out_put_channel_numebr, embd_dim)
            self.loss = SLCPLoss(n_classes=n_class, feat_dim=embd_dim)
        if loss_f == 'ring':
            self.hidden = nn.Linear(out_put_channel_numebr, embd_dim)
            self.fc1 = nn.Linear(embd_dim, n_class)
            self.loss = RingLoss()

    def forward(self, X, labels=None, para_lambda=None):
        temp = self.net_1(X)
        temp = self.rsbu(temp)
        temp = self.pool(temp)
        temp = self.net(temp)
        X = self.averagepool(temp)
        X = X.squeeze_(-1)
        x = self.hidden(X)

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

        if self.loss_f == 'pbo':
            out = self.loss(x, labels, para_lambda)
            return out

        out = self.loss(x, labels)
        return out
