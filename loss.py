import torch
import torch.nn as nn
import torch.nn.functional as F
from Dist import Dist
from torch.nn.parameter import Parameter


class gcpl_loss(torch.nn.Module):
    def __init__(self, n_classes=4, feat_dim=12, init_weight=True):
        super(gcpl_loss, self).__init__()
        self.n_classes = n_classes                                      # 类别数量
        self.feat_dim = feat_dim                                        # 特征维度
        self.centers = nn.Parameter(torch.randn(self.feat_dim, self.n_classes).cuda(), requires_grad=True)
        self.loss_base = nn.CrossEntropyLoss(label_smoothing=0.1)
        if init_weight:
            self.__init_weight()

    def __init_weight(self):
        nn.init.kaiming_normal_(self.centers)

    def forward(self, x, labels=None, para_lambda=0):
        features_square = torch.sum(torch.pow(x, 2), 1, keepdim=True)
        centers_square = torch.sum(torch.pow(self.centers, 2), 0, keepdim=True)
        features_into_centers = 2 * torch.matmul(x, (self.centers))
        dist = features_square + centers_square - features_into_centers           # 样本到原型的距离

        centers_mean = torch.mean(self.centers, dim=1)                            # 原型中心，Tensor:(16,)
        centers_mean = centers_mean.unsqueeze(1)                                  # 原型中心，Tensor:(16, 1)
        w_lmcl = self.centers - centers_mean                                      # 中心向量 Tensor:(16, 4)

        centers = self.centers.transpose(0, 1)
        distance = -dist
        weights = w_lmcl.transpose(0, 1)

        if labels is None:
            return {'logits': distance,
                    'features': x}

        loss_d = self.loss_base(distance, labels)
        loss_reg = regularization(x, centers, labels)

        n = labels.shape[0]  # batch
        similarity_matrix = F.cosine_similarity(x.unsqueeze(1), weights.unsqueeze(0), dim=2)
        similarity_matrix = torch.exp(similarity_matrix/0.2)
        # 根据label选择分子, 使用 gather 方法从每行中获取 label 对应位置的元素作为分子
        fenzi = similarity_matrix.gather(1, labels.view(-1, 1))  # (32, 1) 每行从 label 指定的列中取值作为分子
        # 生成掩码，排除label对应的位置
        mask = torch.ones_like(similarity_matrix, dtype=torch.bool)  # 初始化全True掩码
        mask.scatter_(1, labels.view(-1, 1), False)  # 在每行将 label 对应的位置设置为 False
        # 使用掩码选择非label位置的数据，并求和
        fenmu = (similarity_matrix * mask).sum(dim=1, keepdim=True)  # 按行求和，排除label对应位置的元素
        result = fenzi / (fenmu + fenzi)
        '''
        由于result中，存在0数值，那么在求-log的时候会出错。这时候，我们就将result里面为0的地方
        全部加上1，然后再去求loss矩阵的值，那么-log1 = 0 ，就是我们想要的。
        '''
        mask_ = torch.eye(n, 1)
        if similarity_matrix.device != mask_.device:
            mask_ = mask_.to(similarity_matrix.device)  # 确保y在与x相同的设备上
        result = result + mask_

        log_result = -torch.log(result)
        loss_sur = torch.sum(log_result)/n

        loss = loss_d + 0.001 * loss_reg.item() + para_lambda * loss_sur

        return {'logits': distance,
                'loss': loss}


def regularization(features, centers, labels):
    distance = (features - torch.t(centers.transpose(0, 1))[labels])
    distance = torch.sum(torch.pow(distance, 2), 1, keepdim=True)
    distance = (torch.sum(distance, 0, keepdim=True)) / features.shape[0]
    return distance


class KPFLoss(nn.CrossEntropyLoss):
    def __init__(self, n_classes=4, feat_dim=12):
        super(KPFLoss, self).__init__()
        self.weight_pl = float(0.1)
        self.temp = 1.
        self.Dist = Dist(num_classes=n_classes, feat_dim=feat_dim)
        self.points = self.Dist.centers
        self.radius = nn.Parameter(torch.Tensor(1))
        self.radius.data.fill_(0)
        self.margin_loss = nn.MarginRankingLoss(margin=1.0)

    def forward(self, x, labels=None):
        dist_dot_p = self.Dist(x, center=self.points, metric='dot')
        dist_l2_p = self.Dist(x, center=self.points)
        dist = dist_l2_p - dist_dot_p

        logits = F.softmax(-dist, dim=1)
        if labels is None:
            return {'logits': logits,
                    'features': x}
        loss = F.cross_entropy(-dist / self.temp, labels)

        center_batch = self.points[labels, :]
        _dis_known = (x - center_batch).pow(2).mean(1)
        target = torch.ones(_dis_known.size()).cuda()
        loss_r = self.margin_loss(self.radius, _dis_known, target)
        loss = loss + self.weight_pl * loss_r

        return {'logits': logits,
                'loss': loss}


class ARPLoss(nn.CrossEntropyLoss):
    def __init__(self, n_classes=4, feat_dim=12):
        super(ARPLoss, self).__init__()
        self.use_gpu = 'cuda:0'
        self.weight_pl = float(0.1)
        self.temp = 1.
        self.Dist = Dist(num_classes=n_classes, feat_dim=feat_dim)
        self.points = self.Dist.centers
        self.radius = nn.Parameter(torch.Tensor(1))
        self.radius.data.fill_(0)
        self.margin_loss = nn.MarginRankingLoss(margin=1.0)

    def forward(self, x, labels=None):
        dist_dot_p = self.Dist(x, center=self.points, metric='dot')
        dist_l2_p = self.Dist(x, center=self.points)
        logits = dist_l2_p - dist_dot_p

        if labels is None:
            return {'logits': logits,
                    'features': x}
        loss = F.cross_entropy(logits / self.temp, labels)

        center_batch = self.points[labels, :]
        _dis_known = (x - center_batch).pow(2).mean(1)
        target = torch.ones(_dis_known.size()).cuda()
        loss_r = self.margin_loss(self.radius, _dis_known, target)

        loss = loss + self.weight_pl * loss_r

        return {'logits': logits,
                'loss': loss}

    def fake_loss(self, x):
        logits = self.Dist(x, center=self.points)
        prob = F.softmax(logits, dim=1)
        loss = (prob * torch.log(prob)).sum(1).mean().exp()

        return loss


class SLCPLoss(nn.CrossEntropyLoss):
    def __init__(self, n_classes=4, feat_dim=12):
        super(SLCPLoss, self).__init__()
        self.weight_pl = float(0.1)
        self.Dist = Dist(num_classes=n_classes, feat_dim=feat_dim)
        self.points = self.Dist.centers

    def forward(self, x, labels=None):
        # labels 只有在训练时需要，测试时不需要该参数
        dist_l2_p = self.Dist(x, center=self.points)
        logits = F.softmax(-dist_l2_p, dim=1)
        if labels is None:
            return {'logits': logits,
                    'features': x}
        loss_main = F.cross_entropy(-dist_l2_p, labels)

        center_batch = self.points[labels, :]
        loss_r = F.mse_loss(x, center_batch) / 2

        o_center = self.points.mean(0)
        l_ = (self.points - o_center).pow(2).mean(1)
        # loss_outer = torch.exp(-l_.mean(0))
        loss_outer_std = torch.std(l_)

        loss = loss_main + self.weight_pl * loss_r + loss_outer_std
        return {'logits': logits,
                'loss': loss}


class RingLoss(nn.Module):
    def __init__(self, **options):
        type = 'auto'
        loss_weight = 1.0

        super(RingLoss, self).__init__()
        self.radius = Parameter(torch.Tensor(1))
        self.radius.data.fill_(-1)
        self.loss_weight = loss_weight
        self.type = type

    def forward(self, x, y, labels=None):
        ## calculate softmax loss
        logits = y
        if labels is None:
            return {'logits': logits,
                    'features': x}
        softmax_loss = F.cross_entropy(logits, labels)
        ## calculate ringloss
        x = x.pow(2).sum(dim=1).pow(0.5)
        if self.radius.data[0] < 0: # Initialize the radius with the mean feature norm of first iteration
            self.radius.data.fill_(x.mean().item())
        if self.type == 'l1': # Smooth L1 Loss
            loss1 = F.smooth_l1_loss(x, self.radius.expand_as(x)).mul_(self.loss_weight)
            loss2 = F.smooth_l1_loss(self.radius.expand_as(x), x).mul_(self.loss_weight)
            ringloss = loss1 + loss2
        elif self.type == 'auto': # Divide the L2 Loss by the feature's own norm
            diff = x.sub(self.radius.expand_as(x)) / (x.mean().detach().clamp(min=0.5))
            diff_sq = torch.pow(torch.abs(diff), 2).mean()
            ringloss = diff_sq.mul_(self.loss_weight)
        else: # L2 Loss, if not specified
            diff = x.sub(self.radius.expand_as(x))
            diff_sq = torch.pow(torch.abs(diff), 2).mean()
            ringloss = diff_sq.mul_(self.loss_weight)

        loss = softmax_loss + ringloss

        return {'logits': logits,
                'loss': loss}

class CACLoss(nn.CrossEntropyLoss):
    def __init__(self, n_classes=6, alpha=10):
        super(CACLoss, self).__init__()
        self.num_classes = n_classes
        self.alpha = alpha if alpha is not None else 10.0
        self.means = torch.diag(torch.Tensor([self.alpha for _ in range(self.num_classes)]))
        self.anchors = nn.Parameter(self.means.double(), requires_grad=False)

    def forward(self, x, labels=None):
        n = x.size(0)           # Batch size
        m = self.num_classes    # Number of classes
        d = self.num_classes    # Dimension (same as number of classes)
        features = x

        x = x.unsqueeze(1).expand(n, m, d).double()
        anchors = self.anchors.unsqueeze(0).expand(n, m, d)
        distances = torch.norm(x - anchors, 2, 2)
        if labels is None:
            return {'logits': -distances,
                    'features': features}
        true = torch.gather(distances, 1, labels.view(-1, 1)).view(-1)

        # Generate indices of non-ground truth classes
        non_gt = torch.Tensor(
            [[i for i in range(m) if labels[x] != i] for x in range(len(distances))]
        ).long().to(x.device)
        others = torch.gather(distances, 1, non_gt)

        # Mean of true distances
        anchor = torch.mean(true)

        tuplet = torch.exp(-others + true.unsqueeze(1))
        tuplet = torch.mean(torch.log(1 + torch.sum(tuplet, dim=1)))

        loss = 0.1 * anchor + tuplet
        return {'logits': -distances,
                'loss': loss}


class RPLoss(nn.CrossEntropyLoss):
    def __init__(self, n_classes=4, feat_dim=12):
        super(RPLoss, self).__init__()
        self.weight_pl = float(0.1)
        self.temp = 1.
        self.Dist = Dist(num_classes=n_classes, feat_dim=feat_dim, num_centers=1)
        self.radius = 1

        self.radius = nn.Parameter(torch.Tensor(self.radius))
        self.radius.data.fill_(0)

    def forward(self, x, labels=None):
        dist = self.Dist(x)
        logits = F.softmax(dist, dim=1)
        if labels is None:
            return {'logits': logits,
                    'features': x}
        loss = F.cross_entropy(dist / self.temp, labels)
        center_batch = self.Dist.centers[labels, :]
        _dis = (x - center_batch).pow(2).mean(1)
        loss_r = F.mse_loss(_dis, self.radius)
        loss = loss + self.weight_pl * loss_r

        return {'logits': logits,
                'loss': loss}
