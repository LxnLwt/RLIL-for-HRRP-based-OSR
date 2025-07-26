import sys
import torch.nn as nn
from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc, accuracy_score
from LIL import L_criterion, L_data
import torch.nn.functional as F


def train_one_epoch(model, optimizer, data_loader, device, epoch, para_lambda):
    """
    一个epoch里要做的操作
    Args:
    :param label_smothing: 标签平滑
    :param model: 模型
    :param optimizer:优化器
    :param data_loader: 训练数据加载器
    :param device: cpu 还是gpu
    :param epoch: 第几个epoch
    :return:这个epoch训练损失，训练精度


        lam: 0
        label_smoothing: 标签平滑
    """
    model.train()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()
    loss_f = nn.CrossEntropyLoss(label_smoothing=0.1)

    sample_num = 0

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        HRRP, labels = data
        HRRP = HRRP.to(device)
        labels = labels.long().to(device)

        i_x, y_a, y_b, lam = L_data(HRRP, labels, 10, 10)

        out = model(HRRP, labels)
        logits = out['logits']
        loss = out['loss']

        out_lil = model(i_x, labels)
        logits_lil = out_lil['logits']
        loss_lil = L_criterion(loss_f, logits_lil, y_a, y_b, lam)

        loss = loss + para_lambda * loss_lil

        pred_classes = torch.max(logits, dim=1)[1]
        accu_num += torch.eq(pred_classes[:labels.shape[0]], labels).sum()
        loss.backward()
        accu_loss += loss.detach()
        sample_num += HRRP.shape[0]
        data_loader.desc = "[train epoch {}] train loss: {:.3f}, acc: {:.2f}%".format(epoch + 1,
                                                                                      accu_loss.item() / (step + 1),
                                                                                      (accu_num.item() / sample_num)
                                                                                      * 100)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    """
    验证评估函数
    :param model: 模型
    :param data_loader: 验证加载器
    :param device: cpu 还是gpu
    :param epoch: 第几个epoch
    :return: 这个epoch验证损失，验证精度

    Args:
        label_smoothing: 标签平滑
        lam: center loss: 的权值
    """
    model.eval()
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    step = None
    with torch.no_grad():
        for step, data in enumerate(data_loader):
            hrrp, labels = data
            sample_num += hrrp.shape[0]

            out = model(hrrp.to(device))
            pred = out['logits']
            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels.long().to(device)).sum()
            data_loader.desc = "[valid epoch {}] acc: {:.2f}%".format(
                epoch + 1, (accu_num.item() / sample_num) * 100)

    return accu_num.item() / sample_num


@torch.no_grad()
def evaluate_auroc_dist(model, data_loader_known, data_loader_unknown, data_loader_known_oscr, device):
    """
    验证评估函数
    :param model: 模型
    :param data_loader: 验证加载器
    :param device: cpu 还是gpu
    :param epoch: 第几个epoch
    :return: 这个epoch验证损失，验证精度

    Args:
        label_smoothing: 标签平滑
        lam: center loss: 的权值
    """
    model.eval()
    sample_num = 0

    step = None
    logits_list = []
    labels_list = []
    logits_known_list = []
    labels_known_list = []
    logits_unknown_list = []
    labels_unknown_list = []
    labels_known_oscr_list = []
    pre_labels_known_oscr_list = []
    with torch.no_grad():
        for step, data in enumerate(data_loader_known):
            hrrp, labels = data
            sample_num += hrrp.shape[0]
            out = model(hrrp.to(device))
            pred = out['logits']
            logits = torch.nn.functional.softmax(pred, dim=1)
            logit, _ = torch.max(logits, dim=1)
            logits_known_list.append(logit)
            labels_known_list.append(labels)
        for step, data in enumerate(data_loader_unknown):
            hrrp, labels = data
            sample_num += hrrp.shape[0]
            out = model(hrrp.to(device))
            pred = out['logits']
            logits = torch.nn.functional.softmax(pred, dim=1)
            logit, _ = torch.max(logits, dim=1)
            logits_unknown_list.append(logit)
            labels_unknown_list.append(labels)
        for step, data in enumerate(data_loader_known_oscr):
            hrrp, labels = data
            sample_num += hrrp.shape[0]
            out = model(hrrp.to(device))
            max_distance = out['logits']
            _, index = torch.max(max_distance, dim=1)
            pre_labels_known_oscr_list.append(index)
            labels_known_oscr_list.append(labels)
        logits_list = logits_known_list + logits_unknown_list
        labels_list = labels_known_list + labels_unknown_list
        logits_list = torch.concat(logits_list, dim=0)
        labels_list = torch.concat(labels_list, dim=0)
        pre_y = logits_list.cpu().numpy()
        y = labels_list.cpu().numpy()

        if np.isnan(pre_y).any() or np.isinf(pre_y).any():
            print("Input contains Nan or Inf!")
            # 删除无效值
            mask = np.isnan(pre_y) | np.isinf(pre_y)
            pre_y = pre_y[~mask]
            y = y[~mask]

        # 计算AUROC
        fpr, tpr, thresholds = roc_curve(y, pre_y, pos_label=1)  # 计算 ROC 曲线的参数
        auc_score_roc = roc_auc_score(y, pre_y)  # 计算ROC曲线下的面积 AUC
        # np.savetxt('./txt/oscr/fpr_ring.txt', fpr)
        # np.savetxt('./txt/oscr/tpr_ring.txt', tpr)

        # 计算OSCR
        pre_prob_known = torch.concat(logits_known_list, dim=0).cpu().numpy()             # 已知类置信度
        pred_array_known = torch.concat(pre_labels_known_oscr_list, dim=0).cpu().numpy()  # 已知类预测标签
        labels_known = torch.concat(labels_known_oscr_list, dim=0).cpu().numpy()          # 已知类真实标签

        if np.isnan(pre_prob_known).any() or np.isinf(pre_prob_known).any():
            print("Input contains Nan or Inf!")
            # 删除无效值
            mask = np.isnan(pre_prob_known) | np.isinf(pre_prob_known)
            pre_prob_known = pre_prob_known[~mask]
            pred_array_known = pred_array_known[~mask]
            labels_known = labels_known[~mask]

        # 计算OSCR
        m_x1 = np.zeros(len(pred_array_known))
        m_x1[pred_array_known == labels_known] = 1
        ccr = []
        num = len(pred_array_known)
        for threshold in thresholds:
            TPN = pre_prob_known >= threshold
            CCR = np.sum(TPN * m_x1)
            ccr.append(CCR)
        ccr = np.array(ccr) / num
        # np.savetxt('./txt/oscr/ccr_ring.txt', ccr)
        oscr_score = auc(fpr, ccr)

    return auc_score_roc, oscr_score
