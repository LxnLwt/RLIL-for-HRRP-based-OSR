'''
Author: chenpirate chensy293@mail2.sysu.edu.cn
Date: 2023-02-21 11:10:59
LastEditors: chenpirate chensy293@mail2.sysu.edu.cn
LastEditTime: 2023-09-12 13:43:59
FilePath: /resnetV2/utils/predict.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch


# 有一个训练好的分类网络，并加载好了权重
# 分类网络的输出只是最后一层全连接层的输出，因此需要经过softmax函数，并取最大的概率索引，作为预测类别
# 使用这个网络，对一条信号进行分类，返回预测类别以及分类网络的输出
# 实现这个函数


def predict(hrrp, model, device="cpu"):
    model.eval()
    with torch.no_grad():
        hrrp = torch.from_numpy(hrrp).float().to(device)
        if hrrp.ndim != 3:
            hrrp = hrrp.reshape(1, 1, hrrp.size()[-1])
        out_ = model(hrrp)
        # cos_value = out_['cos_value']
        distance = torch.abs(out_['logits'])
        # centers = out_['centers']
        features = out_['features']

        prob_distance = torch.nn.functional.softmax(distance, dim=1)
        prob1 = prob_distance.cpu().numpy()

        pred = prob1.argmin(axis=1)
        return pred[0], prob_distance, distance, features


def predict_s(hrrp, model, device="cpu"):
    model.eval()
    with torch.no_grad():
        hrrp = torch.from_numpy(hrrp).float().to(device)
        if hrrp.ndim != 3:
            hrrp = hrrp.reshape(1, 1, hrrp.size()[-1])
        out_ = model(hrrp)
        logits = out_['logits']

        prob_logits = torch.nn.functional.softmax(logits, dim=1)
        prob1 = prob_logits.cpu().numpy()

        pred = prob1.argmax(axis=1)
        return pred[0], prob_logits, logits
