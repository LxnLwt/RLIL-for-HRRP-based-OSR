'''
Author: chenpirate chensy293@mail2.sysu.edu.cn
Date: 2023-02-21 11:10:59
LastEditors: chenpirate chensy293@mail2.sysu.edu.cn
LastEditTime: 2023-09-12 12:45:55
FilePath: /resnetV2/utils/dataset.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

from os import cpu_count
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder


# 自定义序列化方式
class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()


class HrrpDataset(Dataset):
    def __init__(self, dataset_path, json_path):
        self.json_path = json_path
        df = pd.read_csv(dataset_path, header=None)
        datas = df.iloc[:, :-1].values
        datas = x_norm(datas)
        datas = torch.from_numpy(datas)
        self.datas = torch.unsqueeze(datas.float(), dim=1)

        self.labels = df.iloc[:, -1]
        labels_json = open(self.json_path, 'r')
        self.class_indict = json.load(labels_json)
        # print(self.labels[0])
        # print(self.class_indict[str(self.labels[0])])

    def __getitem__(self, idx):                                  
        return self.datas[idx], self.class_indict[str(self.labels[idx])]

    def __len__(self):
        return len(self.datas)


def x_norm(x):
    x_norm = np.linalg.norm(x, ord=None, axis=1, keepdims=False).reshape(-1, 1)
    x = x / x_norm
    return x


def my_dataloader(dataset_path, json_path, shuffle=True, drop_last=True, batch_size=1, nw=cpu_count()):
    dataset = HrrpDataset(dataset_path, json_path)
    # print(dataset.__getitem__(0))

    datasize = len(dataset)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=nw, shuffle=shuffle,
                             drop_last=drop_last)
    return data_loader, datasize


if __name__ == '__main__':
    data_loader, train_size = my_dataloader("data/9.12/testdata_013489.csv", "class_indices.json", batch_size=32, nw=16)
    print("using {} HRRP datas for training.".format(train_size))