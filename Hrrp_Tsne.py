import torch
import json
from utils.predict import predict, predict_s
from FSBN import FSBN
from model import AlexNet, resnet18
from utils.tSNE2D import *


if __name__ == "__main__":
    np.random.seed(7)
    model_weight_path1 = "weight/alexnet_ring_data2_l2×0_mixup×0_pwc×0.pth"
    data_path = "data/9.12/testdata_sim_5kc.csv"
    json_label_path = "json/tsne_indices_2.json"
    unknown_targets_path = "data/9.12/unknown_sim_5kc.csv"

    # 加载网络
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AlexNet(loss_f='ring', num_classes=5).to(device)
    model.load_state_dict(torch.load(model_weight_path1, map_location=device))

    # 加载数据并归一化
    df1 = pd.read_csv(data_path)
    x1 = df1.iloc[:, :-1].values
    y1 = df1.iloc[:, -1].values
    x1 = x_norm(x1)
    df2 = pd.read_csv(unknown_targets_path)
    x2 = df2.iloc[:, :-1].values
    y2 = df2.iloc[:, -1].values
    x2 = x_norm(x2)
    x = np.concatenate((x1, x2), axis=0)
    y = np.concatenate((y1, y2), axis=0)
    # 标签转换
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)
    y = y.astype(str)
    for i in range(len(y)):
        y[i] = class_indict[str(y[i])]
    # 特征提取
    T_list1 = []
    for i in range(len(x)):
        _, _, features = predict_s(x[i], model, device)
        # 存储用于绘制散点图的数据
        T_list1.append(features[0].tolist())
    T_array1 = np.array(T_list1)
    S_array1 = np.concatenate((y.reshape(-1, 1), T_array1), axis=1)
    # 绘图
    tSNE = tSNE()
    tSNE.visual_dataset(T_array1, y, f'RowData\n')

