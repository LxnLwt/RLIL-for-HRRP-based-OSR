import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
from model import AlexNet, resnet18
import matplotlib.colors as mcolors

if __name__ == "__main__":
    # 固定随机种子
    np.random.seed(7)
    model_weight_path1 = "weight/alexnet_gcpl_data3_l2×0.2_mixup×0.pth"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AlexNet(loss_f='gcpl', num_classes=5).to(device)
    model.load_state_dict(torch.load(model_weight_path1, map_location=device))

    # 1. 读取你刚刚生成的噪声数据
    df_noise = pd.read_csv('data/9.12/unknown_noise_1to20.csv', header=None)
    x_noise = df_noise.iloc[:, :-1].values  # (N, L)
    var_arr = df_noise.iloc[:, -1].values  # (N,) 方差在最后一列

    # 2. 推理，得到 cross-entropy（保持和你之前的推理代码一致）
    model.eval()
    x_noise_tensor = torch.from_numpy(x_noise).float().to(device)
    if x_noise_tensor.ndim != 3:
        x_noise_tensor = x_noise_tensor.unsqueeze(1)  # (N, 1, L)
    with torch.no_grad():
        out = model(x_noise_tensor)
        prob = F.softmax(out['logits'], dim=1).cpu().numpy()
        pred_idx = np.argmax(prob, axis=1)
        onehot_pred = np.zeros_like(prob)
        onehot_pred[np.arange(prob.shape[0]), pred_idx] = 1
        cross_entropy = -np.sum(onehot_pred * np.log(np.clip(prob, 1e-10, 1.)), axis=1)

    # plt.figure(figsize=(10, 8))
    plt.figure(figsize=(10, 7))
    plt.hist2d(var_arr, cross_entropy, bins=[96, 60], cmap='coolwarm', norm=mcolors.LogNorm())  # 纵向分15格
    cbar = plt.colorbar()
    cbar.set_label('Count', fontsize=20, fontname='Times New Roman')

    plt.xlabel('Noise Variance', fontsize=20, fontweight='bold', fontname='Times New Roman')
    plt.ylabel('Cross-Entropy', fontsize=20, fontweight='bold', fontname='Times New Roman')
    plt.xticks(np.arange(1, 21, 2), fontsize=18, fontname='Times New Roman')
    plt.yticks(fontsize=18, fontname='Times New Roman')
    plt.tight_layout()
    # plt.savefig('./pictures/unknown_noise_variance_heatmap.png', bbox_inches='tight')
    plt.savefig('./pictures/Figure6b.pdf', bbox_inches='tight')
    plt.show()


