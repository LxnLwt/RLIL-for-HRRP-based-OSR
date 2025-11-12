from utils.OpenMax import *
from model import AlexNet, resnet18
from utils.tSNE2D import *
import matplotlib.pyplot as plt
from torch.nn import functional as F
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
import os


if __name__ == "__main__":
    np.random.seed(7)
    # model_weight_path1 = "weight/alexnet_gcpl_data4_l2×0.05_mixup×0_pwc×0.pth"
    model_weight_path1 = "weight/alexnet_gcpl_data4_l2×0_mixup×0_pwc×0.pth"
    data_path = "data/testdata_sim_6kc_27.csv"
    unknown_path = "data/unknown_sim_6kc_27.csv"

    # 加载网络
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AlexNet(loss_f='gcpl', num_classes=6).to(device)
    model.load_state_dict(torch.load(model_weight_path1, map_location=device))

    # 加载数据并归一化
    df1 = pd.read_csv(data_path)
    x1 = df1.iloc[:, :-1].values
    x1 = x_norm(x1)
    # >>> 2抽1采样
    x1 = x1[::10]
    num_1 = x1.shape[0]

    df2 = pd.read_csv(unknown_path)
    x2 = df2.iloc[:, :-1].values
    x2 = x_norm(x2)
    num_2 = x2.shape[0]

    model.eval()
    # 推理
    x1_tensor = torch.from_numpy(x1).float().to(device)
    with torch.no_grad():
        if x1_tensor.ndim != 3:
            x1_tensor = x1_tensor.reshape(num_1, 1, x1_tensor.size()[-1])
        out = model(x1_tensor)
        prob = F.softmax(out['logits'], dim=1).cpu().numpy()  # (N_mix, n_classes)
        pred_idx = np.argmax(prob, axis=1)
        onehot_pred = np.zeros_like(prob)
        onehot_pred[np.arange(prob.shape[0]), pred_idx] = 1
        cross_entropy_1 = -np.sum(onehot_pred * np.log(np.clip(prob, 1e-10, 1.)), axis=1)  # (N_mix,)

    model.eval()
    # 推理
    x2_tensor = torch.from_numpy(x2).float().to(device)
    with torch.no_grad():
        if x2_tensor.ndim != 3:
            x2_tensor = x2_tensor.reshape(num_2, 1, x1_tensor.size()[-1])
        out_2 = model(x2_tensor)
        prob_2 = F.softmax(out_2['logits'], dim=1).cpu().numpy()  # (N_mix, n_classes)
        pred_idx_2 = np.argmax(prob_2, axis=1)
        onehot_pred_2 = np.zeros_like(prob_2)
        onehot_pred_2[np.arange(prob_2.shape[0]), pred_idx_2] = 1
        cross_entropy_2 = -np.sum(onehot_pred_2 * np.log(np.clip(prob_2, 1e-10, 1.)), axis=1)  # (N_mix,)

    mean_entropy_1 = np.mean(cross_entropy_1)
    mean_entropy_2 = np.mean(cross_entropy_2)

    # # 绘制插值后的直方图
    # # 全局设置字体为新罗马
    # plt.rcParams['font.family'] = 'Times New Roman'
    # # 设置画布
    # fig, axes = plt.subplots(1, 1, num="stars", figsize=(9, 6.9))
    # plt.style.use('seaborn-darkgrid')  # 使用 Seaborn 样式
    # palette = plt.get_cmap('tab20c')  # 调色板
    # # 设置背景颜色
    # fig.patch.set_facecolor('white')  # 设置整个图的背景颜色
    # axes.set_facecolor('#E9E9F1')  # 设置子图的背景颜色
    # # 打印调色板中的颜色
    # print(f"{palette.colors} {len(palette.colors)}")
    # # 调用 histplot 作图
    # # 备选颜色'#81B3A9', palette.colors[13], '#9BAED1'
    # ax1 = sns.histplot(cross_entropy_1, kde=False, bins=100, stat='probability', color='#9FC9AD',
    #                    edgecolor='white', label=f'Mean (Known): {mean_entropy_1:.3f}', linewidth=0)
    # ax2 = sns.histplot(cross_entropy_2, kde=False, bins=150, stat='probability', color=palette.colors[16],
    #                    edgecolor='white', label=f'Mean (Unknown): {mean_entropy_2:.3f}', linewidth=0)
    # # 绘制均值竖线
    # axes.axvline(mean_entropy_1, color='#9FC9AD', linestyle='--', linewidth=2,
    #              label='')
    # axes.axvline(mean_entropy_2, color=palette.colors[16], linestyle='--', linewidth=2,
    #              label='')
    # # 在合适位置显示均值文本（避免遮挡，可自调 y 坐标）
    # ymax = axes.get_ylim()[1]
    # axes.text(mean_entropy_1, ymax * 0.6, f"{mean_entropy_1:.3f}",
    #           color='#9FC9AD', fontsize=26, fontweight='bold',
    #           ha='center', va='bottom', rotation=0, fontname='Times New Roman')
    # axes.text(mean_entropy_2, ymax * 0.6, f"{mean_entropy_2:.3f}",
    #           color=palette.colors[16], fontsize=26, fontweight='bold',
    #           ha='center', va='bottom', rotation=0, fontname='Times New Roman')
    # font_properties = FontProperties(family='Times New Roman', size=26)
    # legend = plt.legend(loc='upper right', prop=font_properties, frameon=True)
    # legend.get_frame().set_edgecolor('lightgrey')
    #
    # axes.grid(True, which='both', color='white', linestyle='-', linewidth=1.5)
    # # 设置边框颜色为白色
    # for spine in axes.spines.values():
    #     spine.set_edgecolor('#E9E9F1')
    # # 取消刻度线但保留坐标数字
    # axes.tick_params(axis='both', which='both', length=0, colors='black')
    # # 设置X轴
    # axes.set_xlim(0, 0.8)
    # axes.set_xticks(np.arange(0.1, 0.8, 0.2))
    # axes.tick_params(axis='x', labelsize=26, labelcolor='black')
    # for label in axes.get_xticklabels():
    #     label.set_fontname('Times New Roman')
    # # 设置y轴
    # ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1]) + 0.1
    # axes.set_ylim(0, ymax)
    # axes.set_yticks(np.arange(0, ymax, 0.15))
    # axes.tick_params(axis='y', labelsize=26, labelcolor='black')
    # for label in axes.get_yticklabels():
    #     label.set_fontname('Times New Roman')
    #
    # # 紧密排版
    # plt.tight_layout()
    # # 标签大小
    # plt.xlabel("Entropy", size=26)
    # plt.ylabel("Frequency", size=26)
    # # 刻度大小
    # plt.tick_params(labelsize=26)
    # # 保存图片
    # output_dir = './pictures/'
    # os.makedirs(output_dir, exist_ok=True)
    # plt.savefig('./pictures/Figure5d.pdf', bbox_inches='tight')
    # # 显示图表
    # plt.show()


    # 绘制插值后的直方图
    # 全局设置字体为新罗马
    plt.rcParams['font.family'] = 'Times New Roman'
    # 设置画布
    fig, axes = plt.subplots(1, 1, num="stars", figsize=(9, 6.9))
    plt.style.use('seaborn-darkgrid')  # 使用 Seaborn 样式
    palette = plt.get_cmap('tab20c')  # 调色板
    # 设置背景颜色
    fig.patch.set_facecolor('white')  # 设置整个图的背景颜色
    axes.set_facecolor('#E9E9F1')  # 设置子图的背景颜色
    # 打印调色板中的颜色
    print(f"{palette.colors} {len(palette.colors)}")
    # 调用 histplot 作图
    # 备选颜色'#81B3A9', palette.colors[13], '#9BAED1'
    ax1 = sns.histplot(cross_entropy_1, kde=False, bins=100, stat='probability', color='#9FC9AD',
                       edgecolor='white', label=f'Mean (Known): {mean_entropy_1:.3f}', linewidth=0)
    ax2 = sns.histplot(cross_entropy_2, kde=False, bins=150, stat='probability', color=palette.colors[16],
                       edgecolor='white', label=f'Mean (Unknown): {mean_entropy_2:.3f}', linewidth=0)
    # 绘制均值竖线
    axes.axvline(mean_entropy_1, color='#9FC9AD', linestyle='--', linewidth=2,
                 label='')
    axes.axvline(mean_entropy_2, color=palette.colors[16], linestyle='--', linewidth=2,
                 label='')
    # 在合适位置显示均值文本（避免遮挡，可自调 y 坐标）
    ymax = axes.get_ylim()[1]
    axes.text(mean_entropy_1, ymax * 0.6, f"{mean_entropy_1:.3f}",
              color='#9FC9AD', fontsize=26, fontweight='bold',
              ha='center', va='bottom', rotation=0, fontname='Times New Roman')
    axes.text(mean_entropy_2, ymax * 0.6, f"{mean_entropy_2:.3f}",
              color=palette.colors[16], fontsize=26, fontweight='bold',
              ha='center', va='bottom', rotation=0, fontname='Times New Roman')
    font_properties = FontProperties(family='Times New Roman', size=26)
    legend = plt.legend(loc='upper right', prop=font_properties, frameon=True)
    legend.get_frame().set_edgecolor('lightgrey')

    axes.grid(True, which='both', color='white', linestyle='-', linewidth=1.5)
    # 设置边框颜色为白色
    for spine in axes.spines.values():
        spine.set_edgecolor('#E9E9F1')
    # 取消刻度线但保留坐标数字
    axes.tick_params(axis='both', which='both', length=0, colors='black')
    # 设置X轴
    axes.set_xlim(0, 0.8)
    axes.set_xticks(np.arange(0.1, 0.8, 0.2))
    axes.tick_params(axis='x', labelsize=26, labelcolor='black')
    for label in axes.get_xticklabels():
        label.set_fontname('Times New Roman')
    # 设置y轴
    ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1]) + 0.1
    axes.set_ylim(0, ymax)
    axes.set_yticks(np.arange(0, ymax, 0.2))
    axes.tick_params(axis='y', labelsize=26, labelcolor='black')
    for label in axes.get_yticklabels():
        label.set_fontname('Times New Roman')

    # 紧密排版
    plt.tight_layout()
    # 标签大小
    plt.xlabel("Entropy", size=26)
    plt.ylabel("Frequency", size=26)
    # 刻度大小
    plt.tick_params(labelsize=26)
    # 保存图片
    output_dir = './pictures/'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig('./pictures/Figure5c.pdf', bbox_inches='tight')
    # 显示图表

    plt.show()
