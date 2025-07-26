import numpy as np
from utils.predict import predict, predict_s
from model import resnet18, AlexNet
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.font_manager import FontProperties
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


if __name__ == "__main__":
    np.random.seed(7)
    # model_weight_path1 = "weight/alexnet_gcpl_data4_l2×0.05_mixup×0_pwc×0.pth"
    # # model_weight_path1 = "weight/alexnet_gcpl_data4_l2×0_mixup×0_pwc×0.pth"
    # data_path = "data/9.12/testdata_sim_6kc_27.csv"
    # unknown_path = "data/9.12/unknown_sim_6kc_27.csv"
    #
    # # 加载网络
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = AlexNet(loss_f='gcpl', num_classes=6).to(device)
    # model.load_state_dict(torch.load(model_weight_path1, map_location=device))
    #
    # # 读取测试数据
    # df1 = pd.read_csv(data_path)                             # 读取库外测试数据
    # x1 = df1.iloc[:, :-1].values
    # y1 = df1.iloc[:, -1].values
    # x1 = x_norm(x1)
    # df2 = pd.read_csv(unknown_path)                  # 读取库内测试数据
    # x2 = df2.iloc[:, :-1].values
    # y2 = df2.iloc[:, -1].values
    # x2 = x_norm(x2)
    # x = np.concatenate((x1, x2), axis=0)
    # y = np.concatenate((y1, y2), axis=0)
    #
    # logits_norm = []
    # for i in range(len(x1)):
    #     pred, pro_logits, logits = predict_s(x1[i], model, device)
    #     norms = torch.norm(logits, dim=1)
    #     logits_norm.append(norms.item())
    #
    # for i in range(len(x2)):
    #     pred, pro_logits, logits = predict_s(x2[i], model, device)
    #     norms = torch.norm(logits, dim=1)
    #     logits_norm.append(norms.item())
    #
    # logits_norm_known = np.array(logits_norm)
    # mean_logits_norm = np.mean(logits_norm_known)
    # print("norm:", mean_logits_norm)
    #
    # np.savetxt('./txt/histplot/norm_logits_regularization.txt', logits_norm_known)
    # # np.savetxt('./txt/histplot/norm_logits.txt', logits_norm_known)

    # 加载数据
    norm_logits_regularization = np.loadtxt('./txt/histplot/norm_logits_regularization.txt')
    norm_logits = np.loadtxt('./txt/histplot/norm_logits.txt')

    # 计算均值
    mean_norm = np.mean(norm_logits)
    mean_norm_r = np.mean(norm_logits_regularization)
    print(f"Mean of norm_parameters: {mean_norm:.6f}")
    print(f"Mean of norm_parameters_r: {mean_norm_r:.6f}")

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

    # histplot作图, '#9FC9AD'(绿色)
    ax1 = sns.histplot(norm_logits, kde=False, bins=70, stat='probability', color='#9BAED1',
                       edgecolor='white', label=f'Mean (No Constraint): {mean_norm:.3f}', linewidth=0, ax=axes)
    ax2 = sns.histplot(norm_logits_regularization, kde=False, bins=70, stat='probability', color='#E3B7A2',
                       edgecolor='white', label=f'Mean (Add Constraint): {mean_norm_r:.3f}', linewidth=0, ax=axes)

    # 绘制均值竖线
    axes.axvline(mean_norm, color='#9BAED1', linestyle='--', linewidth=2,
                 label='')
    axes.axvline(mean_norm_r, color='#E3B7A2', linestyle='--', linewidth=2,
                 label='')
    # 在合适位置显示均值文本（避免遮挡，可自调 y 坐标）
    ymax = axes.get_ylim()[1]
    axes.text(mean_norm, ymax * 0.92, f"{mean_norm:.3f}",
              color='#9BAED1', fontsize=26, fontweight='bold',
              ha='center', va='bottom', rotation=0, fontname='Times New Roman')
    axes.text(mean_norm_r, ymax * 0.77, f"{mean_norm_r:.3f}",
              color='#E3B7A2', fontsize=26, fontweight='bold',
              ha='center', va='bottom', rotation=0, fontname='Times New Roman')
    font_properties = FontProperties(family='Times New Roman', size=26)
    legend = plt.legend(loc='upper right', prop=font_properties, frameon=True)
    legend.get_frame().set_edgecolor('lightgrey')

    # 设置图例
    font_properties = FontProperties(family='Times New Roman', size=26)
    legend = plt.legend(loc='upper right', prop=font_properties, frameon=True)
    legend.get_frame().set_edgecolor('lightgrey')
    # 设置网格线
    axes.grid(True, which='both', color='white', linestyle='-', linewidth=1.5)
    # 设置边框颜色为白色
    for spine in axes.spines.values():
        spine.set_edgecolor('#E9E9F1')
    # 取消刻度线但保留坐标数字
    axes.tick_params(axis='both', which='both', length=0, colors='black')

    # 设置X轴
    axes.set_xlim(0, 30)
    axes.set_xticks(np.arange(0, 30, 5))
    axes.tick_params(axis='x', labelsize=26, labelcolor='black')
    for label in axes.get_xticklabels():
        label.set_fontname('Times New Roman')
    # 设置y轴
    ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1]) + 0.02
    axes.set_ylim(0, ymax)
    axes.set_yticks(np.arange(0, ymax, 0.02))
    axes.tick_params(axis='y', labelsize=26, labelcolor='black')
    for label in axes.get_yticklabels():
        label.set_fontname('Times New Roman')

    # # 创建放大镜子图
    # axins = inset_axes(axes, width="40%", height="40%", loc='center', bbox_to_anchor=(-0.15, 0.0, 1, 1),
    #                    bbox_transform=axes.transAxes)
    # axins.set_facecolor('#E9E9F1')
    # # 在放大镜子图中重新绘制直方图
    # sns.histplot(norm_logits, kde=False, bins=40, stat='probability', color='#81B3A9',
    #              edgecolor='white', ax=axins)
    # sns.histplot(norm_logits_regularization, kde=False, bins=40, stat='probability', color=palette.colors[13],
    #              edgecolor='white', ax=axins)
    # # 设置放大镜子图的X轴和Y轴范围
    # x1, x2, y1, y2 = 0.1, 1.5, 0, 0.005
    # axins.set_xlim(x1, x2)
    # axins.set_ylim(y1, y2)
    # # 设置放大镜子图的Y轴刻度间隔
    # axins.yaxis.set_major_locator(MultipleLocator(0.001))
    # axins.xaxis.set_major_locator(MultipleLocator(0.3))
    # # 去掉放大镜子图的刻度线，但保留刻度标签，字体为新罗马
    # axins.tick_params(axis='both', which='both', length=0, colors='black', labelsize=22, labelrotation=0)
    # for label in axins.get_xticklabels() + axins.get_yticklabels():
    #     label.set_fontname('Times New Roman')
    # # 设置放大镜子图的X和Y轴标签为空字符串以去除它们
    # axins.set_xlabel("", size=22)
    # axins.set_ylabel("", size=22)
    # # 连接主图和放大镜子图
    # mark_inset(axes, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    # 紧密排版
    plt.tight_layout()
    # 标签大小
    plt.xlabel("Norm of Logits", size=26)
    plt.ylabel("Frequency", size=26)
    # 刻度大小
    plt.tick_params(labelsize=26)
    # 保存图片
    output_dir = './pictures/'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "Figure5b.pdf"), bbox_inches='tight')

    # 显示图表
    plt.show()

