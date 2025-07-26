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
    # # model_weight_path1 = "weight/alexnet_gcpl_data4_l2×0.05_mixup×0_pwc×0.pth"
    # model_weight_path1 = "weight/alexnet_gcpl_data4_l2×0_mixup×0_pwc×0.pth"
    # data_path = "data/9.12/testdata_sim_6kc_27.csv"
    # unknown_path = "data/9.12/unknown_sim_6kc_27.csv"
    #
    # # 加载网络
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = AlexNet(loss_f='gcpl', num_classes=6).to(device)
    # model.load_state_dict(torch.load(model_weight_path1, map_location=device))
    #
    # # 统计所有参数的绝对值
    # param_values = []
    # for param in model.parameters():
    #     if param.requires_grad:
    #         param_values.extend(param.data.cpu().numpy().flatten())
    #
    # param_values = np.array(param_values)
    # param_abs = np.abs(param_values)
    #
    # # np.savetxt('./txt/histplot/norm_parameters_regularization.txt', param_abs)
    # np.savetxt('./txt/histplot/norm_parameters.txt', param_abs)

    # 加载数据
    norm_parameters_r = np.loadtxt('./txt/histplot/norm_parameters_regularization.txt')
    norm_parameters = np.loadtxt('./txt/histplot/norm_parameters.txt')

    # 计算均值
    mean_norm = np.mean(norm_parameters)
    mean_norm_r = np.mean(norm_parameters_r)
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

    # histplot作图
    ax1 = sns.histplot(norm_parameters, kde=False, bins=400, stat='probability', color='#9BAED1',
                       edgecolor='white', label=f'Mean (No Constraint): {mean_norm:.3f}', linewidth=0, ax=axes)
    ax2 = sns.histplot(norm_parameters_r, kde=False, bins=225, stat='probability', color='#E3B7A2',
                       edgecolor='white', label=f'Mean (Add Constraint): {mean_norm_r:.3f}', linewidth=0, ax=axes)

    # 绘制均值竖线
    axes.axvline(mean_norm, color='#9BAED1', linestyle='--', linewidth=2,
                 label='')
    axes.axvline(mean_norm_r, color='#E3B7A2', linestyle='--', linewidth=2,
                 label='')
    # 在合适位置显示均值文本（避免遮挡，可自调 y 坐标）
    ymax = axes.get_ylim()[1]
    axes.text(mean_norm, ymax * 0.9, f"{mean_norm:.3f}",
              color='#9BAED1', fontsize=26, fontweight='bold',
              ha='center', va='bottom', rotation=0, fontname='Times New Roman')
    axes.text(mean_norm_r, ymax * 0.75, f"{mean_norm_r:.3f}",
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
    axes.set_xlim(0, 0.15)
    axes.set_xticks(np.arange(0, 0.15, 0.03))
    axes.tick_params(axis='x', labelsize=26, labelcolor='black')
    for label in axes.get_xticklabels():
        label.set_fontname('Times New Roman')
    # 设置y轴
    ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1]) + 0.03
    axes.set_ylim(0, ymax)
    axes.set_yticks(np.arange(0, ymax, 0.05))
    axes.tick_params(axis='y', labelsize=26, labelcolor='black')
    for label in axes.get_yticklabels():
        label.set_fontname('Times New Roman')
    # 紧密排版
    plt.tight_layout()
    # 标签大小
    plt.xlabel("Parameter Value (Absolute)", size=26)
    plt.ylabel("Frequency", size=26)
    # 刻度大小
    plt.tick_params(labelsize=26)
    # 保存图片
    output_dir = './pictures/'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "Figure5a.pdf"), bbox_inches='tight')

    # 显示图表
    plt.show()
