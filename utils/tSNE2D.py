from sklearn.manifold import TSNE
from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class tSNE(object):
    def __init__(self):
        # self.data = data
        # self.labels = labels
        self.start_time = time()

    def set_plt(self, title):
        end_time = time()
        # plt.title(f'{title} time consume:{end_time - self.start_time:.3f} s')
        plt.legend(title='', fontsize='small', loc=0, markerscale=0.8, handlelength=0.5).get_frame().set_facecolor('none')
        plt.ylabel('')
        plt.xlabel('')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.show()
        # plt.savefig(r'/pictures/Tsne/{}.svg'.format(title), format='svg', bbox_inches='tight')

    def t_sne(self, data, labels, title):
        # t-sne处理
        print('starting T-SNE process')
        data = TSNE(n_components=2, random_state=42, init='pca').fit_transform(data)

        # 获取唯一标签
        unique_labels = np.unique(labels)
        for label in unique_labels:
            # 获取当前标签对应的数据
            label_data = data[labels == label]
            # 保存到对应的 txt 文件中
            np.savetxt(f'txt/tsne/RING/tsne_data_{label}.txt', label_data, fmt='%.6f')

        df = pd.DataFrame(data, columns=['x', 'y'])  # 转换成df表
        df.insert(loc=1, column='label', value=labels)

        print('Finished')

        # 绘图

        # PALETTE = ['#2f4f4f', '#8b4513', '#006400', '#4682b4', '#4b0082', '#ff0000', '#ffd700', '#7fff00', '#00ffff',
        #            '#0000ff', '#ff69b4', '#ffe4c4'
        #            ]
        PALETTE = ['#2f4f4f', '#228b22', '#7f0000', '#000080', '#ff8c00', '#ffff00', '#00ff00', '#00ffff', '#ff00ff',
                   '#1e90ff', '#ffdead', '#ff69b4'
                   ]

        # xxx = sns.color_palette(PALETTE)
        sns.scatterplot(x='x', y='y', hue='label', s=50, palette=PALETTE, data=df)
        # More marker customization,更具scatter_kws参数控制颜色，透明度，点的大小

        self.set_plt(title)

    def visual_dataset(self, data, labels, title):
        self.t_sne(data, labels, title)


def x_norm(x):
    x_norm1 = np.linalg.norm(x, ord=None, axis=1, keepdims=False).reshape(-1, 1)
    x = x / x_norm1
    return x


if __name__ == '__main__':
    df = pd.read_csv('/home/private/hrrp/data/test/56.csv', header=None)
    data1, labels1 = df.iloc[:, :-1].values, df.iloc[:, -1]
    data1 = x_norm(data1)
    print(labels1.shape)
    title1 = f'RowData\n'
    tSNE = tSNE()
    tSNE.visual_dataset(data1, labels1, title1)