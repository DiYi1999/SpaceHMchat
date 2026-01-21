import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.font_manager
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# matplotlib.font_manager._rebuild()
import scienceplots
# import latex
import os
from pathlib import Path
from matplotlib.colors import ListedColormap


def plot_adj_heatmap(A_w, save_path, indexs_list=None, columns_list=None):
    """

        Args:
            A_w: (node_num1, node_num2)权重邻接矩阵，即相关性矩阵，可以做可视化用
            save_path: 保存路径, 以.pdf结尾
            indexs_list: 行名列表，A_w短边的那维
            columns_list: 列名列表，A_w长边的那维

        Returns:

        """
    plt.style.use(['science', 'ieee', 'high-vis'])
    heatmap_save_path = save_path
    dirname = os.path.dirname(heatmap_save_path)
    Path(dirname).mkdir(parents=True, exist_ok=True)
    np.fill_diagonal(A_w, 1)
    A_w = np.floor(A_w * 100) / 100
    if A_w.shape[0] > A_w.shape[1]:
        A_w = A_w.T
    if indexs_list is not None and columns_list is not None:
        A_w = pd.DataFrame(A_w, index=indexs_list, columns=columns_list)
    node_num1 = A_w.shape[0]
    node_num2 = A_w.shape[1]
    if not os.path.exists(heatmap_save_path):
        pdf = PdfPages(heatmap_save_path)

        for cmap in ['Reds', 'Purples']:
            for linecolor in ['black', 'white']:
                fig, ax = plt.subplots(figsize=(9/20*node_num2, 6/20*node_num1))
                sns.heatmap(data=A_w, annot=True, fmt=".2f", linewidths=.5, cmap=cmap, linecolor=linecolor, ax=ax)
                fig.patch.set_edgecolor('black')
                pdf.savefig(fig)
                plt.close()

        pdf.close()



def plot_W_heatmap(A_w, indexs_list=None, columns_list=None, fig_size=None, pdf=None):
    """

        Args:
            A_w: (node_num1, node_num2)权重邻接矩阵，即相关性矩阵，可以做可视化用
            indexs_list: 行名列表，A_w短边的那维
            columns_list: 列名列表，A_w长边的那维
            pdf: PdfPages对象

        Returns:

        """

    plt.style.use(['science', 'ieee', 'high-vis'])

    if  A_w.mean() < 1:
        A_w = np.floor(A_w * 100) / 100

    if A_w.shape[0] > A_w.shape[1]:
        A_w = A_w.T

    if indexs_list is not None and columns_list is not None:
        A_w = pd.DataFrame(A_w, index=indexs_list, columns=columns_list)

    if indexs_list is not None and columns_list is not None:
        A_w_abs = np.abs(A_w.values)
        A_w_abs = pd.DataFrame(A_w_abs, index=indexs_list, columns=columns_list)
    else:
        A_w_abs = np.abs(A_w)

    node_num1 = A_w.shape[0]
    node_num2 = A_w.shape[1]

    # 展开pdf
    pdf = pdf

    for cmap in ['Reds', 'Purples']:
        for linecolor in ['black', 'white']:

            if fig_size is not None:
                fig, ax = plt.subplots(figsize=fig_size)
            else:
                fig, ax = plt.subplots(figsize=(9 / 20 * node_num2, 6 / 20 * node_num1))
            if not np.all(A_w.values == A_w.values.astype(int)):
                sns.heatmap(data=A_w_abs, annot=A_w, fmt=".2f", linewidths=.5, cmap=cmap, linecolor=linecolor, ax=ax)
            else:
                sns.heatmap(data=A_w_abs, annot=A_w.astype(int), fmt="d", linewidths=.5, cmap=cmap, linecolor=linecolor, ax=ax)
            fig.patch.set_edgecolor('black')
            pdf.savefig(fig)
            plt.close()

    return pdf