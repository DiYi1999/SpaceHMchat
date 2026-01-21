import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import seaborn as sns
from matplotlib.lines import lineStyles
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.spatial.distance import cdist
import matplotlib.font_manager
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# matplotlib.font_manager._rebuild()
import scienceplots
# import latex
from scipy.interpolate import make_interp_spline



def channels_plot_AD(orig_tensor, reco_tensor, anomaly_label_tensor, anomaly_detect_tensor, anomaly_score_tensor,
                    timestamp_label=None, data_name=None, name_list=None, plot_parm=None, pdf=None):
    """
    此函数负责将输入数据画图，每通道一图，每图上下俩幅

    Args:
        orig_tensor: 原始数据
        reco_tensor: 重构数据/预测数据
        anomaly_label_tensor: 异常真实标签
        anomaly_detect_tensor: 异常检测标签
        anomaly_score_tensor: 异常分数
        timestamp_label: 时间戳标签，若不传入则默认使用索引, 若传入则作为x轴
        data_name: 数据集名称
        name_list: 各通道名称列表
        plot_parm: 画图参数
        pdf: 画布所在pdf， PdfPages对象

    Returns:

    """
    if plot_parm is not None:
        plt.rcParams['figure.figsize'] = plot_parm['figsize']
        if plot_parm['fontsize'] is not None: plt.rcParams.update({'font.size': plot_parm['fontsize']})
    else:
        plt.rcParams['figure.figsize'] = 6, 2
    for dim in range(orig_tensor.shape[1]):
        y_t, y_p, l_t, l_p, a_s = orig_tensor[:, dim].cpu().numpy(), \
                                  reco_tensor[:, dim].cpu().numpy(), \
                                  anomaly_label_tensor[:, dim].cpu().numpy(), \
                                  anomaly_detect_tensor[:, dim].cpu().numpy(), \
                                  anomaly_score_tensor[:, dim].cpu().numpy()
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.set_ylabel('Value')
        if name_list is not None:
            ax1.set_title(name_list[dim])
        else:
            ax1.set_title('channel ' + str(dim+1))
        ax1.plot(y_t, linestyle=(0,(5,1)), linewidth=1.0, label='Ground Truth', color='k')
        ax1.plot(y_p, linestyle='-', alpha=0.6, linewidth=1.0, label='Predicted', color='r')
        if np.any(l_p):
            ax3 = ax2.twinx()
            ax3.fill_between(np.arange(l_p.shape[0]), l_p, color='red', alpha=0.3, label='Anomaly_pred')
        if np.any(l_t):
            ax4 = ax1.twinx()
            ax4.fill_between(np.arange(l_t.shape[0]), l_t, color='blue', alpha=0.3, label='Anomaly_true')
        ax2.plot(a_s, linewidth=1.0, color='m')
        "ax2画异常分数，linewidth=1.0表示线条宽度为1.0，color='y'表示黄色"
        ax2.set_xlabel('Timestamp')
        ax2.set_ylabel('Anomaly Score')
        pdf.savefig(fig)
        plt.close()

    return pdf




def channels_plot_one_fig_AD(orig_tensor, reco_tensor, anomaly_label_tensor, anomaly_detect_tensor, anomaly_score_tensor,
                             timestamp_label=None, data_name=None, name_list=None, plot_parm=None, fig_save_path=None, mark_each_or_all='all'):
    """
    此函数负责将输入数据画图，每通道一图，每图上下俩幅

    Args:
        orig_tensor: 原始数据
        reco_tensor: 重构数据/预测数据
        anomaly_label_tensor: 异常真实标签
        anomaly_detect_tensor: 异常检测标签
        anomaly_score_tensor: 异常分数
        timestamp_label: 时间戳标签，若不传入则默认使用索引, 若传入则作为x轴
        data_name: 数据集名称
        name_list: 各通道名称列表
        plot_parm: 画图参数
        fig_save_path: 保存图像的路径
        mark_each_or_all: 'each' or 'all', 对各个通道各自标记超过阈值的各自的异常，还是统一检测出来的异常标识在所有通道上

    Returns:

    """
    timestamp_label = np.arange(orig_tensor.shape[0])

    num_channels = orig_tensor.shape[1]
    ncols = 2
    nrows = (num_channels + 1) // ncols

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 6, nrows * 1.5), sharex='col')
    axs = np.array(axs).reshape(nrows, ncols)

    for i in range(num_channels):
        row = i // ncols
        col = i % ncols

        y_t = orig_tensor[:, i].cpu().numpy()
        y_p = reco_tensor[:, i].cpu().numpy()
        if mark_each_or_all == 'each':
            l_t = anomaly_label_tensor[:, i].cpu().numpy()
        else:
            l_t = torch.any(anomaly_label_tensor, dim=1).int().cpu().numpy()
        if mark_each_or_all == 'each':
            l_p = anomaly_detect_tensor[:, i].cpu().numpy()
        else:
            l_p = torch.any(anomaly_detect_tensor, dim=1).int().cpu().numpy()
        if mark_each_or_all == 'each':
            a_s = anomaly_score_tensor[:, i].cpu().numpy()
        else:
            a_s = torch.max(anomaly_score_tensor, dim=1).values.cpu().numpy()

        ax1 = axs[row, col]
        # ax2 = axs[row + 1, col]

        ax1.set_ylabel('Value')
        ax1.set_title(name_list[i] if name_list else f'Channel {i+1}')
        ax1.plot(timestamp_label, y_t, linestyle='-', alpha=0.3, linewidth=1.0, label='Ground Truth', color='k')
        ax1.plot(timestamp_label, y_p, linestyle='-', alpha=1.0, linewidth=0.5, label='Prediction', color='blue')

        if np.any(l_p):
            norm = plt.Normalize(vmin=np.min(a_s[l_p > 0])-0.42, vmax=np.max(a_s[l_p > 0]))
            cmap = plt.cm.Reds
            colors = [
                cmap(norm(score)) if lp == 1 else (0,0,0,0)
                for lp, score in zip(l_p, a_s)
            ]
            ax3 = ax1.twinx()
            ax3.bar(x=timestamp_label, height=l_p, bottom=0, 
                    width=timestamp_label[1] - timestamp_label[0], align='center',
                    color=colors, facecolor=colors,
                    edgecolor='none', linewidth=0,
                    alpha=0.6, label='Anomaly_pred')
    plt.tight_layout()
    
    if fig_save_path:
        os.makedirs(os.path.dirname(fig_save_path), exist_ok=True)
        fig.savefig(fig_save_path + '/AD_result_figure_2400dpi.png', dpi=2400, bbox_inches='tight') # 图片太大，客户端不渲染，dpi=300就够了
        fig.savefig(fig_save_path + '/AD_result_figure.png', dpi=96, bbox_inches='tight')
        fig.savefig(fig_save_path + '/AD_result_figure.svg', format='svg', bbox_inches='tight')
        fig.savefig(fig_save_path + '/AD_result_figure.eps', format='eps', bbox_inches='tight')
        plt.close(fig)

    return fig




def channels_plot_FC(orig_tensor, reco_tensor,
                    data_name=None, name_list=None, plot_parm=None, pdf=None):
    """
    此函数负责将输入数据画图，每通道一图，每图上下俩幅

    Args:
        orig_tensor: 原始数据
        reco_tensor: 重构数据/预测数据
        data_name: 数据集名称
        name_list: 各通道名称列表
        plot_parm: 画图参数
        pdf: 画布所在pdf， PdfPages对象

    Returns:

    """
    if plot_parm is not None:
        plt.rcParams['figure.figsize'] = plot_parm['figsize']
        if plot_parm['fontsize'] is not None: plt.rcParams.update({'font.size': plot_parm['fontsize']})
    else:
        plt.rcParams['figure.figsize'] = 6, 1.3
    for dim in range(orig_tensor.shape[1]):
        y_t, y_p = orig_tensor[:, dim].cpu().numpy(), \
                   reco_tensor[:, dim].cpu().numpy()
        fig, ax1 = plt.subplots(1, 1, sharex=True)
        ax1.set_xlabel('Timestamp')
        if name_list is not None:
            ax1.set_ylabel(name_list[dim])
        else:
            ax1.set_ylabel('Value of channel' + str(dim+1))
        ax1.plot(y_t, linestyle='-', alpha=0.3, linewidth=2.0, label='Ground Truth', color='k')
        ax1.plot(y_p, linestyle='-', alpha=1.0, linewidth=0.5, label='Prediction', color='r')
        pdf.savefig(fig)
        plt.close()

    return pdf


def channels_plot_RE(orig_tensor, reco_tensor,
                     data_name=None, name_list=None
                     , plot_parm=None
                     , pdf=None):
    """
    此函数负责将输入数据画图，每通道一图，每图上下俩幅

    Args:
        orig_tensor: 原始数据
        reco_tensor: 重构数据/预测数据
        data_name: 数据集名称
        name_list: 各通道名称列表
        plot_parm: 画图参数
        pdf: 画布所在pdf， PdfPages对象

    Returns:

    """
    if plot_parm is not None:
        plt.rcParams['figure.figsize'] = plot_parm['figsize']
        if plot_parm['fontsize'] is not None: plt.rcParams.update({'font.size': plot_parm['fontsize']})
    else:
        plt.rcParams['figure.figsize'] = 6, 1.5

    for dim in range(orig_tensor.shape[1]):
        y_t, y_p = orig_tensor[:, dim].cpu().numpy(), \
                   reco_tensor[:, dim].cpu().numpy()
        fig, ax1 = plt.subplots(1, 1, sharex=True)
        ax1.set_ylabel('Value')
        ax1.set_xlabel('Timestamp')
        if name_list is not None:
            ax1.set_title(name_list[dim])
        else:
            ax1.set_title('channel ' + str(dim+1))
        ax1.plot(y_t, linestyle=(0,(5,1)), linewidth=1.0, label='Ground Truth', color='k')
        ax1.plot(y_p, linestyle='-', alpha=0.6, linewidth=1.0, label='Prediction', color='r')
        pdf.savefig(fig)
        plt.close()

    return pdf



def channels_plot_RE_T(orig_tensor, reco_tensor,
                       data_name=None, name_list=None, time_label=None,
                       plot_parm=None,
                       pdf=None):
    """
    此函数负责将输入数据画图，每通道一图，每图上下俩幅，与channels_plot_RE的区别在于每幅图的横轴是time_label

    Args:
        orig_tensor: 原始数据
        reco_tensor: 重构数据/预测数据
        data_name: 数据集名称
        name_list: 各通道名称列表
        time_label: 时间顺序标签，以其作为x轴，并绘制散点图而并非折线图
        plot_parm: 画图参数
        pdf: 画布所在pdf， PdfPages对象

    Returns:

    """
    if plot_parm is not None:
        plt.rcParams['figure.figsize'] = plot_parm['figsize']
        if plot_parm['fontsize'] is not None: plt.rcParams.update({'font.size': plot_parm['fontsize']})
    else:
        plt.rcParams['figure.figsize'] = 4, 2
        plt.rcParams.update({'font.size': 12})

    time_label = time_label.cpu().numpy()
    fig, ax = plt.subplots()
    ax.plot(time_label, linewidth=1.0, color='k')
    ax.set_xlabel('Step')
    ax.set_ylabel('Timestamp')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close()
    for dim in range(orig_tensor.shape[1]):
        y_t, y_p = orig_tensor[:, dim].cpu().numpy(), \
                   reco_tensor[:, dim].cpu().numpy()
        fig, ax1 = plt.subplots(1, 1, sharex=True)
        ax1.set_ylabel('Value')
        ax1.set_xlabel('Timestamp')
        if name_list is not None:
            ax1.set_title(name_list[dim])
        else:
            ax1.set_title('channel ' + str(dim+1))
        ax1.scatter(time_label, y_t, s=0.1, label='Ground Truth', color='k')
        ax1.scatter(time_label, y_p, s=0.1, label='Prediction', color='r')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close()

    return pdf



def channels_plot_onepage(orig_sample_tensor, ratio=None, color='k', pdf=None):
    """
    此函数负责将输入数据各通道画图，画在一页PDF上

    Args:
        orig_sample_tensor: 测试集数据，要画的样本，实际上后面没用到
        ratio: 样本长度和测试机的长度比例用来调整画图图幅大小
        color: 绘图颜色
        pdf:

    Returns:

    """
    if ratio is None:
        ratio = 1
    plt.rcParams['figure.figsize'] = 3, 1*orig_sample_tensor.shape[1]
    fig, axs = plt.subplots(orig_sample_tensor.shape[1], 1, sharex=True)
    for dim, ax in enumerate(axs):
        ax.plot(orig_sample_tensor[:, dim].cpu().numpy(), linewidth=1.0, color=color)
        ax.set_ylabel('channel' + str(dim))
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close()

    return pdf


def score_kernel_density_plot(anomaly_score_tensor, anomaly_label_vector, pdf):
    pass
