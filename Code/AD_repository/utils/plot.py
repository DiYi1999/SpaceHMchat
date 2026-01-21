import os
import pickle
from pathlib import Path
import numpy as np
import torch
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.font_manager
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# matplotlib.font_manager._rebuild()
import scienceplots
# import latex
from scipy.interpolate import make_interp_spline
from AD_repository.utils.plot_sup import *
from AD_repository.main_sub import get_plot_pram



def MyPlot_AD(args, X_orig, Y_fore, label, all_label, S, detect_01, exam_result, args_dict, timestamp_label=None):
    """
    
    Args:
        args: 全局参数
        X_orig: (node_num, len)原始数据
        Y_fore: (node_num, len)重构/预测数据
        label: (lag)每个时间的异常实际标签
        all_label: (node_num, len)每个节点每个时间的 实际的的 异常01矩阵
        S: (node_num, node_num) 异常分数矩阵
        detect_01: (node_num, len) 检测出来的 异常01矩阵
        exam_result: 字典，包含了各种performance指标
        args_dict: 字典，包含了各种参数
        timestamp_label: 时间顺序标签，如果不是None，那将以其作为x轴，并绘制散点图而并非折线图

    Returns:

    """
    """转置以便画图"""
    orig_tensor = X_orig.T.contiguous()
    'orig_tensor原始数据: (len, node_num)'
    fore_tensor = Y_fore.T.contiguous()
    'fore_tensor重构/预测数据: (len, node_num)'
    anomaly_label_vector = label
    'anomaly_label_vector异常实际标签: (len)'
    anomaly_label_tensor = all_label.T.contiguous()
    'anomaly_label_tensor异常实际标签: (len, node_num)'
    anomaly_score_tensor = S.T.contiguous()
    'anomaly_score_tensor异常分数矩阵: (len, node_num)'
    anomaly_detect_tensor = detect_01.T.contiguous()
    'anomaly_detect_tensor检测出来的 异常01矩阵: (len, node_num)'

    """如果len太长，画图时间太长、画图文件太大，因此一旦len大于2048，等间隔采样2048个点"""
    if orig_tensor.shape[0] > 2048:
        print('原始数据长度【{}】大于2048，进行等间隔采样'.format(orig_tensor.shape[0]))
        step = orig_tensor.shape[0] // 2048
        orig_tensor = orig_tensor[::step, :]
        fore_tensor = fore_tensor[::step, :]
        anomaly_label_vector = anomaly_label_vector[::step]
        anomaly_label_tensor = anomaly_label_tensor[::step, :]
        anomaly_score_tensor = anomaly_score_tensor[::step, :]
        anomaly_detect_tensor = anomaly_detect_tensor[::step, :]
        timestamp_label = timestamp_label[::step] if timestamp_label is not None else None


    """相关设置"""

    "设置字体"
    plt.style.use(['science', 'ieee', 'high-vis'])
    plot_parm = get_plot_pram(args)

    "设置路径"
    # plot_dirname = '_'.join([f'{k}={v}' for k, v in exam_result.items()])
    # 上面这个保存v的时候会保存很多位，不好看，所以要改成只保留三位小数的科学计数法
    plot_dirname = '_'.join([f'{k}={float(v):.3e}' for k, v in exam_result.items()])
    # plot_dirname名称太长windows不支持，所以只保存前66个字符
    plot_dirname = plot_dirname[:66]
    plot_dirname_path = args.plot_save_path + '/' + plot_dirname + '.pdf'
    dirname = os.path.dirname(plot_dirname_path)
    Path(dirname).mkdir(parents=True, exist_ok=True)


    """保存这些数据成pkl以备以后启用"""
    if args.if_plot_data_save:
        data_save_path = args.plot_save_path + '/' + plot_dirname + '.pkl'
        data_save_dict = {'args': args,
                          'X_orig': X_orig,
                          'Y_fore': Y_fore,
                          'label': label,
                          'all_label': all_label,
                          'S': S,
                          'detect_01': detect_01,
                          'exam_result': exam_result,
                          'args_dict': args_dict}
        with open(data_save_path, 'wb') as f:
            pickle.dump(data_save_dict, f)


    """获得画图name_list"""
    try:
        try:
            try:
                orig_data_csv_path = os.path.join(args.root_path, args.data_path, '{}_train.csv'.format(args.data_name))
                name_list = pd.read_csv(orig_data_csv_path, nrows=0, sep=',', index_col=False).columns.tolist()
            except:
                orig_data_csv_path = os.path.join(args.root_path, args.data_path, '{}_Train.csv'.format(args.data_name))
                name_list = pd.read_csv(orig_data_csv_path, nrows=0, sep=',', index_col=False).columns.tolist()
        except:
            try:
                orig_data_csv_path = os.path.join('/', args.root_path, args.data_path, '{}_train.csv'.format(args.data_name))
                name_list = pd.read_csv(orig_data_csv_path, nrows=0, sep=',', index_col=False).columns.tolist()
            except:
                orig_data_csv_path = os.path.join('/', args.root_path, args.data_path, '{}_Train.csv'.format(args.data_name))
                name_list = pd.read_csv(orig_data_csv_path, nrows=0, sep=',', index_col=False).columns.tolist()
    except:
        try:
            orig_data_csv_path = os.path.join(args.root_path, args.data_path, '{}.csv'.format(args.data_name))
            name_list = pd.read_csv(orig_data_csv_path, nrows=0, sep=',', index_col=False).columns.tolist()
        except:
            orig_data_csv_path = os.path.join('/', args.root_path, args.data_path, '{}.csv'.format(args.data_name))
            name_list = pd.read_csv(orig_data_csv_path, nrows=0, sep=',', index_col=False).columns.tolist()
    if len(name_list) == orig_tensor.shape[1]:
        name_list = name_list
    else:
        print('画图的name_list列名长度不对，从原文件读取出【{}】，共【{}】列数据，'
              '采取了从后向前截取方法，使用【{}】，共【{}】列数据，但还是建议检查一下'.format(
                  name_list, len(name_list), name_list[-orig_tensor.shape[1]:], orig_tensor.shape[1]))
        name_list = name_list[-orig_tensor.shape[1]:]


    """画图"""
    
    "先画PNG"
    fig = channels_plot_one_fig_AD(orig_tensor=orig_tensor,
                                    reco_tensor=fore_tensor,
                                    anomaly_label_tensor=anomaly_label_tensor,
                                    anomaly_detect_tensor=anomaly_detect_tensor,
                                    anomaly_score_tensor=anomaly_score_tensor,
                                    timestamp_label=timestamp_label, 
                                    data_name=args.data_name,
                                    name_list=name_list,
                                    plot_parm=plot_parm,
                                    fig_save_path=args.plot_save_path,
                                    mark_each_or_all='all')

    "展开pdf"
    pdf = PdfPages(plot_dirname_path)

    "写参数到pdf第一页"
    # pdf打开后先在第一页把参数写上，参考自https://stackoverflow.com/questions/49444008/add-text-with-pdfpages-matplotlib
    firstPage = plt.figure(figsize=(6, 6))
    firstPage.clf()  # 清空画布
    txt = '@'.join([f'{k}={v}' for k, v in {**exam_result, **args_dict}.items()])
    hang = 18
    hang_len = len(txt) // hang
    txt = '\n'.join(txt[i * hang_len:(i + 1) * hang_len] for i in range(0, hang + 1))
    firstPage.text(0.5, 0.5, txt, ha='center', va='center')
    pdf.savefig(firstPage)
    plt.close()

    "画每个通道原始数据和重构/预测数据"
    pdf = channels_plot_AD(orig_tensor=orig_tensor,
                           reco_tensor=fore_tensor,
                           anomaly_label_tensor=anomaly_label_tensor,
                           anomaly_detect_tensor=anomaly_detect_tensor,
                           anomaly_score_tensor=anomaly_score_tensor,
                           timestamp_label=timestamp_label,
                           data_name=args.data_name,
                           name_list=name_list,
                           plot_parm=plot_parm,
                           pdf=pdf)

    "画异常分数核密度估计图"
    pdf = score_kernel_density_plot(anomaly_score_tensor, anomaly_label_vector, pdf)


    """关闭pdf"""
    pdf.close()



def MyPlot_FC(args, X_orig, Y_fore, exam_result, args_dict, name_list=None):
    """

        Args:
            args: 全局参数
            X_orig: (node_num, len)原始数据
            Y_fore: (node_num, len)重构/预测数据
            exam_result: 字典，包含了各种performance指标
            args_dict: 字典，包含了各种参数

        Returns:

        """
    """转置以便画图"""

    orig_tensor = X_orig.T.contiguous()
    'orig_tensor原始数据: (len, node_num)'
    fore_tensor = Y_fore.T.contiguous()
    'fore_tensor重构/预测数据: (len, node_num)'

    """相关设置"""

    "设置字体"
    plt.style.use(['science', 'ieee', 'high-vis'])
    plot_parm = get_plot_pram(args)

    "设置路径"
    # plot_dirname = '_'.join([f'{k}={v}' for k, v in exam_result.items()])
    # 上面这个保存v的时候会保存很多位，不好看，所以要改成只保留三位小数的科学计数法
    plot_dirname = '_'.join([f'{k}={float(v):.3e}' for k, v in exam_result.items()])
    # plot_dirname名称太长windows不支持，所以只保存前66个字符
    plot_dirname = plot_dirname[:66]
    plot_dirname_path = args.plot_save_path + '/' + plot_dirname + '.pdf'
    dirname = os.path.dirname(plot_dirname_path)
    Path(dirname).mkdir(parents=True, exist_ok=True)


    """保存这些数据成pkl以备以后启用"""
    if args.if_plot_data_save:
        data_save_path = args.plot_save_path + '/' + plot_dirname + '.pkl'
        data_save_dict = {'args': args,
                            'X_orig': X_orig,
                            'Y_fore': Y_fore,
                            'exam_result': exam_result,
                            'args_dict': args_dict}
        with open(data_save_path, 'wb') as f:
            pickle.dump(data_save_dict, f)


    """获得画图name_list，原始文件表头"""
    if name_list is None:
        try:
            orig_data_csv_path = os.path.join(args.root_path, args.data_path, '{}_train.csv'.format(args.data_name))
            name_list = pd.read_csv(orig_data_csv_path, nrows=0, sep=',', index_col=False).columns.tolist()
        except:
            orig_data_csv_path = os.path.join(args.root_path, args.data_path, '{}.csv'.format(args.data_name))
            name_list = pd.read_csv(orig_data_csv_path, nrows=0, sep=',', index_col=False).columns.tolist()
        if len(name_list) == orig_tensor.shape[1]:
            name_list = name_list
        else:
            print('画图的name_list列名长度不对，从原文件读取出【{}】，共【{}】列数据'
                '，采取了从后向前截取方法，使用【{}】，共【{}】列数据，但还是建议检查一下',
                name_list, len(name_list), name_list[-orig_tensor.shape[1]:], orig_tensor.shape[1])
            name_list = name_list[-orig_tensor.shape[1]:]
    else:
        name_list = name_list


    """画图"""

    "展开pdf"
    pdf = PdfPages(plot_dirname_path)

    "写参数到pdf第一页"
    # pdf打开后先在第一页把参数写上，参考自https://stackoverflow.com/questions/49444008/add-text-with-pdfpages-matplotlib
    firstPage = plt.figure(figsize=(6, 6))
    firstPage.clf()  # 清空画布
    txt = '@'.join([f'{k}={v}' for k, v in {**exam_result, **args_dict}.items()])
    hang = 18
    hang_len = len(txt) // hang
    txt = '\n'.join(txt[i * hang_len:(i + 1) * hang_len] for i in range(0, hang + 1))
    firstPage.text(0.5, 0.5, txt, ha='center', va='center')
    pdf.savefig(firstPage)
    plt.close()

    "画每个通道原始数据和重构/预测数据"
    pdf = channels_plot_FC(orig_tensor=orig_tensor,
                           reco_tensor=fore_tensor,
                           data_name=args.data_name,
                           name_list=name_list,
                           plot_parm=plot_parm,
                           pdf=pdf)

    """关闭pdf"""
    pdf.close()


def MyPlot_RE(args, X_orig, Y_fore, exam_result, args_dict, time_label=None):
    """

        Args:
            args: 全局参数
            X_orig: (node_num, len)原始数据
            Y_fore: (node_num, len)重构/预测数据
            exam_result: 字典，包含了各种performance指标
            args_dict: 字典，包含了各种参数
            time_label: 时间顺序标签，如果不是None，那将以其作为x轴，并绘制散点图而并非折线图

        Returns:

        """
    """转置以便画图"""

    orig_tensor = X_orig.T.contiguous()
    'orig_tensor原始数据: (len, node_num)'
    fore_tensor = Y_fore.T.contiguous()
    'fore_tensor重构/预测数据: (len, node_num)'

    """相关设置"""

    "设置字体"
    plt.style.use(['science', 'ieee', 'high-vis'])
    plot_parm = get_plot_pram(args)

    "设置路径"
    # plot_dirname = '_'.join([f'{k}={v}' for k, v in exam_result.items()])
    # 上面这个保存v的时候会保存很多位，不好看，所以要改成只保留三位小数的科学计数法
    plot_dirname = '_'.join([f'{k}={float(v):.3e}' for k, v in exam_result.items()])
    # plot_dirname名称太长windows不支持，所以只保存前66个字符
    plot_dirname = plot_dirname[:66]
    plot_dirname_path = args.plot_save_path + '/' + plot_dirname + '.pdf'
    dirname = os.path.dirname(plot_dirname_path)
    Path(dirname).mkdir(parents=True, exist_ok=True)


    """保存这些数据成pkl以备以后启用"""
    if args.if_plot_data_save:
        data_save_path = args.plot_save_path + '/' + plot_dirname + '.pkl'
        data_save_dict = {'args': args,
                            'X_orig': X_orig,
                            'Y_fore': Y_fore,
                            'exam_result': exam_result,
                            'args_dict': args_dict,
                            'time_label': time_label}
        with open(data_save_path, 'wb') as f:
            pickle.dump(data_save_dict, f)


    """获得画图name_list"""
    try:
        orig_data_csv_path = os.path.join(args.root_path, args.data_path, '{}_train.csv'.format(args.data_name))
        name_list = pd.read_csv(orig_data_csv_path, nrows=0, sep=',', index_col=False).columns.tolist()
    except:
        orig_data_csv_path = os.path.join(args.root_path, args.data_path, '{}.csv'.format(args.data_name))
        name_list = pd.read_csv(orig_data_csv_path, nrows=0, sep=',', index_col=False).columns.tolist()
    if len(name_list) == orig_tensor.shape[1]:
        name_list = name_list
    else:
        print('画图的name_list列名长度不对，从原文件读取出【{}】，共【{}】列数据'
              '，采取了从后向前截取方法，使用【{}】，共【{}】列数据，但还是建议检查一下'.format(
            name_list, len(name_list), name_list[-orig_tensor.shape[1]:], orig_tensor.shape[1]))
        name_list = name_list[-orig_tensor.shape[1]:]


    """画图"""

    "展开pdf"
    pdf = PdfPages(plot_dirname_path)

    "写参数到pdf第一页"
    # pdf打开后先在第一页把参数写上，参考自https://stackoverflow.com/questions/49444008/add-text-with-pdfpages-matplotlib
    firstPage = plt.figure(figsize=(6, 6))
    firstPage.clf()  # 清空画布
    txt = '@'.join([f'{k}={v}' for k, v in {**exam_result, **args_dict}.items()])
    hang = 18
    hang_len = len(txt) // hang
    txt = '\n'.join(txt[i * hang_len:(i + 1) * hang_len] for i in range(0, hang + 1))
    firstPage.text(0.5, 0.5, txt, ha='center', va='center')
    pdf.savefig(firstPage)
    plt.close()

    "画每个通道原始数据和重构/预测数据"
    if time_label is not None:
        pdf = channels_plot_RE_T(orig_tensor=orig_tensor,
                                 reco_tensor=fore_tensor,
                                 data_name=args.data_name,
                                 name_list=name_list,
                                 time_label=time_label,
                                 plot_parm=plot_parm,
                                 pdf=pdf)
    else:
        pdf = channels_plot_RE(orig_tensor=orig_tensor,
                               reco_tensor=fore_tensor,
                               data_name=args.data_name,
                               name_list=name_list,
                               plot_parm=plot_parm,
                               pdf=pdf)

    """关闭pdf"""
    pdf.close()


