"""
该文件中MIC函数的相关参数设置不很熟练，得多调整，可供参考的俩个是：
之前那篇用MIC建图的文章用的是pstats(j, alpha=9, c=5, est="mic_e")
另外一个是minepy的官方文档，用的是alpha=0.6, c=15, est="mic_approx"
"""
from pathlib import Path

import pandas as pd
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from minepy import MINE
try:
    from minepy import MINE
    MINE_loaded = True
    # We recommend using minepy
except:
    from AD_repository.graph.dp_mic.mic import mic_val
    MINE_loaded = False
    # but the python version of minepy stopped maintenance many years ago, if you got installation errors, an alternative from https://github.com/jlazarsfeld/dp-mic
    # Another alternative: https://github.com/jrestrepo86/minepy/tree/main
from copent import copent
import torch
import os
from AD_repository.utils.plot_heatmap import plot_adj_heatmap
from AD_repository.utils.process import *
from scipy.stats import kendalltau
# from sklearn.metrics import mutual_info_score, normalized_mutual_info_score % don't use this!
from sklearn.feature_selection import mutual_info_regression



def A_w_calculate(args, data_normal):
    """
    该函数用于通过一段长度为cal_len多维X的数据，计算A_w，
    至于计算邻接矩阵的方法，可选MIC和Copent，分别是最大信息系数和Copula熵

    Args:
        args: 参数集合
        data_normal: (all_len, node_num)

    Returns:
        A_w: (node_num, node_num)权重邻接矩阵，即相关性矩阵，可以做可视化用
    """
    # data = pd.read_csv(os.path.join(args.root_path, args.data_path,
    #                                 '{}_train.csv'.format(args.data_name)),
    #                    sep=',', index_col=False)
    # cau_data = torch.from_numpy(data.values[:args.graph_ca_len, :]).to(batch_x.device)
    # # (graph_ca_len, node_num)
    # # 对cau_data进行按列最大最小归一化
    # cau_data = (cau_data - cau_data.min(dim=0)[0]) / (cau_data.max(dim=0)[0] - cau_data.min(dim=0)[0] + 1e-8)
    # # (graph_ca_len, node_num)

    # 判断输入是否为np.array，如果是pandas.DataFrame则取出为np.array，如果是Tensor则转为np.array
    if isinstance(data_normal, np.ndarray):
        data = data_normal
    elif isinstance(data_normal, pd.DataFrame):
        data = data_normal.values
    elif isinstance(data_normal, torch.Tensor):
        data = data_normal.cpu().numpy()
    else:
        raise ValueError("data_normal should be np.ndarray or pd.DataFrame")
    ca_len, node_num = data.shape
    'ca_len: 用于计算的、意欲截取原始数据的长度'
    # 创建全0邻接矩阵
    A_w = np.zeros((node_num, node_num)).astype(np.float32)
    # 计算MIC/Copent熵
    if args.graph_ca_meth == "MIC":
        for i in range(0, node_num - 1):
            for j in range(i + 1, node_num):
                if MINE_loaded:
                    mine = MINE(alpha=args.MIC_alpha, c=args.MIC_c, est="mic_approx")
                    '关于这俩参数的取值，可以参考minepy的官方文档，也可以参考之前那篇用MIC建图的文章' \
                    '之前那篇用MIC建图的文章用的是pstats(j, alpha=9, c=5, est="mic_e")' \
                    '另外一个是minepy的官方文档，用的是alpha=0.6, c=15, est="mic_approx"'
                    mine.compute_score(data[:, i], data[:, j])
                    mic_coff = mine.mic()
                else:
                    D = [(data[num,i],data[num,j]) for num in range(ca_len)]
                    index, mic_coff = mic_val(D=D, B=None, alpha=args.MIC_alpha, c=args.MIC_c, variant="mass")
                    # mic_val(D=None, B=None, alpha=None, c=1, variant="mass", ranges=None, private=None, eps=1, ptable=None)
                    '关于这俩参数的取值，可以参考minepy的官方文档，也可以参考之前那篇用MIC建图的文章'
                A_w[i, j] = mic_coff
                A_w[j, i] = A_w[i, j]
        # 对角线置1
        np.fill_diagonal(A_w, 1)
    elif args.graph_ca_meth == "Copent":
        for i in range(0, node_num - 1):
            for j in range(i + 1, node_num):
                data1 = data[:, [i, j]]
                # 这[:, [i, j]]和[:, i], [:, j]是一样的，但是[:, [i, j]]可以保持二维？
                A_w[i, j] = copent(data1)
                A_w[j, i] = A_w[i, j]
        # A_w进行01归一化
        A_w = (A_w - A_w.min()) / (A_w.max() - A_w.min() + 1e-5)
        # 对角线置1
        np.fill_diagonal(A_w, 1)
    elif args.graph_ca_meth == "Cosine":
        data_T = data.T
        'data_T: (node_num, graph_ca_len)'
        A_w = np.matmul(data_T, data)
        'A_w: (node_num, node_num)'
        A_w = A_w / ((np.linalg.norm(data_T, axis=1, keepdims=True) * np.linalg.norm(data, axis=0, keepdims=True)) + 1e-5)
        'np.linalg是numpy的线性代数库，np.linalg.norm是求范数，axis=1表示按行求，axis=0表示按列求'
        'A_w: (node_num, node_num)'
        # 对角线置1
        np.fill_diagonal(A_w, 1)
        # # 余弦相似度算出来是-1到1，这里归一化到0到1
        # A_w = (A_w + 1) / 2
        # 错了错了错了，余弦相似度-1是代表负相关，0才是不相干，所以不能归一化，应该取绝对值
        A_w = np.abs(A_w)
    elif args.graph_ca_meth == "Kendall":
        for i in range(0, node_num - 1):
            for j in range(i + 1, node_num):
                tau, _ = kendalltau(data[:, i], data[:, j])
                if np.isnan(tau):
                    tau = 0
                # kendalltau在输入的某一列全是一样的数，即常值数列时，会返回nan
                A_w[i, j] = tau
                A_w[j, i] = A_w[i, j]
        # 对角线置1
        np.fill_diagonal(A_w, 1)
        # Kendall系数是负数时代表负相关
        A_w = np.abs(A_w)
    elif args.graph_ca_meth == "MutualInfo":
        for i in range(0, node_num - 1):
            X = data[:, i+1:]
            'X: (ca_len, node_num-1 - i)'
            y = data[:, i]
            'y: (ca_len,)'
            mi = mutual_info_regression(X, y)
            'mi: (node_num-1 - i,)'
            A_w[i, i+1:] = mi
            A_w[i+1:, i] = mi
        # True mutual information can’t be negative. If its estimate turns out to be negative, it is replaced by zero.
        A_w[A_w < 0] = 0
        # Normalize A_w to [0, 1]
        A_w = A_w / A_w.max() if A_w.max() != 0 else A_w
        # 对角线置1
        np.fill_diagonal(A_w, 1)
    else:
        raise ValueError("method should be MIC or Copent or Cosine")

    return A_w


def A_w_csv_and_plot(args, A_w, csv_dir=None):
    """
    该函数用于将计算得到的A_w保存为csv文件，并绘制heatmap

    Args:
        args: 参数集合
        A_w: (node_num, node_num)权重邻接矩阵，即相关性矩阵，可以做可视化用
        csv_dir: csv文件保存路径，以.csv结尾

    Returns:
        csv_dir: 保存的csv文件路径
    """

    """A_w矩阵保存成csv文件，如何不存在文件路径则创建，存在则覆盖"""
    if not os.path.exists(csv_dir):
        Path(os.path.dirname(csv_dir)).mkdir(parents=True, exist_ok=True)
        # parents=True表示如果 上级目录 不存在则创建，若为False则不创建，若为None则只有最后一级不存在才创建
        # exist_ok=True表示如果目录已经存在则不会报错，若为False则会报错
        A_w_df = pd.DataFrame(A_w)
        A_w_df.to_csv(csv_dir, index=False, header=False)

    """绘制heatmap"""
    # 把csv_dir的'_A_w.csv'换成'_adj_heatmap.pdf'
    file_path = csv_dir.replace('_A_w.csv', '_adj_heatmap.pdf')
    plot_adj_heatmap(A_w, file_path)

    return csv_dir


def A_other_calculate(args, A_w, if_return_norm=False):
    """
    根据A_w计算A、A_self、A_norm、A_self_norm

    Args:
        args: 参数集合
        A_w: (node_num, node_num)权重邻接矩阵
        if_return_norm: 是否计算并返回归一化后的邻接矩阵和自连接的归一化邻接矩阵

    Returns:
        A: (node_num, node_num)非自连接的01邻接矩阵
        A_self: (node_num, node_num)自连接的01邻接矩阵
        A_w: (node_num, node_num)权重邻接矩阵，即相关性矩阵，可以做可视化用
        A_norm: (node_num, node_num)非自连接的归一化邻接矩阵，如果if_return_norm=True则计算
        A_self_norm: (node_num, node_num)自连接的归一化邻接矩阵，如果if_return_norm=True则计算
    """

    "根据A_w计算A、A_self、A_norm、A_self_norm"
    # # 根据args.graph_ca_ratio这个比例来选取个阈值，可以考虑使用np.partition
    # A_w_flatten = A_w.flatten()
    # k = int(len(A_w_flatten) * args.graph_ca_ratio)
    # graph_ca_thre = np.partition(A_w_flatten, -k)[-k]

    # A_w比阈值大的置1，小的置0
    node_num = A_w.shape[0]
    A = np.zeros((node_num, node_num)).astype(np.float32)
    A[A_w >= args.graph_ca_thre] = 1
    # A_w之后会决定是否自连接，现在对角线先全置0，但其实这里也不大需要，因为前面计算时压根就没计算对角线
    np.fill_diagonal(A, 0)
    # 自连接的邻接矩阵
    A_self = A + np.eye(node_num).astype(np.float32)

    # 如果A的某一行全是0，则将A_w那一行的第二最大值对应的位置在A里置1，列也是一样
    for i in range(node_num):
        if np.sum(A[i]) == 0:
            max2_index = np.argsort(A_w[i])[-2]
            A[i, max2_index] = 1
            A[max2_index, i] = 1

    if if_return_norm:
        # 每行相加并对角化为度矩阵
        degree = np.sum(A, axis=1)
        # D = np.diag(degree)
        degree_inv_sqrt = np.power(degree, -0.5)
        degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0
        D_inv_sqrt = np.diag(degree_inv_sqrt)
        # 归一化邻接矩阵
        A_norm = np.matmul(np.matmul(D_inv_sqrt, A), D_inv_sqrt)
        # 自连接的归一化邻接矩阵
        degree_self = np.sum(A_self, axis=1)
        degree_self_inv_sqrt = np.power(degree_self, -0.5)
        degree_self_inv_sqrt[np.isinf(degree_self_inv_sqrt)] = 0
        D_self_inv_sqrt = np.diag(degree_self_inv_sqrt)
        A_self_norm = np.matmul(np.matmul(D_self_inv_sqrt, A_self), D_self_inv_sqrt)

        return A, A_self, A_w, A_norm, A_self_norm
    else:
        return A, A_self, A_w



def get_PINN_A(args, A):
    A0 = A[:args.sensor_num, :args.sensor_num]
    'A0: (sensor_num, sensor_num)'
    A1 = A[args.sensor_num:, args.sensor_num:]
    'A1: (5+4, 5+4)'
    A2 = A[:args.sensor_num, args.sensor_num:]
    'A2: (sensor_num, 5+4)'
    A3 = A[args.sensor_num:, :args.sensor_num]
    'A3: (5+4, sensor_num)'
    A00 = np.concatenate((A0, A0), axis=1)
    A00 = np.concatenate((A00, A00), axis=0)
    'A00: (sensor_num*2, sensor_num*2)'
    A10 = np.concatenate((A3, A3), axis=1)
    'A10: (5+4, sensor_num*2)'
    A01 = np.concatenate((A2, A2), axis=0)
    'A01: (sensor_num*2, 5+4)'
    A11 = A1
    'A11: (5+4, 5+4)'
    A = np.concatenate((np.concatenate((A00, A01), axis=1), np.concatenate((A10, A11), axis=1),), axis=0)
    'A: (sensor_num*2+5+4, sensor_num*2+5+4)'

    return A



def graph_calculate_from_prior(args, if_return_norm=False):
    """
    该函数直接从数据集文件夹导入数据集邻接矩阵文件

    Args:
        args:

    Returns:
        A: (node_num, node_num)非自连接的01邻接矩阵
        A_self: (node_num, node_num)自连接的01邻接矩阵
        A_norm: (node_num, node_num)非自连接的归一化邻接矩阵，如果if_return_norm=True则计算
        A_self_norm: (node_num, node_num)自连接的归一化邻接矩阵，如果if_return_norm=True则计算
    """
    "一切开始之前，先定义文件路径，以.csv结尾"
    A_file_path = os.path.join(args.root_path, args.data_path, 'adjacency_matrix/Prior_Adj_01_of_XJTU_SPS_for_Modeling_and_PINN.csv')
    if args.if_timestamp:
        A_file_path = os.path.join(args.root_path, args.data_path,
                                   'adjacency_matrix/Prior_Adj_01_of_XJTU_SPS_for_Modeling_and_PINN_with_t.csv')
    if args.if_add_work_condition:
        A_file_path = os.path.join(args.root_path, args.data_path,
                                   'adjacency_matrix/Prior_Adj_01_of_XJTU_SPS_for_Modeling_and_PINN_with_w.csv')
    if args.if_add_work_condition and args.if_timestamp:
        A_file_path = os.path.join(args.root_path, args.data_path,
                                   'adjacency_matrix/Prior_Adj_01_of_XJTU_SPS_for_Modeling_and_PINN_with_t_and_w.csv')
    "读取A文件"
    A = pd.read_csv(A_file_path, index_col=0, header=0).values
    "如果用的是PINN，得要将sensor+5+4维度的A 进行拆分拼接成sensor*2+5+4"
    # 对角线置0
    np.fill_diagonal(A, 0)
    "计算A_self"
    A_self = A + np.eye(A.shape[0]).astype(np.float32)
    "计算A_norm和A_self_norm"
    if args.model_selection == 'SPS_Model_PINN':
        A = get_PINN_A(args, A)
        'A: (sensor_num*2+5+4, sensor_num*2+5+4)'
        A_self = get_PINN_A(args, A_self)
        'A_self: (sensor_num*2+5+4, sensor_num*2+5+4)'
    if if_return_norm:
        # 每行相加并对角化为度矩阵
        degree = np.sum(A, axis=1)
        # D = np.diag(degree)
        degree_inv_sqrt = np.power(degree, -0.5)
        degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0
        D_inv_sqrt = np.diag(degree_inv_sqrt)
        # 归一化邻接矩阵
        A_norm = np.matmul(np.matmul(D_inv_sqrt, A), D_inv_sqrt)
        # 自连接的归一化邻接矩阵
        degree_self = np.sum(A_self, axis=1)
        degree_self_inv_sqrt = np.power(degree_self, -0.5)
        degree_self_inv_sqrt[np.isinf(degree_self_inv_sqrt)] = 0
        D_self_inv_sqrt = np.diag(degree_self_inv_sqrt)
        A_self_norm = np.matmul(np.matmul(D_self_inv_sqrt, A_self), D_self_inv_sqrt)

        return A, A_self, A_norm, A_self_norm
    else:
        return A, A_self



def Graph_calculate(args, if_return_norm=False, flag='train'):
    """
    该函数用于通过一段长度为cal_len多维X的数据，计算邻接矩阵，需要放在数据集创建的那个文件中，
    在读入全部原始数据并完成了相应的归一化等等操作以后，将标准化后数据以及用于计算的长度输入，输出01邻接矩阵
    至于计算邻接矩阵的方法，可选MIC和Copent，分别是最大信息系数和Copula熵

    Args:
        args: 参数集合
        if_return_norm: 是否计算并返回归一化后的邻接矩阵和自连接的归一化邻接矩阵
        flag: 现在正在train、val、test哪个阶段，用于判断是否需要计算A_w还是直接读取文件

    Returns:
        A: (node_num, node_num)非自连接的01邻接矩阵
        A_self: (node_num, node_num)自连接的01邻接矩阵
        A_w: (node_num, node_num)权重邻接矩阵，即相关性矩阵，可以做可视化用
        A_norm: (node_num, node_num)非自连接的归一化邻接矩阵，如果if_return_norm=True则计算
        A_self_norm: (node_num, node_num)自连接的归一化邻接矩阵，如果if_return_norm=True则计算
    """
    "一切开始之前，先定义A_w保存路径，以.csv结尾"
    if args.graph_ca_meth == "MIC":
        file_name = args.graph_ca_meth + "_a" + str(args.MIC_alpha) + "_c" + str(args.MIC_c) \
                    + "_ca_len" + str(args.graph_ca_len)
    else:
        file_name = args.graph_ca_meth + "_ca_len" + str(args.graph_ca_len)
    if args.Decompose == 'STL':
        De_info = "_" + args.Decompose + "_season" + str(args.STL_seasonal)
        file_name = file_name + De_info
    else:
        De_info = "_" + args.Decompose + "_" + str(args.Wavelet_wave) + "_lv" + str(args.Wavelet_level)
        file_name = file_name + De_info
    if args.preMA:
        other_info = "_scale" + str(args.scale) + "_preMA_win" + str(args.preMA_win) \
                     + "_if_t" + str(args.if_timestamp) + "_if_w" + str(args.if_add_work_condition)
    else:
        other_info = "_scale" + str(args.scale) + "_preMA" + str(args.preMA) \
                     + "_if_t" + str(args.if_timestamp) + "_if_w" + str(args.if_add_work_condition)
    file_name = file_name + other_info
    csv_dir = args.table_save_path + '/' + file_name + "_A_w.csv"


    # 计算A_w
    if os.path.exists(csv_dir):
        A_w = pd.read_csv(csv_dir, header=None).values
    else:
        # 读取数据
        data_normal = read_data_for_graph_calculate(args)
        'data_normal: (graph_ca_len, node_num)'
        "如果是train阶段，则计算A_w"
        A_w = A_w_calculate(args, data_normal)
        "训练时，保存A_w为csv文件，并绘制heatmap"
        A_w_csv_and_plot(args, A_w, csv_dir)
    # if flag == 'train':
    #     "如果是train阶段，则计算A_w"
    #     A_w = A_w_calculate(args, data_normal)
    #     "训练时，保存A_w为csv文件，并绘制heatmap"
    #     A_w_csv_and_plot(args, A_w, csv_dir)
    # elif flag in ['val', 'test']:
    #     "如果是val阶段，这里有个特殊情况，说明如下，但最终采用策略是：如果有A_w文件，则直接读取，否则计算A_w，但全程不保存"
    #     # 按理lighting会该先调用train_dataloader，然后再调用val_dataloader，
    #     # 但实际情况是lighting为了不需要等待漫长的训练过程才发现验证代码有错，https://zhuanlan.zhihu.com/p/120331610
    #     # 会在开始加载dataloader并开始训练时，就提前执行 “验证代码”：val_dataloader、validation_step、validation_epoch_end.
    #     # 这会导致此时还没有train_dataloader，还没有计算A_w，会报错，
    #     # 所以这里如果没有A_w文件，就先计算一个，但是不保存，只用来最初的走通验证代码
    #     "如果A_w文件存在，则直接读取,否则计算A_w"
    #     if os.path.exists(csv_dir):
    #         A_w = pd.read_csv(csv_dir, header=None).values
    #     else:
    #         A_w = A_w_calculate(args, data_normal)
    #         print("there is no A_w file, so calculate A_w in {}".format(flag))
    # # elif flag == 'test':
    # #     "测试阶段，直接读取A_w文件，用train阶段的A_w文件"
    # #     "如果A_w文件存在，则直接读取"
    # #     if os.path.exists(csv_dir):
    # #         A_w = pd.read_csv(csv_dir, header=None).values
    # #     else:
    # #         raise ValueError("A_w file does not exist, please check why test stage dont have this file")


    "根据A_w计算A、A_self、A_norm、A_self_norm"
    if if_return_norm:
        A, A_self, A_w, A_norm, A_self_norm = A_other_calculate(args, A_w, if_return_norm)
    else:
        A, A_self, A_w = A_other_calculate(args, A_w, if_return_norm)

    if if_return_norm:
        return A, A_self, A_w, A_norm, A_self_norm
    else:
        return A, A_self, A_w



def read_data_for_graph_calculate(args):
    """
    该函数要与dataset保持统一

    Args:
        args: 参数集合

    Returns:
        data_normal: (all_len, node_num)标准化后的数据集
    """
    """导入数据"""
    try:
        # f = open(os.path.join(args.root_path, args.data_path, '{}.pkl'.format(args.data_name)), "rb")
        # data = pickle.load(f).values.reshape((-1, x_dim))
        # f.close()
        # 首先读取os.path.join(args.root_path, args.data_path, '{}.csv'.format(args.data_name)
        # 若失败，将args.data_path的第一个斜杠及其前内容删掉再试，不需要报错
        try:
            data_df = pd.read_csv(os.path.join(args.root_path, args.data_path, '{}.csv'.format(args.data_name)),
                               sep=',', index_col=False)
        except:
            new_data_path = args.data_path[args.data_path.find('/')+1:]
            data_df = pd.read_csv(os.path.join(args.root_path, new_data_path, '{}.csv'.format(args.data_name)),
                               sep=',', index_col=False)
        if args.features == 'S':
            data = data_df[[args.target]].values
        else:
            data = data_df.drop(['Time'], axis=1).values

        # 读取Work_Condition数据
        Work_Condition_data_name = args.data_name + '_Work_Condition'
        try:
            work_condition_df = pd.read_csv(os.path.join(args.root_path, args.data_path, '{}.csv'.format(Work_Condition_data_name)),
                                            sep=',', index_col=False)
        except:
            new_data_path = args.data_path[args.data_path.find('/') + 1:]
            work_condition_df = pd.read_csv(os.path.join(args.root_path, new_data_path, '{}.csv'.format(Work_Condition_data_name)),
                                            sep=',', index_col=False)
        work_condition = work_condition_df.drop(['Time'], axis=1).values

        # # 如果是在SPS_Model_PINN时候，可以直接使用物理信息的仿真结果
        if args.model_selection == 'SPS_Model_PINN':
            if args.SPS_Model_PINN_if_has_Phy_of_BCR:
                file_name = str(args.data_name) + '_reconstruct' + '_physical_simulate_result.csv'
                PI_data_df = pd.read_csv(os.path.join(args.root_path, args.data_path, file_name),
                                            sep=',', index_col=False)
            else:
                file_name = str(args.data_name) + '_reconstruct' + '_physical_simulate_result_withoutBCR.csv'
                PI_data_df = pd.read_csv(os.path.join(args.root_path, args.data_path, file_name),
                                         sep=',', index_col=False)
            PI_data = PI_data_df.values

    except (KeyError, FileNotFoundError):
        data = None

    """df_stamp是时间标签信息"""
    df_stamp = data_df[['Time']]
    df_stamp['Time'] = pd.to_datetime(df_stamp.Time)

    df_stamp['day'] = df_stamp.Time.apply(lambda row: row.day, 1)
    df_stamp['hour'] = df_stamp.Time.apply(lambda row: row.hour, 1)
    df_stamp['minute'] = df_stamp.Time.apply(lambda row: row.minute, 1)
    df_stamp['second'] = df_stamp.Time.apply(lambda row: row.second, 1)
    data_stamp = df_stamp.drop(['Time'], axis=1).values
    # data_stamp再补一列timestamp，从0到len(data_stamp)
    data_stamp = np.concatenate([data_stamp, np.arange(len(data_stamp)).reshape(-1, 1)], axis=1)

    """***数据划分***"""
    # 一个周期95min，一共4个周期，前三个周期是训练集，前三周期随便选一个兼做验证集，最后一个周期是测试集
    train_len = int(args.dataset_split_ratio * len(data))
    border1s = [0, train_len-(train_len//8), train_len]
    border2s = [train_len-(train_len//8), train_len, len(data)]
    # train_len = int(len(data) * 0.75)
    # border1s = [0, int(train_len/3), train_len]
    # border2s = [train_len, 2*int(train_len/3), len(data)]
    border1 = border1s[0]
    border2 = border2s[0]
    # 切割数据
    data = data[border1:border2]
    data_stamp = data_stamp[border1:border2]
    work_condition = work_condition[border1:border2]
    PI_data = PI_data[border1:border2] if args.model_selection == 'SPS_Model_PINN' else None

    """不使用全部数据，只能使用截取部分数据，根据args.only_use_data_ratio"""
    # 采样频率是args.exp_frequency，截取的时候，由于仿真初始值的设置，必须截取95min的倍数，也就是数据点数目为95*60*args.exp_frequency的整数倍
    if args.only_use_data_ratio < 1:
        lim_len = int(int(len(data)*args.only_use_data_ratio) // (95*60*args.exp_frequency) * (95*60*args.exp_frequency))
        data = data[-lim_len:]
        data_stamp = data_stamp[-lim_len:]
        work_condition = work_condition[-lim_len:]
        PI_data = PI_data[-lim_len:] if args.model_selection == 'SPS_Model_PINN' else None

    """***preprocessing***"""

    """数据标准化归一化"""
    if args.scale:
        # norm_data, data, scale_list, mean_list = args.normalize(data, args.flag, args.scaler)
        # norm_data_stamp, _, _, _ = args.normalize(data_stamp, args.flag, args.timestamp_scaler)
        # norm_work_condition, work_condition, scale_list_work_condition, mean_list_work_condition = args.normalize(work_condition, args.flag,  args.work_condition_scaler)
        norm_data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
        norm_data_stamp = (data_stamp - np.min(data_stamp, axis=0)) / (np.max(data_stamp, axis=0) - np.min(data_stamp, axis=0) + 1e-8)
        norm_work_condition = (work_condition - np.mean(work_condition, axis=0)) / (np.std(work_condition, axis=0) + 1e-8)
    else:
        norm_data = data
        # scale_list = [1.0] * data.shape[1]
        # mean_list = [0.0] * data.shape[1]
        norm_data_stamp = data_stamp
        norm_work_condition = work_condition
        # scale_list_work_condition = [1.0] * work_condition.shape[1]
        # mean_list_work_condition = [0.0] * work_condition.shape[1]

    """进行数据缺失"""
    miss_data, miss_norm_data = make_missing_data(data, args.missing_rate, args.missvalue, norm_data) \
        if args.missing_rate > 0 else (data, norm_data)

    """nan填充:用前一个或者后一个时间步进行nan填充"""
    if np.isnan(data).any():
        data = nan_filling(data)
    if np.isnan(norm_data).any():
        norm_data = nan_filling(norm_data)
    if np.isnan(miss_data).any():
        miss_data = nan_filling(miss_data) if args.missing_rate > 0 else data
    if np.isnan(miss_norm_data).any():
        miss_norm_data = nan_filling(miss_norm_data) if args.missing_rate > 0 else norm_data

    """加入噪声"""
    # 注意这里用的信噪比不是比例而是dB，因为领域内常用的是dB，设置时要注意  https://blog.csdn.net/qq_58860480/article/details/140583800
    if args.add_noise_SNR > 0:
        signal_power = np.mean(data ** 2)
        noise_power = signal_power / (10 ** (args.add_noise_SNR / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), data.shape)
        data = data + noise
        norm_data = norm_data + noise
        miss_data = miss_data + noise if args.missing_rate > 0 else data
        miss_norm_data = miss_norm_data + noise if args.missing_rate > 0 else norm_data

    """含噪数据滑动平均预处理"""
    if args.preMA:
        data = preMA(data, args.preMA_win)
        norm_data = preMA(norm_data, args.preMA_win)
        miss_data = preMA(miss_data, args.preMA_win) if args.missing_rate > 0 else data
        miss_norm_data = preMA(miss_norm_data, args.preMA_win) if args.missing_rate > 0 else norm_data

    """数据定型"""
    out_data = miss_norm_data if args.scale else miss_data
    out_data_stamp = norm_data_stamp if args.scale else data_stamp
    out_work_condition = norm_work_condition if args.scale else work_condition
    if args.model_selection == 'SPS_Model_PINN':
        out_data = np.concatenate([out_data, PI_data], axis=1)
    if args.if_timestamp:
        out_data = np.concatenate([out_data, out_data_stamp], axis=1)
    if args.if_add_work_condition:
        out_data = np.concatenate([out_data, out_work_condition], axis=1)

    """取graph_ca_len长度的数据"""
    out_data = out_data[:args.graph_ca_len, :]
    'out_data: (graph_ca_len, node_num)'

    return out_data




