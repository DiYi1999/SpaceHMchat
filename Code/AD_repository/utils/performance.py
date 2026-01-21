from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from AD_repository.utils.data import *
import numpy as np
from scipy.stats import iqr
import torch
import os
import json
import pandas as pd


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / (true + 1e-8)))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true + 1e-8))

def performance_FC(pred, true):
    mse = MSE(pred, true)
    mae = MAE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    return mse, mae, rmse, mape, mspe, rse, corr


def performance_AD_auto(total_err_scores, gt_labels, topk=1, focus_on='F1', threshold=None):
    """
    在实际使用时是没有真实标签的，这个函数主要是为开发者提供验证功能，同时你也可以在这里开发一些阈值推荐算法，你所计算出的推荐阈值将在对话界面中返回给用户。
    In actual use, there are no true labels. This function is mainly to provide verification functions for developers. At the same time, you can also develop some threshold recommendation algorithms here. The recommended thresholds you calculate will be returned to users in the dialogue interface.

    :param total_err_scores:  (node_num, all_len)
    :param gt_labels: true labels, (all_len,)
    :param topk:
    :param focus_on: 
    :param threshold: 

    :return: F1, acc, pre, rec, auc_score, recommend_threshold
    """

    if total_err_scores.shape[0] > total_err_scores.shape[1]:
        total_err_scores = np.transpose(total_err_scores)
        """(node_num, all_len)"""
    total_features = total_err_scores.shape[0]
    """node_num"""

    # topk_indices = np.argpartition(total_err_scores, range(total_features-1-topk, total_features-1), axis=0)[-topk-1:-1]
    topk_indices = np.argpartition(total_err_scores, range(total_features - topk - 1, total_features), axis=0)[-topk:]
    """(k, all_len)
    total_err_scores里面每列最大的k个 的索引，按列来的，k是1的话，就相当于找到每列27个最大的那个的索引
    即：按列合并scores，每个时刻的多个传感器里只选异常分数最大的那个分数作为该时刻的最终异常分数
    """
    # np.argpartition返回的索引是把total_err_scores的第total_features-topk-1位到第total_features位（就是最后k位）填上他该有的数（按排序大小），最后[-topk:]取出来最大的k个

    total_topk_err_scores = []
    """就是各个时刻的异常分（2044，），k=1时是把每个时刻的多个传感器里只选异常分数最大的那个分数作为该时刻的最终异常分数，k=k时是把每个时刻的多个传感器里只选k个异常分数最大的那几个分数相加的和作为该时刻的最终异常分数"""
    topk_err_score_map = []

    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)
    """(all_len,)"""
    # np.take_along_axis(a, indices, axis)是把a的axis轴上的indices索引取出来，然后相加

    pred_labels = np.zeros(len(total_topk_err_scores))
    """使用计算出最大F1的阈值时 的预测结果0、1列"""
    pred_labels[total_topk_err_scores > threshold] = 1

    # for i in range(len(pred_labels)):
    #     pred_labels[i] = int(pred_labels[i])
    #     gt_labels[i] = int(gt_labels[i])

    acc = accuracy_score(gt_labels, pred_labels)
    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)
    f1_score1 = f1_score(gt_labels, pred_labels)

    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)

    print(f'f1,acc,pre,rec:{f1_score1, acc, pre, rec}')

    ############# Here you can create some threshold recommendation argorithm ###############
    recommend_threshold = None

    return f1_score1, acc, pre, rec, auc_score, recommend_threshold




def detect_AD_fragment(Score_AD, threshold, json_save_path, timestamp_label, recommend_threshold=None, mean_or_each='each'):
    """
    将异常分数与阈值进行比较，判断异常区域，对应timestamp_list得到检测到的异常时间片段列表，并将阈值、异常占比、异常时间片段等各种信息 保存为json文件。

    :param Score_AD: (node_num, all_len), 传感器异常分数
    :param threshold: 阈值
    :param json_save_path: 保存json的路径
    :param timestamp_label: 时间戳列表，(all_len), 时间戳
    :param recommend_threshold: 推荐的阈值
    :param mean_or_each: 'mean' or 'each', 是否对所有传感器的异常分数取平均值，还是每个传感器单独计算
    """

    # 转化为timedelta
    if timestamp_label is not None:
        if isinstance(timestamp_label, torch.Tensor):
            timestamp_label = timestamp_label.cpu().numpy()
        if isinstance(timestamp_label, np.ndarray):
            # timestamp_label = pd.to_timedelta(timestamp_label, unit='s')
            timestamp_label = pd.to_datetime(timestamp_label, unit='s')
            # timestamp_label = pd.to_datetime(timestamp_label)
            # timestamp_label = pd.to_datetime(timestamp_label.astype(np.int64))
        else:
            raise ValueError("timestamp_label must be a torch.Tensor or np.ndarray")
    # one_x_width = timestamp_label[1] - timestamp_label[0] if timestamp_label is not None else 1

    if mean_or_each == 'mean':
        Score_AD = np.mean(Score_AD, axis=0)
        """(all_len,)"""
        Score_AD = np.squeeze(Score_AD)
        """(all_len,)"""
        Over_thre = Score_AD >= threshold
        """(all_len,)"""
    elif mean_or_each == 'each':
        Over_thre = Score_AD >= threshold
        """(node_num, all_len)"""
        Over_thre = np.sum(Over_thre, axis=0) > 0
        """(all_len,)"""
    else:
        raise ValueError("mean_or_each must be 'mean' or 'each'")
    
    fragments = []
    start_idx = None  # 持续异常的起始索引
    for i, is_over in enumerate(Over_thre):
        if is_over:
            if start_idx is None:
                start_idx = i
        else:
            if start_idx is not None:
                if start_idx == i - 1:
                    fragments.append([str(timestamp_label[start_idx])])
                else:
                    fragments.append([str(timestamp_label[start_idx]), str(timestamp_label[i - 1])])
                start_idx = None
    if start_idx is not None:  # 如果最后一个片段没有结束
        if start_idx == len(Over_thre) - 1:
            fragments.append([str(timestamp_label[start_idx])])
        else:
            fragments.append([str(timestamp_label[start_idx]), str(timestamp_label[-1])])

    # 计算异常占比
    anomaly_count = sum(Over_thre)
    total_count = len(Over_thre)
    anomaly_ratio = anomaly_count / total_count if total_count > 0 else 0

    # 准备要保存的字典
    AD_result = {
        "threshold": float(threshold),
        "anomaly_ratio": float(anomaly_ratio),
        "anomaly_timestamp_list": fragments,
        "recommend_threshold": recommend_threshold
    }
    # 将结果保存为JSON文件
    json_save_path = os.path.join(json_save_path, "AD_result.json")
    if not os.path.exists(os.path.dirname(json_save_path)):
        os.makedirs(os.path.dirname(json_save_path))
    with open(json_save_path, 'w') as f:
        json.dump(AD_result, f, indent=4)
    # print(f"AD results saved to {json_save_path}")
    
    return fragments, threshold, anomaly_ratio
    
