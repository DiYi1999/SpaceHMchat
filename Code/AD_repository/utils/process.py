import argparse

import numpy as np
import torch
import pandas as pd


def moving_average(anomaly_score_tensor, window_num):
    """
    对异常评分的每一列分别进行移动平均，平滑异常评分曲线

    Args:
        anomaly_score_tensor: (all_len, node_num)异常打分
        window_num: 滑动平均窗长

    Returns:

    """
    # 输入若是（node_num, all_len）则按行进行移动平均
    if anomaly_score_tensor.shape[0] < anomaly_score_tensor.shape[1]:
        # 先转numpy
        anomaly_score_array = anomaly_score_tensor.cpu().numpy()
        # 逐行移动平均
        for i in range(anomaly_score_tensor.shape[0]):
            # 滑动平均
            anomaly_score_array[i, :] = np.convolve(anomaly_score_array[i, :], np.ones(window_num) / window_num, mode='same')
        # 再转回tensor
        normal_anomaly_score_tensor = torch.from_numpy(anomaly_score_array).float().to(anomaly_score_tensor.device)
    # 输入若是（all_len, node_num）则按列进行移动平均
    else:
        # 先转成numpy
        anomaly_score_array = anomaly_score_tensor.cpu().numpy()
        # 逐列移动平均
        for i in range(anomaly_score_tensor.shape[1]):
            # 滑动平均
            anomaly_score_array[:, i] = np.convolve(anomaly_score_array[:, i], np.ones(window_num) / window_num, mode='same')
        # 再转回tensor
        normal_anomaly_score_tensor = torch.from_numpy(anomaly_score_array).float().to(anomaly_score_tensor.device)

    return normal_anomaly_score_tensor


# 用前方最近的观测值填补缺失值
def preIDW(data):
    """
    用前方最近的观测值填补缺失值

    Args:
        data: (all_len, node_num)，numpy数组

    Returns:

    """
    df_data = pd.DataFrame(data) if isinstance(data, np.ndarray) else data
    # (all_len, node_num)
    df_data = df_data.fillna(method="ffill", axis=0)
    df_data = df_data.fillna(method="backfill", axis=0)
    data = df_data.values if isinstance(data, np.ndarray) else df_data

    return data


# # 用前方最近的观测值填补缺失值，这个是用在batch上的，针对的是(batch_size, node_num, lag)的torch tensor，弃用了
# def preIDW(x_batch):
#     x_data = x_batch.cpu().numpy()
#     # (batch_size, node_num, lag)
#     for i in range(x_data.shape[0]):
#         i_batch = pd.DataFrame(x_data[i])
#         # (node_num, lag)
#         i_batch = i_batch.fillna(method="ffill", axis=1)
#         i_batch = i_batch.fillna(method="backfill", axis=1)
#         x_data[i] = i_batch.values
#     return torch.FloatTensor(x_data).type_as(x_batch)



def remove_outliers(data0, factor=1):
    """
    IQR方法去除异常值，但是不是去掉，而是用最近的正常值代替。但这个不是用的1、3分位数，而是用的5、95分位数，factor用于计算异常值的上下界：lower_bound = Q5 - factor * IQR、 upper_bound = Q95 + factor * IQR

    Args:
        data: (all_len, node_num)，numpy数组
        factor: 阈值

    Returns:

    """
    if isinstance(data0, torch.Tensor):
        data = data0.cpu().numpy()
    if isinstance(data0, pd.DataFrame):
        data = data0.values
    if isinstance(data0, np.ndarray):
        data = data0

    # IQR方法去除异常值，但是不是去掉，而是用最近的正常值代替
    for i in range(data.shape[1]):
        # 第i列
        col = data[:, i]
        # 计算Q1, Q3
        Q10 = np.percentile(col, 10)
        Q90 = np.percentile(col, 90)
        # 计算IQR
        IQR = Q90 - Q10
        # 计算异常值的上下界
        lower_bound = Q10 - factor * IQR
        upper_bound = Q90 + factor * IQR
        # 先把异常值替换成nan
        col[col < lower_bound] = np.nan
        col[col > upper_bound] = np.nan
        # 用前方最近的观测值填补缺失值
        col = pd.DataFrame(col).fillna(method="ffill", axis=0).values
        col = pd.DataFrame(col).fillna(method="backfill", axis=0).values
        data[:, i] = col

    if isinstance(data, torch.Tensor):
        data = torch.FloatTensor(data).to(data0.device)
    if isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data, columns=data0.columns)
    if isinstance(data, np.ndarray):
        data = data

    return data



# def preMA(data_array, window_size=50):
#     """
#     move_average对data的每一列分别进行滑动平均，但是不是滑窗式，而是将每隔window_size内的数据全部用这个window_size内的平均值代替

#     Args:
#         data_array: (lag, node_num)，numpy数组
#         window_size:

#     Returns:

#     """
#     # 先将data转换成numpy array
#     # data_array = data.values

#     # 遍历每一列
#     for i in range(data_array.shape[1]):
#         # 遍历每一个窗口
#         for j in range(data_array.shape[0] // window_size):
#             # 每一个窗口的起始和终止索引
#             start_index = j * window_size
#             end_index = (j + 1) * window_size
#             # 每一个窗口的平均值
#             mean = np.mean(data_array[start_index:end_index, i])
#             # 将这个窗口内的数据全部用这个平均值代替
#             data_array[start_index:end_index, i] = mean
#         # 最后一个窗口的处理
#         start_index = (data_array.shape[0] // window_size) * window_size
#         end_index = data_array.shape[0]
#         mean = np.mean(data_array[start_index:end_index, i])
#         data_array[start_index:end_index, i] = mean
#     # 最后将numpy array转换成pandas dataframe
#     # data_out = pd.DataFrame(data_array, columns=data.columns)

#     return data_array

def preMA(data_array, window_size=50):
    """
    上面那个函数不是滑窗式，而是将每隔window_size内的数据全部用这个window_size内的平均值代替，计算效率高但是不精确
    现在改成真正的滑窗式的移动平均

    Args:
        data_array: (lag, node_num)，numpy数组
        window_size:

    Returns:

    """
    # 先将data转换成numpy array
    # data_array = data.values

    result = np.copy(data_array)

    # 遍历每一列
    for i in range(data_array.shape[1]):
        # 使用pandas的rolling函数进行滑动平均
        series = pd.Series(data_array[:, i])
        # center=True使窗口以当前点为中心
        smoothed = series.rolling(window=window_size, center=True).mean()        
        # 处理开头和结尾的NaN值
        smoothed = smoothed.fillna(method='bfill').fillna(method='ffill')        
        # 将结果赋值回数组
        result[:, i] = smoothed.values

    # 最后将numpy array转换成pandas dataframe
    # data_out = pd.DataFrame(data_array, columns=data.columns)

    return result



def make_missing_data(data, missing_rate, missvalue, norm_data=None):
    """
    生成缺失数据

    Args:
        data: (all_len, node_num), numpy数组
        missing_rate: 缺失率
        missvalue: 缺失值
        norm_data: (all_len, node_num), 归一化后的数据，可以不输入

    Returns:

    """
    # 如果是pandas dataframe则转换成numpy array
    missing_data = data.copy() if isinstance(data, np.ndarray) else data.values.copy()
    # 缺失数据的数量
    missing_num = int(missing_data.size * missing_rate)
    # 缺失数据的位置
    missing_position = np.random.choice(missing_data.size, missing_num, replace=False)
    # 缺失数据的位置转换成二维
    missing_position_2d = np.unravel_index(missing_position, missing_data.shape)
    # 缺失数据的位置赋值
    missing_data[missing_position_2d] = missvalue
    # 如果是pandas dataframe则转换成pandas dataframe
    missing_data = pd.DataFrame(missing_data, columns=data.columns) if isinstance(data, pd.DataFrame) else missing_data

    if norm_data is not None:
        missing_norm_data = norm_data.copy() if isinstance(norm_data, np.ndarray) else norm_data.values.copy()
        missing_norm_data[missing_position_2d] = missvalue
        missing_norm_data = pd.DataFrame(missing_norm_data, columns=norm_data.columns) if isinstance(norm_data, pd.DataFrame) else missing_norm_data
        return missing_data, missing_norm_data

    return missing_data


def nan_filling(data):
    """
    用前方最近的观测值填补缺失值

    Args:
        data: (all_len, node_num)，numpy数组

    Returns:

    """
    df_data = pd.DataFrame(data) if isinstance(data, np.ndarray) else data
    # (all_len, node_num)
    df_data = df_data.fillna(method="ffill", axis=0)
    df_data = df_data.fillna(method="backfill", axis=0)
    data = df_data.values if isinstance(data, np.ndarray) else df_data

    return data
