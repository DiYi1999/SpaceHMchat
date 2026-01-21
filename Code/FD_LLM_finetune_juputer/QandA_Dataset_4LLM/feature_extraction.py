import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import pandas as pd





def temporal_feature_extract(X):
    """

    Args:
        X: (batch_size, node_num, lag)

    Returns:

    """
    data_type = type(X)
    if not isinstance(X, torch.Tensor):
        if data_type == pd.DataFrame:
            X = torch.tensor(X.values, dtype=torch.float32)
        elif data_type == np.ndarray:
            X = torch.tensor(X, dtype=torch.float32)
        else:
            raise ValueError("Unsupported data type. Please provide a Tensor, DataFrame, or ndarray.")

    if_2D = X.dim() == 2
    if if_2D:
        X = X.unsqueeze(0).permute(0, 2, 1)
        'X: (lag, node_num) -> (1, node_num, lag)'

    batch_size, node_num, lag = X.size()

    X_mean = torch.mean(X, dim=2)
    '均值 Mean value：(batch_size, node_num)'
    X_abs_mean = torch.mean(torch.abs(X), dim=2)
    '绝对平均值 Absolute mean value：(batch_size, node_num)'   # 可考虑省略
    X_var = torch.var(X, dim=2)
    '方差 Variance：(batch_size, node_num)'   # 可考虑省略
    X_max = torch.max(X, dim=2)[0]
    '最大值：(batch_size, node_num)'   # 可考虑省略
    X_min = torch.min(X, dim=2)[0]
    '最小值：(batch_size, node_num)'   # 可考虑省略
    X_std = torch.std(X, dim=2)
    '标准差 Standard deviation：(batch_size, node_num)'
    X_smra = (torch.mean(torch.sqrt(torch.abs(X)), dim=2))**2
    '方根幅值 Square root amplitude：(batch_size, node_num)'
    X_rms = torch.sqrt(torch.mean(X**2, dim=2))
    '均方根值：(batch_size, node_num)'
    X_p = torch.max(torch.abs(X), dim=2)[0]
    '峰值 Peak value：(batch_size, node_num)'
    X_ptp = torch.max(X, dim=2)[0] - torch.min(X, dim=2)[0]
    '峰-峰值：(batch_size, node_num)'
    X_CrestFactor = X_p / X_rms
    '峰值因子 Peak index：(batch_size, node_num)'
    X_WaveFactor = X_rms / X_mean
    '波形因子 Waveform index：(batch_size, node_num)'
    X_ImpulseFactor = X_p / X_mean
    # X_ImpulseFactor = X_p / X_abs_mean   # 也有除绝对平均值的
    '脉冲因子：(batch_size, node_num)'
    X_ClearanceFactor = X_p / X_smra
    '裕度因子：(batch_size, node_num)'
    X_Skewness = torch.mean((X - X_mean.unsqueeze(2))**3, dim=2) / (X_std**3)
    '偏度 Skewness：(batch_size, node_num)' # 这里讲实话应该除以N-1的
    X_Kurtosis = torch.mean((X - X_mean.unsqueeze(2))**4, dim=2) / (X_std**4)
    '峭度 Kurtosis：(batch_size, node_num)'
    # X_KurtosisFactor = X_Kurtosis / X_Skewness
    # '峭度因子 Kurtosis index：(batch_size, node_num)'   # 这个省略吧没啥用

    # # 所有数字保留三位小数
    # X_mean = torch.round(X_mean * 1000) / 1000
    # X_abs_mean = torch.round(X_abs_mean * 1000) / 1000
    # X_var = torch.round(X_var * 1000) / 1000
    # X_max = torch.round(X_max * 1000) / 1000
    # X_min = torch.round(X_min * 1000) / 1000
    # X_std = torch.round(X_std * 1000) / 1000
    # X_smra = torch.round(X_smra * 1000) / 1000
    # X_rms = torch.round(X_rms * 1000) / 1000
    # X_p = torch.round(X_p * 1000) / 1000
    # X_ptp = torch.round(X_ptp * 1000) / 1000
    # X_CrestFactor = torch.round(X_CrestFactor * 1000) / 1000
    # X_WaveFactor = torch.round(X_WaveFactor * 1000) / 1000
    # X_ImpulseFactor = torch.round(X_ImpulseFactor * 1000) / 1000
    # X_ClearanceFactor = torch.round(X_ClearanceFactor * 1000) / 1000
    # X_Skewness = torch.round(X_Skewness * 1000) / 1000
    # X_Kurtosis = torch.round(X_Kurtosis * 1000) / 1000


    if if_2D:
        X_mean = X_mean.squeeze(0)
        '(batch_size, node_num) -> (node_num)'
        X_abs_mean = X_abs_mean.squeeze(0)
        X_var = X_var.squeeze(0)
        X_max = X_max.squeeze(0)
        X_min = X_min.squeeze(0)
        X_std = X_std.squeeze(0)
        X_smra = X_smra.squeeze(0)
        X_rms = X_rms.squeeze(0)
        X_p = X_p.squeeze(0)
        X_ptp = X_ptp.squeeze(0)
        X_CrestFactor = X_CrestFactor.squeeze(0)
        X_WaveFactor = X_WaveFactor.squeeze(0)
        X_ImpulseFactor = X_ImpulseFactor.squeeze(0)
        X_ClearanceFactor = X_ClearanceFactor.squeeze(0)
        X_Skewness = X_Skewness.squeeze(0)
        X_Kurtosis = X_Kurtosis.squeeze(0)

    # 创建成字典
    feature_dict = {'时域平均值': X_mean.numpy(),
                    # '时域绝对平均值': X_abs_mean.numpy(),
                    # '时域方差': X_var.numpy(),
                    # '时域最大值': X_max.numpy(),
                    # '时域最小值': X_min.numpy(),
                    '时域标准差': X_std.numpy(),
                    '时域方根幅值': X_smra.numpy(),
                    '时域均方根值': X_rms.numpy(),
                    '时域峰值': X_p.numpy(),
                    # '时域峰-峰值': X_ptp.numpy(),
                    # '时域峰值因子': X_CrestFactor.numpy(),
                    # '时域波形因子': X_WaveFactor.numpy(),
                    # '时域脉冲因子': X_ImpulseFactor.numpy(),
                    # '时域裕度因子': X_ClearanceFactor.numpy(),
                    # '时域偏度': X_Skewness.numpy(),
                    # '时域峭度': X_Kurtosis.numpy()
                    }

    return feature_dict








def frequency_feature_extract(X, exp_frequency=1):
    """

    Args:
        X: (batch_size, node_num, lag)

    Returns:

    """
    data_type = type(X)
    if not isinstance(X, torch.Tensor):
        if data_type == pd.DataFrame:
            X = torch.tensor(X.values, dtype=torch.float32)
        elif data_type == np.ndarray:
            X = torch.tensor(X, dtype=torch.float32)
        else:
            raise ValueError("Unsupported data type. Please provide a Tensor, DataFrame, or ndarray.")

    if_2D = X.dim() == 2
    if if_2D:
        X = X.unsqueeze(0).permute(0, 2, 1)
        'X: (lag, node_num) -> (1, node_num, lag)'

    batch_size, node_num, lag = X.size()

    X_fft = torch.fft.rfft(X, n=N, dim=2)
    'FFT：(batch_size, node_num, lag//2+1)'

    X_fft_abs = torch.abs(X_fft)
    'FFT幅值：(batch_size, node_num, lag//2+1)'
    X_fft_abs = X_fft_abs * 2 / N  # 归一化
    X_fft_abs[..., 0] = X_fft_abs[..., 0] / 2  # 直流分量特殊处理
    X_fft_angle = torch.angle(X_fft)
    'FFT相位：(batch_size, node_num, lag//2+1)'
    X_fft_power = X_fft_abs**2 / (N**2)
    'FFT功率：(batch_size, node_num, lag//2+1)'
    freq_resolution = exp_frequency / N
    '频率分辨率：(1)'

    freqs = torch.fft.rfftfreq(n=N, d=1/exp_frequency, requires_grad=True)
    '频率点：(lag//2+1)'

    F_12 = torch.sqrt(torch.mean(X_fft_abs, dim=2))
    'F_12：频谱均值(batch_size, node_num)'
    F_13 = torch.sqrt(torch.mean((X_fft_abs - F_12.unsqueeze(2))**2, dim=2))
    'F_13：频谱均方根值(batch_size, node_num)'
    F_14 = torch.mean((X_fft_abs - F_12.unsqueeze(2))**3, dim=2) / (F_13**3)
    'F_14：功率谱偏度(batch_size, node_num)'
    F_15 = torch.mean((X_fft_abs - F_12.unsqueeze(2))**4, dim=2) / (F_13**4)
    'F_15：功率谱峰度(batch_size, node_num)'
    F_16 = torch.mean(freqs.unsqueeze(0).unsqueeze(0) * X_fft_abs, dim=2) / torch.mean(X_fft_abs, dim=2)
    'F_16：频率重心(batch_size, node_num)'
    F_17 = torch.sqrt(torch.mean((freqs.unsqueeze(0).unsqueeze(0) - F_16.unsqueeze(2))**2 * X_fft_abs, dim=2))
    'F_17：频率根方差(batch_size, node_num)'
    F_18 = torch.sqrt(torch.mean((freqs**2 * X_fft_abs), dim=2) / torch.mean(X_fft_abs, dim=2))
    'F_18：频率均方根(batch_size, node_num)'
    F_19 = torch.sqrt(torch.mean((freqs**4 * X_fft_abs), dim=2) / torch.mean((freqs**2 * X_fft_abs), dim=2))
    'F_19：平均频率(batch_size, node_num)'
    F_20 = torch.mean((freqs**2 * X_fft_abs), dim=2) / torch.sqrt(
        torch.mean(X_fft_abs, dim=2) * torch.mean((freqs**4 * X_fft_abs), dim=2)
    )
    'F_20：频率稳定系数(batch_size, node_num)'
    F_21 = F_17 / F_16
    'F_21：变异系数(batch_size, node_num)'
    F_22 = torch.mean((freqs.unsqueeze(0).unsqueeze(0) - F_16.unsqueeze(2))**3 * X_fft_abs, dim=2) / (F_17**3)
    'F_22：频率偏度(batch_size, node_num)'
    F_23 = torch.mean((freqs.unsqueeze(0).unsqueeze(0) - F_16.unsqueeze(2))**4 * X_fft_abs, dim=2) / (F_17**4)
    'F_23：频率峰度(batch_size, node_num)'
    F_24 = torch.mean((freqs.unsqueeze(0).unsqueeze(0) - F_16.unsqueeze(2))**0.5 * X_fft_abs, dim=2) / (F_17**0.5)
    'F_24：频率标准差(batch_size, node_num)'

    # # 所有数字保留三位小数
    # F_12 = torch.round(F_12 * 1000) / 1000
    # F_13 = torch.round(F_13 * 1000) / 1000
    # F_14 = torch.round(F_14 * 1000) / 1000
    # F_15 = torch.round(F_15 * 1000) / 1000
    # F_16 = torch.round(F_16 * 1000) / 1000
    # F_17 = torch.round(F_17 * 1000) / 1000
    # F_18 = torch.round(F_18 * 1000) / 1000
    # F_19 = torch.round(F_19 * 1000) / 1000
    # F_20 = torch.round(F_20 * 1000) / 1000
    # F_21 = torch.round(F_21 * 1000) / 1000
    # F_22 = torch.round(F_22 * 1000) / 1000
    # F_23 = torch.round(F_23 * 1000) / 1000
    # F_24 = torch.round(F_24 * 1000) / 1000

    if if_2D:
        F_12 = F_12.squeeze(0)
        '(batch_size, node_num) -> (node_num)'
        F_13 = F_13.squeeze(0)
        F_14 = F_14.squeeze(0)
        F_15 = F_15.squeeze(0)
        F_16 = F_16.squeeze(0)
        F_17 = F_17.squeeze(0)
        F_18 = F_18.squeeze(0)
        F_19 = F_19.squeeze(0)
        F_20 = F_20.squeeze(0)
        F_21 = F_21.squeeze(0)
        F_22 = F_22.squeeze(0)
        F_23 = F_23.squeeze(0)
        F_24 = F_24.squeeze(0)

    # 创建成字典
    feature_dict = {# '频域均值': F_12.detach().numpy(),
                    '频域均方根值': F_13.detach().numpy(),
                    # '频域偏度': F_14.detach().numpy(),
                    # '频域峰度': F_15.detach().numpy(),
                    '频率重心': F_16.detach().numpy(),
                    '频率根方差': F_17.detach().numpy(),
                    # '频率均方根': F_18.detach().numpy(),
                    '平均频率': F_19.detach().numpy(),
                    '频率稳定系数': F_20.detach().numpy(),
                    # '变异系数': F_21.detach().numpy(),
                    # '频率偏度': F_22.detach().numpy(),
                    # '频率峰度': F_23.detach().numpy(),
                    # '频率标准差': F_24.detach().numpy()
                    }

    # return F_12, F_13, F_14, F_15, F_16, F_17, F_18, F_19, F_20, F_21, F_22, F_23, F_24

    return feature_dict







