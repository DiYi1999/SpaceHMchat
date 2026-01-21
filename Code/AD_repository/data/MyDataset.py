import os
import pickle
import numpy as np
import pandas as pd
import random
import torch
from scienceplots import new_data_path
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
# import sys
# from pathlib import Path
# project_root = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(project_root))
from AD_repository.utils.process import *
from AD_repository.utils.decompose import *
from AD_repository.utils.decompose import *

warnings.filterwarnings('ignore')








class XJTU_SPS_for_AD_Dataset(Dataset):
    def __init__(self, args, root_path='/data/DiYi/DATA/Our_Exp_Data', flag='train', lag=None,
                 features='M', data_path='XJTU-SPS Dataset/XJTU-SPS for AD', data_name='XJTU-SPS for AD',
                 missing_rate=0, missvalue=np.nan, target='OT', scale=False, scaler=None, timestamp_scaler=None, work_condition_scaler=None):
        # size [seq_len, label_len pred_len]
        # info
        self.args = args
        self.lag = lag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.scaler = scaler
        self.timestamp_scaler = timestamp_scaler
        # self.work_condition_scaler = work_condition_scaler
        # self.label_scaler = label_scaler

        self.root_path = root_path
        self.data_path = data_path
        self.data_name = data_name

        self.__read_data__()

    def get_data_dim(self, dataset):
        return self.args.sensor_num

    def __read_data__(self):
        """
        get data from pkl files

        return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
        """

        """导入数据"""
        if self.flag == 'test':
            self.data_name = self.data_name + '_Test'
            try:
                label_df = pd.read_csv(
                    os.path.join(self.root_path, self.data_path, '{}_AnomalyLabel.csv'.format(self.data_name)),
                    sep=',', index_col=False)
            except:
                label_df = pd.read_csv(
                    os.path.join('/', self.root_path, self.data_path, '{}_AnomalyLabel.csv'.format(self.data_name)),
                    sep=',', index_col=False)
            timestamp_label = label_df['Time'].values
            label = label_df.drop(['Time'], axis=1).values
            all_label = None
        else:
            self.data_name = self.data_name + '_Train'
            timestamp_label = None
            label = None
            all_label = None
        try:
            ### /data/DiYi/DATA/Our_Exp_Data/XJTU-SPS Dataset/XJTU-SPS for AD/XJTU-SPS for AD_Test_AnomalyLabel.csv
            # f = open(os.path.join(self.root_path, self.data_path, '{}.pkl'.format(self.data_name)), "rb")
            # data = pickle.load(f).values.reshape((-1, x_dim))
            # f.close()
            # 首先读取os.path.join(self.root_path, self.data_path, '{}.csv'.format(self.data_name)
            # 若失败，将self.data_path的第一个斜杠及其前内容删掉再试，不需要报错
            try:
                data_df = pd.read_csv(os.path.join(self.root_path, self.data_path, '{}.csv'.format(self.data_name)),
                                    sep=',', index_col=False)
            except:
                try:
                    data_df = pd.read_csv(os.path.join('/', self.root_path, self.data_path, '{}.csv'.format(self.data_name)),
                                        sep=',', index_col=False)
                except:
                    new_data_path = self.data_path[self.data_path.find('/')+1:]
                    data_df = pd.read_csv(os.path.join(self.root_path, new_data_path, '{}.csv'.format(self.data_name)),
                                        sep=',', index_col=False)
            if self.features == 'S':
                data = data_df[[self.target]].values
            else:
                data = data_df.drop(['Time'], axis=1).values

            # # 读取Work_Condition数据
            # Work_Condition_data_name = self.data_name + '_Work_Condition'
            # try:
            #     work_condition_df = pd.read_csv(os.path.join(self.root_path, self.data_path, '{}.csv'.format(Work_Condition_data_name)),
            #                                     sep=',', index_col=False)
            # except:
            #     new_data_path = self.data_path[self.data_path.find('/') + 1:]
            #     work_condition_df = pd.read_csv(os.path.join(self.root_path, new_data_path, '{}.csv'.format(Work_Condition_data_name)),
            #                                     sep=',', index_col=False)
            # work_condition = work_condition_df.drop(['Time'], axis=1).values

        except (KeyError, FileNotFoundError):
            raise FileNotFoundError(f'Data file not found, please check the path and file name. the code: os.path.join(self.root_path, self.data_path, data_name.csv got ({self.root_path}, {self.data_path}, {self.data_name}).')
            data = None

        x_dim = self.get_data_dim(self.data_name)
        if self.features != 'S':
            if data.shape[1] != x_dim:
                raise ValueError('Data shape error, please check it')
        if label is not None and len(data) != len(label):
            raise ValueError('Data and work_condition shape error, please check it')

        df_stamp = data_df[['Time']]
        df_stamp['Time'] = pd.to_datetime(df_stamp.Time)

        df_stamp['day'] = df_stamp.Time.apply(lambda row: row.day, 1)
        df_stamp['hour'] = df_stamp.Time.apply(lambda row: row.hour, 1)
        df_stamp['minute'] = df_stamp.Time.apply(lambda row: row.minute, 1)
        df_stamp['second'] = df_stamp.Time.apply(lambda row: row.second, 1)
        data_stamp = df_stamp.drop(['Time'], axis=1).values
        data_stamp = np.concatenate([data_stamp, np.arange(len(data_stamp)).reshape(-1, 1)], axis=1)
        # timestamp_label = label_df['Time'].values得到的是时间戳的字符串形式，而Dataloader能接受并进行拼接处理的只有部分格式，咱这个会被识别为timestamp_label.dtype  # 可能是 dtype('O') （object 类型），会报错TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found object，因此需要先转为数值，等以后用到时再使用timestamp_label = pd.to_datetime(timestamp_label, unit='s')变回时间戳
        if timestamp_label is not None:
            # timestamp_label = pd.Series(pd.to_datetime(timestamp_label)).dt.isoformat().values  # '2024-01-01T12:00:00'
            # timestamp_label = pd.to_datetime(timestamp_label).strftime('%Y-%m-%d %H:%M:%S').values
            # timestamp_label = pd.to_datetime(timestamp_label).astype(np.int64)
            timestamp_label = pd.to_datetime(timestamp_label).astype(np.int64) // 10**9
            timestamp_label = np.array(timestamp_label, dtype=np.int64)
            # 必须用np.int64，不许转成 np.float64，因为float32只有约7位十进制数字的精度，Unix时间戳(秒)现在已经是10位数字(>1600000000)，对于较大的时间戳值，float32无法精确表示
            # must be np.int64, not np.float64, because float32 only has about 7 decimal digits of precision, and Unix timestamps (in seconds) are now 10 digits (>1600000000), for large timestamp values, float32 cannot represent them accurately.
            # 这一步也是为了避免报错TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'pandas.core.indexes.base.Index'>
            # this step is also to avoid the error TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'pandas.core.indexes.base.Index'>


        """***数据划分***"""
        # 如果是测试集，直接全用，如果是训练和验证集，划分成训练集和验证集
        train_len = len(data)
        border1s = [0, train_len - (train_len // self.args.dataset_tra_d_val), 0]
        border2s = [train_len - (train_len // self.args.dataset_tra_d_val), train_len, len(data)]
        # train_len = int(self.args.dataset_split_ratio * len(data))
        # border1s = [0, train_len-(train_len//self.args.dataset_tra_d_val), train_len]
        # border2s = [train_len-(train_len//self.args.dataset_tra_d_val), train_len, len(data)]
        # train_len = int(self.args.dataset_split_ratio * len(data))
        # border1s = [0, train_len-(train_len//8), train_len]
        # border2s = [train_len-(train_len//8), train_len, len(data)]
        # train_len = int(len(data) * 0.75)
        # border1s = [0, int(train_len/3), train_len]
        # border2s = [train_len, 2*int(train_len/3), len(data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        # cutting data
        data = data[border1:border2]
        data_stamp = data_stamp[border1:border2]
        # work_condition = work_condition[border1:border2]

        """数据量减少实验：不使用全部数据，只能使用截取部分数据，根据self.args.only_use_data_ratio,但只在训练阶段"""
        """experiment of using less data: not using all data, only using a portion of data according to self.args.only_use_data_ratio, but only in training phase"""
        # 采样频率是self.args.exp_frequency，截取的时候，由于仿真初始值的设置，必须截取95min的倍数，也就是数据点数目为95*60*self.args.exp_frequency的整数倍
        # The sampling frequency is self.args.exp_frequency, when cutting, due to the setting of the initial value of the simulation, it must be a multiple of 95min, that is, the number of data points is an integer multiple of 95*60*self.args.exp_frequency
        if self.args.only_use_data_ratio < 1 and self.flag in ['train', 'val']:
            lim_len = int(len(data) * self.args.only_use_data_ratio)
            data = data[-lim_len:]
            data_stamp = data_stamp[-lim_len:]
            # work_condition = work_condition[-lim_len:]

        """***preprocessing***"""

        """normalize the data"""
        if self.scale:
            data_norm, data, scale_list, mean_list = self.normalize(data, self.flag, self.scaler)
            data_stamp_norm, _, _, _ = self.normalize(data_stamp, self.flag, self.timestamp_scaler)
            # work_condition_norm, work_condition, scale_list_work_condition, mean_list_work_condition = self.normalize(
            #     work_condition, self.flag, self.work_condition_scaler)
        else:
            data_norm = data
            scale_list = [1.0] * data.shape[1]
            mean_list = [0.0] * data.shape[1]
            data_stamp_norm = data_stamp
            # work_condition_norm = work_condition
            # scale_list_work_condition = [1.0] * work_condition.shape[1]
            # mean_list_work_condition = [0.0] * work_condition.shape[1]

        """在进行数据缺失或者加噪声之前，保留原始数据用于评估MSE，且创建dirty数据用于画图展示"""
        """before adding noise or missing data, keep the original data for evaluating MSE, and create dirty data for plotting"""
        orig_data = data.copy()
        orig_data_norm = data_norm.copy()
        dirty_data = data.copy()
        dirty_data_norm = data_norm.copy()

        """***鲁棒性测试dirty_data创建"""
        """ create dirty_data for robustness test"""
        """加入噪声"""
        """Add noise"""
        # 注意这里用的信噪比不是比例，而是dB，因为领域内常用的是dB，设置时要注意  https://blog.csdn.net/qq_58860480/article/details/140583800
        # Note that the signal-to-noise ratio used here is not a ratio, but dB, because dB is commonly used in the field, so be careful when setting it  https://blog.csdn.net/qq_58860480/article/details/140583800
        if self.args.add_noise_SNR > 0:
            signal_power = np.mean(dirty_data ** 2)
            noise_power = signal_power / (10 ** (self.args.add_noise_SNR / 10))
            noise = np.random.normal(0, np.sqrt(noise_power), dirty_data.shape)
            dirty_data = dirty_data + noise
            signal_power = np.mean(dirty_data_norm ** 2)
            noise_power = signal_power / (10 ** (self.args.add_noise_SNR / 10))
            noise = np.random.normal(0, np.sqrt(noise_power), dirty_data_norm.shape)
            dirty_data_norm = dirty_data_norm + noise
            data, data_norm = dirty_data, dirty_data_norm  # 同步到data和data_norm
        """加入野点"""
        """Add outliers"""
        if self.args.add_outliers:
            outliers_amp = (orig_data.max(axis=0) - orig_data.min(axis=0))
            outliers = np.random.normal(1.4 * outliers_amp, 2.0 * outliers_amp, dirty_data.shape) \
                       * (np.random.choice([-1, 1], size=dirty_data.shape, p=[0.5, 0.5]))
            outliers = outliers * (np.random.rand(*dirty_data.shape) < self.args.outlier_rate)
            dirty_data = dirty_data + outliers
            outliers_amp = (orig_data_norm.max(axis=0) - orig_data_norm.min(axis=0))
            outliers = np.random.normal(1.4 * outliers_amp, 2.0 * outliers_amp, dirty_data_norm.shape) \
                       * (np.random.choice([-1, 1], size=dirty_data_norm.shape, p=[0.5, 0.5]))
            outliers = outliers * (np.random.rand(*dirty_data_norm.shape) < self.args.outlier_rate)
            dirty_data_norm = dirty_data_norm + outliers
            data, data_norm = dirty_data, dirty_data_norm  # 同步到data和data_norm
        """进行数据缺失"""
        """Add missing data"""
        if self.args.missing_rate > 0:
            dirty_data, dirty_data_norm = make_missing_data(dirty_data, self.args.missing_rate,
                                                            self.args.missvalue, dirty_data_norm)
            data, data_norm = dirty_data, dirty_data_norm  # 同步到data和data_norm

        """***初步预处理"""
        """preprocessing of dirty_data"""
        """nan填充:用前一个或者后一个时间步进行nan填充"""
        """Fill nan: use the previous or next time step to fill nan"""
        if np.isnan(data).any():
            data = nan_filling(data)
            data_norm = nan_filling(data_norm)
            dirty_data, dirty_data_norm = data, data_norm
            # 讲道理这里dirty_data就是dirty的，不应该补全，但是如果里面nan太多，画图展示估计会报错或者打断plot线，补一下吧，方便画图
        """含噪数据滑动平均预处理"""
        """Smoothing noisy data with moving average"""
        # 先去野点、异常点
        # First remove outliers and abnormal points
        if self.args.remove_outliers:
            data = remove_outliers(data, factor=1.0)
            data_norm = remove_outliers(data_norm, factor=1.0)
        # 再滑动平均
        # Then moving average
        if self.args.preMA:
            data = preMA(data, self.args.preMA_win)
            data_norm = preMA(data_norm, self.args.preMA_win)

        """数据定型"""
        """ shaping data """
        self.data = data
        self.data_norm = data_norm
        self.orig_data = orig_data
        self.orig_data_norm = orig_data_norm
        self.dirty_data = dirty_data
        self.dirty_data_norm = dirty_data_norm
        self.data_stamp = data_stamp_norm
        self.label = label
        self.all_label = all_label
        self.timestamp_label = timestamp_label
        # self.work_condition = work_condition
        # self.work_condition_norm = work_condition_norm
        self.scaler_info = {'scale_list': np.array(scale_list).astype(np.float32),
                            'mean_list': np.array(mean_list).astype(np.float32)}
                            # 'scale_list_work_condition': np.array(scale_list_work_condition).astype(np.float32),
                            # 'mean_list_work_condition': np.array(mean_list_work_condition).astype(np.float32)}

    def __getitem__(self, index):
        ##### x
        s_begin = index * self.args.lag_step
        s_end = s_begin + self.lag
        x_batch = self.data[s_begin:s_end]
        x_batch_norm = self.data_norm[s_begin:s_end]

        ##### y
        if self.args.BaseOn == 'reconstruct':
            r_begin = s_begin
            r_end = s_end
        elif self.args.BaseOn == 'forecast':
            r_begin = s_end - self.args.label_len
            r_end = s_end + self.args.pred_len
        else:
            raise ValueError('BaseOn must be "reconstruct" or "forecast"')
        y_batch = self.data[r_begin:r_end]
        y_batch_norm = self.data_norm[r_begin:r_end]
        "# y必须使用数据污染后经过初步处理的data数据而不能使用orig_data，因为使用orig_data来计算loss进行优化是信息泄露"
        "# y must use the data after contamination and preliminary processing rather than orig_data, because using orig_data to calculate loss for optimization is information leakage"
        y_orig_batch = self.orig_data[r_begin:r_end]
        y_orig_batch_norm = self.orig_data_norm[r_begin:r_end]
        "# 这个orig_data是用来计算MSE，评估去噪等数据恢复性能的，注意不能参与loss计算等训练过程，避免信息泄露"
        "# this orig_data is used to calculate MSE, evaluate denoising and other data recovery performance, note that it cannot participate in loss calculation and other training processes to avoid information leakage"
        y_dirty_batch = self.dirty_data[r_begin:r_end]
        y_dirty_batch_norm = self.dirty_data_norm[r_begin:r_end]
        "# 这个dirty_data是用来画图的时候作为实际值展示噪声添加程度的，没经过preMA和去野点的纯dirty数据"
        "# this dirty_data is used for plotting as the actual value to show the degree of noise added, the pure dirty data without preMA and outlier removal"

        # ##### work condition
        # WC_batch = self.work_condition[r_end - self.lag:r_end]
        # WC_batch_norm = self.work_condition_norm[r_end - self.lag:r_end]
        # "work condition use r_begin:r_end, not s_begin:s_end, taht is not information leakage," \
        # "because it is the available known condition, and the physical model also uses this, " \
        # "not belonging to future data. "

        ##### label
        # if self.label is not None:
        #     label = self.label[r_begin:r_end]
        #     # label里面只有有1，那么label就是1，没1就是0
        #     label = 1 if np.any(label==1) else 0
        # else:
        #     label = None
        label = self.label
        all_label = self.all_label
        timestamp_label = self.timestamp_label

        ##### t
        # datetime_batch = self.data_stamp[s_begin:s_end]
        # datetime_batch = self.data_stamp[r_end - self.lag:r_end]
        x_datetime_batch = self.data_stamp[s_begin:s_end]
        y_datetime_batch = self.data_stamp[r_begin:r_end]

        return (x_batch.astype(np.float32),
                x_batch_norm.astype(np.float32),
                # WC_batch.astype(np.float32),
                # WC_batch_norm.astype(np.float32),
                y_batch.astype(np.float32),
                y_batch_norm.astype(np.float32),
                y_orig_batch.astype(np.float32),
                y_orig_batch_norm.astype(np.float32),
                y_dirty_batch.astype(np.float32),
                y_dirty_batch_norm.astype(np.float32),
                # label.astype(np.float32) if self.label is not None else np.array([-1], dtype=np.float32), # lighting不然返回None，没办法只能返回字符串None
                # all_label.astype(np.float32) if self.all_label is not None else np.array([-1], dtype=np.float32),
                # timestamp_label if self.timestamp_label is not None else np.array([-1], dtype=np.float32),
                label.astype(np.float32) if self.label is not None else 'None', # lighting不然返回None，没办法只能返回字符串None
                all_label.astype(np.float32) if self.all_label is not None else 'None',
                timestamp_label if self.timestamp_label is not None else 'None',
                # datetime_batch.astype(np.float32),
                x_datetime_batch.astype(np.float32),
                y_datetime_batch.astype(np.float32),
                self.scaler_info)

    def __len__(self):
        if self.args.BaseOn == 'reconstruct':
            return (len(self.data) - self.lag) // self.args.lag_step + 1
        elif self.args.BaseOn == 'forecast':
            return (len(self.data) - self.lag - self.args.pred_len) // self.args.lag_step + 1

    def normalize(self, data, flag, scaler):
        """
        returns normalized and standardized data.

        data : array-like
        flag: 'train' or 'val' or 'test'
        scaler: StandardScaler or MinMaxScaler
        """
        if flag == 'train':
            scaler.fit(data)
        elif flag in ['val', 'test']:
            # 验证时，先判断self.scaler是否已经fit过，没有的话就fit，有就不用了
            # During validation, first check whether self.scaler has been fitted, if not, fit it, if yes, do not fit it again
            if not hasattr(scaler, 'scale_'):
                scaler.fit(data)
        else:
            pass
        data_norm = scaler.transform(data)
        scale_list = scaler.scale_.tolist()
        mean_list = scaler.mean_.tolist() if hasattr(scaler, 'mean_') else scaler.min_.tolist()

        # df = np.asarray(df, dtype=np.float32)
        # if len(df.shape) == 1:
        #     raise ValueError('Data must be a 2-D array')
        # if np.any(np.isnan(df).sum() != 0):
        #     print('Data contains null values. Will be replaced with 0')
        #     df = np.nan_to_num(df)
        # # 按列对df进行minmax归一化
        # df = (df - df.min(axis=0)) / (df.max(axis=0) - df.min(axis=0))
        # print('Data normalized')
        # # 有的列值全相同，除了0，这样归一化后全是nan，所以要把nan替换成0
        # df = np.nan_to_num(df)

        return data_norm, data, scale_list, mean_list

    def my_inverse_transform(self, data, scaler_str=None):
        """调用此函数时一定注意此函数会打断梯度, Note that this function will break the gradient when called

        data: tensor or numpy, shape: (batch_size, node_num, len)
        scaler_str: str, 调用哪个归一化器 which normalizer: 'data' or 'timestamp' or 'work_condition'
        """
        # 如果data是tensor，先转为numpy，但是记得最后要转回tensor输出
        if type(data) == torch.Tensor:
            output = data.numpy()
        else:
            output = data
        # 如果是(node_num, len),则转为(len, node_num),因为scaler.inverse_transform要求这样,但是最后要转回来
        if data.shape[0] < data.shape[1]:
            output = output.T
        # 开始逆归一化/标准化
        if scaler_str == 'data' or scaler_str == None:
            output = self.scaler.inverse_transform(output)
        elif scaler_str == 'timestamp':
            output = self.timestamp_scaler.inverse_transform(output)
        # elif scaler_str == 'work_condition':
        #     output = self.work_condition_scaler.inverse_transform(output)
        else:
            raise ValueError('scaler_str must be "data" or "timestamp" or "work_condition"')
        # 转回data原来的形状
        if data.shape[0] < data.shape[1]:
            output = output.T
        # 转回tensor
        if type(data) == torch.Tensor:
            output = torch.from_numpy(output).float()
        return output











