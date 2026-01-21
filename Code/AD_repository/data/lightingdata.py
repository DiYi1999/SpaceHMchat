def import_lightning():
    try:
        import lightning.pytorch as pl
    except ModuleNotFoundError:
        import pytorch_lightning as pl
    return pl
pl = import_lightning()

from AD_repository.data.MyDataset import *
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import time
import warnings
warnings.filterwarnings('ignore')


class MyLigDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super(MyLigDataModule, self).__init__()
        self.data_name = args.data_name
        self.shuffle_flag = True
        # self.shuffle_flag = False
        # # 样本顺序先别打乱，因为有些物理模型/PINN方法的电池电量是要安时累加的，使用者可以根据自己的需求更改
        # # The sample order is not shuffled for now, because some physical models/PINN methods require the battery capacity to be accumulated in ampere-hours. Users can modify it according to their needs.
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.drop_last = False
        # 如果改成True，MyModel里面Y的相关代码也得该切片了
        # If changed to True, the relevant code of Y in MyModel must also be modified.
        self.args = args
        self.ready_dataset_module()
        self.data_set = None

        self.scaler = StandardScaler()
        # self.scaler = MinMaxScaler()
        self.timestamp_scaler = MinMaxScaler()
        self.work_condition_scaler = StandardScaler()

    def ready_dataset_module(self):
        if self.args.Dataset in globals():
            self.DataSet = globals()[self.args.Dataset]
        else:
            self.args.Dataset = self.args.Dataset.replace(' ', '_')
            self.args.Dataset = self.args.Dataset.replace('-', '_')
            if self.args.Dataset in globals():
                self.DataSet = globals()[self.args.Dataset]
            else:
                str = self.args.Dataset
                index = len('_Dataset')
                str_new = str[: -index-1] + str[-index:]
                self.DataSet = globals()[str_new]

    def train_dataloader(self):
        args = self.args
        self.data_set = self.DataSet(
            args,
            root_path=args.root_path,
            data_path=args.data_path,
            data_name=args.data_name,
            missing_rate=args.missing_rate,
            missvalue=args.missvalue,
            flag='train',
            lag=args.lag,
            features=args.features,
            target=args.target,
            scale=args.scale,
            scaler=self.scaler,
            timestamp_scaler=self.timestamp_scaler,
            work_condition_scaler=self.work_condition_scaler
        )
        print('train', len(self.data_set))
        data_loader = DataLoader(
            self.data_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle_flag,
            num_workers=self.num_workers,
            drop_last=self.drop_last)
        return data_loader

    def val_dataloader(self):
        args = self.args
        self.data_set = self.DataSet(
            args,
            root_path=args.root_path,
            data_path=args.data_path,
            data_name=args.data_name,
            missing_rate=args.missing_rate,
            missvalue=args.missvalue,
            flag='val',
            lag=args.lag,
            features=args.features,
            target=args.target,
            scale=args.scale,
            scaler=self.scaler,
            timestamp_scaler=self.timestamp_scaler,
            work_condition_scaler=self.work_condition_scaler
        )
        print('val', len(self.data_set))
        data_loader = DataLoader(
            self.data_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle_flag,
            num_workers=self.num_workers,
            drop_last=self.drop_last)
        return data_loader

    def test_dataloader(self):
        self.shuffle_flag = False
        # 测试集样本顺序别打乱，因为还要将输出样本拼起来用来画图
        # The test set sample order should not be shuffled, because the output samples need to be spliced together for plotting.
        args = self.args
        self.data_set = self.DataSet(
            args,
            root_path=args.root_path,
            data_path=args.data_path,
            data_name=args.data_name,
            missing_rate=args.missing_rate,
            missvalue=args.missvalue,
            flag='test',
            lag=args.lag,
            features=args.features,
            target=args.target,
            scale=args.scale,
            scaler=self.scaler,
            timestamp_scaler=self.timestamp_scaler,
            work_condition_scaler=self.work_condition_scaler
        )
        print('test', len(self.data_set))
        data_loader = DataLoader(
            self.data_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle_flag,
            num_workers=self.num_workers,
            drop_last=self.drop_last)
        return data_loader

    def predict_dataloader(self):
        data_loader = self.test_dataloader()
        return data_loader
