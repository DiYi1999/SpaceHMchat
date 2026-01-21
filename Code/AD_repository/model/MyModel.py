from typing import Any
import time

import yaml
from AD_repository.utils.post_processing import *

def import_lightning():
    try:
        import lightning.pytorch as pl
    except ModuleNotFoundError:
        import pytorch_lightning as pl
    return pl
pl = import_lightning()

from AD_repository.utils.performance import *
from AD_repository.utils.process import *
from AD_repository.utils.plot import *
from AD_repository.graph.graph_calculate import graph_calculate_from_prior, Graph_calculate
from AD_repository.utils.MyOptimizers import *
from AD_repository.utils.decompose import *
from AD_repository.model.ours.SPS_Model import SPS_Model
from AD_repository.model.ours.SPS_Model_Phy import SPS_Model_Phy



class MyLigModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.training_loss = 0
        self.val_loss = []
        self.val_loss_end = 0
        self.train_epoch_end = 0
        self.test_time = 0

        # 计算邻接矩阵和多项式拟合列表
        if args.model_selection in ['SPS_Model_PINN', 'SPS_Model_NN'] and args.graph_ca_meth == 'Prior':
            A_eye0, A_eye1 = graph_calculate_from_prior(args)
            self.A = A_eye1 if args.self_edge == True else A_eye0
        elif args.model_selection in ['SPS_Model_PINN', 'SPS_Model_NN'] and args.graph_ca_meth in ['MIC', 'Cosine', 'Copent']:
            A_eye0, A_eye1, A_w = Graph_calculate(args, if_return_norm=False)
            self.A = A_eye1 if args.self_edge == True else A_eye0
        else:
            self.A = None
        # self.A = torch.Tensor(self.A).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) \
        #     if self.A is not None and not isinstance(self.A, torch.Tensor) else self.A

        self.SPS_Model = SPS_Model(self.args)
        # self.SPS_Model = torch.compile(SPS_Model(self.args))

        self.loss = torch.nn.MSELoss()

        self.trues = []
        'trues: 原始数据(但是不裁剪掉重复值) F(batch_num * batch_size, node_num, pred_len)'
        self.preds = []
        'preds: 重建/预测数据(但是不裁剪掉重复值) F(batch_num * batch_size, node_num, pred_len)'

        self.Y_orig = torch.Tensor()
        'Y_orig: 原始数据 R(node_num, data_len) or F(node_num, data_len - lag)'
        self.Y_fore = torch.Tensor()
        'Y_fore: 重建/预测数据 R(node_num, data_len) or F(node_num, data_len - lag)'
        self.label = torch.Tensor()
        'label: 异常标签 R(data_len) or F(data_len - lag)'
        self.all_label = torch.Tensor()
        'all_label: 异常标签 R(node_num, data_len) or F(node_num, data_len - lag)'
        self.timestamp_label = torch.Tensor()
        'timestamp_label: 时间戳标签 R(data_len) or F(data_len - lag)'
        self.S = torch.Tensor()
        'S: 异常分数矩阵 R(node_num, data_len) or F(node_num, data_len - lag)'
        self.detect_01 = torch.Tensor()
        'detect_01: 异常检测结果01矩阵 R(node_num, data_len) or F(node_num, data_len - lag)'
        self.scaler_info = None
        "'scaler_info: dict, {'scale_list': scale_list, 'mean_list': mean_list, 'scale_list_work_condition': scale_list_work_condition, 'mean_list_work_condition': mean_list_work_condition}'"
        self.exam_result = {}
        '储存performance各指标'

        # self.automatic_optimization = True
        self.save_hyperparameters()
        self.configure_optimizers()

    def configure_optimizers(self):

        return MyOptimizers(self)

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        # Implement your own custom logic to clip gradients
        # You can call `self.clip_gradients` with your settings:
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm
        )

    def training_step(self, batch, batch_idx):
        """如果是在SPS_Model_PINN时，可以直接使用物理信息的仿真结果"""
        if self.args.model_selection == 'SPS_Model_PINN':
            # X, X_norm, WC, WC_norm, Y, Y_norm, Y_primitive, Y_primitive_norm, Y_dirty, Y_dirty_norm, label, all_label, timestamp_label, T, T_of_y, scaler_info, PI_info_batch = batch
            X, X_norm, Y, Y_norm, Y_primitive, Y_primitive_norm, Y_dirty, Y_dirty_norm, label, all_label, timestamp_label, T, T_of_y, scaler_info, PI_info_batch = batch
            PI_info_batch = PI_info_batch.permute(0, 2, 1)
            'PI_info_batch: (batch_size, node_num, lag)   not be normalized'
        else:
            # X, X_norm, WC, WC_norm, Y, Y_norm, Y_primitive, Y_primitive_norm, Y_dirty, Y_dirty_norm, label, all_label, timestamp_label, T, T_of_y, scaler_info = batch
            X, X_norm, Y, Y_norm, Y_primitive, Y_primitive_norm, Y_dirty, Y_dirty_norm, label, all_label, timestamp_label, T, T_of_y, scaler_info = batch
            PI_info_batch = None
        X = X.permute(0, 2, 1)
        'X: (batch_size, node_num, lag)   not be normalized'
        X_norm = X_norm.permute(0, 2, 1)
        'X_norm: (batch_size, node_num, lag)   normalized'
        if self.args.if_add_work_condition:
            raise Exception("anomaly detection dataset do not return WC data, please check the dataset!")
            WC = WC.permute(0, 2, 1)
            'WC:工况 (batch_size, 4, lag)   not be normalized'
            WC_norm = WC_norm.permute(0, 2, 1)
            'WC_norm:工况 (batch_size, 4, lag)   normalized'
        else:
            WC = torch.zeros_like(X_norm[:, :4, :])
            'WC:工况 (batch_size, 4, lag)   not be normalized'
            WC_norm = torch.zeros_like(X_norm[:, :4, :])
            'WC_norm:工况 (batch_size, 4, lag)   normalized'
        Y = Y.permute(0, 2, 1) if self.args.BaseOn == 'reconstruct' else Y.permute(0, 2, 1)[:, :, -self.args.pred_len:]
        'Y: (batch_size, node_num, lag/pred_len)   not be normalized'
        Y_norm = Y_norm.permute(0, 2, 1) if self.args.BaseOn == 'reconstruct' else Y_norm.permute(0, 2, 1)[:, :, -self.args.pred_len:]
        'Y_norm: (batch_size, node_num, lag/pred_len)   normalized'
        Y_primitive = Y_primitive.permute(0, 2, 1) if self.args.BaseOn == 'reconstruct' else Y_primitive.permute(0, 2, 1)[:, :, -self.args.pred_len:]
        'Y_primitive: (batch_size, node_num, lag/pred_len)   not be normalized  Y没缺失没加噪声前的原始数据，仅用来计算MSE，为避免信息泄露，训练阶段不被允许使用'
        Y_primitive_norm = Y_primitive_norm.permute(0, 2, 1) if self.args.BaseOn == 'reconstruct' else Y_primitive_norm.permute(0, 2, 1)[:, :, -self.args.pred_len:]
        'Y_primitive_norm: (batch_size, node_num, lag/pred_len)   normalized  Y没缺失没加噪声前的原始数据，仅用来计算MSE，为避免信息泄露，训练阶段不被允许使用'
        T = T.permute(0, 2, 1)
        'T: (batch_size, 4, lag)'
        T_of_y = T_of_y.permute(0, 2, 1)
        'T_of_y: (batch_size, 4, label_len+pred_len / lag)'
        # init_sample = init_sample.permute(0, 2, 1)
        # 'init_sample: (batch_size, node_num, lag)   not be normalized'
        # init_sample_norm = init_sample_norm.permute(0, 2, 1)
        # 'init_sample_norm: (batch_size, node_num, lag)   normalized'
        init_sample = None
        'init_sample: (batch_size, node_num, lag)   not be normalized'
        init_sample_norm = None
        'init_sample_norm: (batch_size, node_num, lag)   normalized'
        for key, value in scaler_info.items():
            scaler_info[key] = value[0,:]
        self.scaler_info = scaler_info
        "'scaler_info: dict, {'scale_list': scale_list, 'mean_list': mean_list, 'scale_list_work_condition': scale_list_work_condition, 'mean_list_work_condition': mean_list_work_condition}'"
        A = torch.Tensor(self.A).to(X.device) if self.A is not None else None
        'A: (node_num, node_num)'

        H_norm = self.SPS_Model(A, X, X_norm, WC, WC_norm, T, T_of_y, Y, init_sample, init_sample_norm, scaler_info, PI_info_batch)
        'H_norm: (batch_size, node_num, lag/pred_len)   normalized'

        if self.args.BaseOn == 'forecast':
            Y = Y[:, :, -self.args.pred_len:]
            'Y: R(batch_size, node_num, pred_len) or F(batch_size, node_num, pred_len)'
            Y_norm = Y_norm[:, :, -self.args.pred_len:]
            'Y_norm: R(batch_size, node_num, pred_len) or F(batch_size, node_num, pred_len)'
        if H_norm.shape != Y_norm.shape:
            raise Exception("H_norm.shape != Y_norm.shape, please check the model!")
        if self.args.channel_to_channel == 'M':
            loss = self.loss(H_norm, Y_norm)
        elif self.args.channel_to_channel == 'MS':
            loss = self.loss(H_norm[:, self.args.MS_which, :], Y_norm[:, self.args.MS_which, :])
        else:
            raise Exception("No such channel_to_channel! must be 'M' or 'MS'!")

        self.log(name='training_loss', value=loss, on_epoch=True, prog_bar=True, logger=True)
        # on_epoch=True: 每个epoch都记录
        # prog_bar=True: 进度条显示
        # logger=True: tensorboard显示
        self.training_loss = loss.item()

        # 记录epoch
        self.train_epoch_end = self.current_epoch

        return loss


    # def on_after_backward(self) -> None:
    #     """
    #     此函数主要是报错PyTorch Lightning DDP crashes with unused parameters时用于排查是哪些参数
    #     """
    #     print("on_after_backward enter")
    #     for p in self.named_parameters():
    #         if p[1].grad is None:
    #             print(p)
    #     print("on_after_backward exit")


    def validation_step(self, batch, batch_idx):
        """
        validation_step 是当调用 trainer.validate() 或者
        在训练过程中每个 epoch 结束后，如果在 Trainer 类的初始化中设置了 check_val_every_n_epoch 参数，
        那么 validation_step 就会被调用。
        """
        """如果是在SPS_Model_PINN时，可以直接使用物理信息的仿真结果"""
        if self.args.model_selection == 'SPS_Model_PINN':
            # X, X_norm, WC, WC_norm, Y, Y_norm, Y_primitive, Y_primitive_norm, Y_dirty, Y_dirty_norm, label, all_label, timestamp_label, T, T_of_y, scaler_info, PI_info_batch = batch
            X, X_norm, Y, Y_norm, Y_primitive, Y_primitive_norm, Y_dirty, Y_dirty_norm, label, all_label, timestamp_label, T, T_of_y, scaler_info, PI_info_batch = batch
            PI_info_batch = PI_info_batch.permute(0, 2, 1)
            'PI_info_batch: (batch_size, node_num, lag)   not be normalized'
        else:
            # X, X_norm, WC, WC_norm, Y, Y_norm, Y_primitive, Y_primitive_norm, Y_dirty, Y_dirty_norm, label, all_label, timestamp_label, T, T_of_y, scaler_info = batch
            X, X_norm, Y, Y_norm, Y_primitive, Y_primitive_norm, Y_dirty, Y_dirty_norm, label, all_label, timestamp_label, T, T_of_y, scaler_info = batch
            PI_info_batch = None
        X = X.permute(0, 2, 1)
        'X: (batch_size, node_num, lag)   not be normalized'
        X_norm = X_norm.permute(0, 2, 1)
        'X_norm: (batch_size, node_num, lag)   normalized'
        if self.args.if_add_work_condition:
            raise Exception("anomaly detection dataset do not return WC data, please check the dataset!")
            WC = WC.permute(0, 2, 1)
            'WC:工况 (batch_size, 4, lag)   not be normalized'
            WC_norm = WC_norm.permute(0, 2, 1)
            'WC_norm:工况 (batch_size, 4, lag)   normalized'
        else:
            WC = torch.zeros_like(X_norm[:, :4, :])
            'WC:工况 (batch_size, 4, lag)   not be normalized'
            WC_norm = torch.zeros_like(X_norm[:, :4, :])
            'WC_norm:工况 (batch_size, 4, lag)   normalized'
        Y = Y.permute(0, 2, 1) if self.args.BaseOn == 'reconstruct' else Y.permute(0, 2, 1)[:, :, -self.args.pred_len:]
        'Y: (batch_size, node_num, lag/pred_len)   not be normalized'
        Y_norm = Y_norm.permute(0, 2, 1) if self.args.BaseOn == 'reconstruct' else Y_norm.permute(0, 2, 1)[:, :, -self.args.pred_len:]
        'Y_norm: (batch_size, node_num, lag/pred_len)   normalized'
        Y_primitive = Y_primitive.permute(0, 2, 1) if self.args.BaseOn == 'reconstruct' else Y_primitive.permute(0, 2, 1)[:, :, -self.args.pred_len:]
        'Y_primitive: (batch_size, node_num, lag/pred_len)   not be normalized  Y没缺失没加噪声前的原始数据，仅用来计算MSE，为避免信息泄露，训练阶段不被允许使用'
        Y_primitive_norm = Y_primitive_norm.permute(0, 2, 1) if self.args.BaseOn == 'reconstruct' else Y_primitive_norm.permute(0, 2, 1)[:, :, -self.args.pred_len:]
        'Y_primitive_norm: (batch_size, node_num, lag/pred_len)   normalized  Y没缺失没加噪声前的原始数据，仅用来计算MSE，为避免信息泄露，训练阶段不被允许使用'
        T = T.permute(0, 2, 1)
        'T: (batch_size, 4, lag)'
        T_of_y = T_of_y.permute(0, 2, 1)
        'T_of_y: (batch_size, 4, label_len+pred_len / lag)'
        # init_sample = init_sample.permute(0, 2, 1)
        # 'init_sample: (batch_size, node_num, lag)   not be normalized'
        # init_sample_norm = init_sample_norm.permute(0, 2, 1)
        # 'init_sample_norm: (batch_size, node_num, lag)   normalized'
        init_sample = None
        'init_sample: (batch_size, node_num, lag)   not be normalized'
        init_sample_norm = None
        'init_sample_norm: (batch_size, node_num, lag)   normalized'
        for key, value in scaler_info.items():
            scaler_info[key] = value[0,:]
        self.scaler_info = scaler_info
        "'scaler_info: dict, {'scale_list': scale_list, 'mean_list': mean_list, 'scale_list_work_condition': scale_list_work_condition, 'mean_list_work_condition': mean_list_work_condition}'"
        A = torch.Tensor(self.A).to(X.device) if self.A is not None else None
        'A: (node_num, node_num)'

        H_norm = self.SPS_Model(A, X, X_norm, WC, WC_norm, T, T_of_y, Y, init_sample, init_sample_norm, scaler_info, PI_info_batch)
        'H_norm: (batch_size, node_num, lag/pred_len)   normalized'

        if self.args.BaseOn == 'forecast':
            Y = Y[:, :, -self.args.pred_len:]
            'Y: R(batch_size, node_num, pred_len) or F(batch_size, node_num, pred_len)'
            Y_norm = Y_norm[:, :, -self.args.pred_len:]
            'Y_norm: R(batch_size, node_num, pred_len) or F(batch_size, node_num, pred_len)'
        if H_norm.shape != Y_norm.shape:
            raise Exception("H_norm.shape != Y_norm.shape, please check the model!")
        if self.args.channel_to_channel == 'M':
            loss = self.loss(H_norm, Y_norm)
        elif self.args.channel_to_channel == 'MS':
            loss = self.loss(H_norm[:, self.args.MS_which, :], Y_norm[:, self.args.MS_which, :])
        else:
            raise Exception("No such channel_to_channel! must be 'M' or 'MS'!")

        # self.val_loss.append(loss.item())
        self.val_loss.append(loss)

        return {"val_loss": loss}


    def on_validation_epoch_end(self):
        self.val_loss_end = torch.stack(self.val_loss).mean()
        # self.log(name='validation_epoch_loss', value=self.val_loss_end, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(name='validation_epoch_loss', value=self.val_loss_end, on_epoch=True, prog_bar=True, logger=True)
        # on_epoch=True: 每个epoch都记录
        # prog_bar=True: 进度条显示
        # logger=True: tensorboard显示
        # sync_dist=True: 多GPU训练时，同步不同GPU的结果
        self.val_loss.clear()
        # .clear()清空列表

        return self.val_loss_end


    # def on_validation_end(self):
    #     # 将结果保存到csv文件中
    #     dirname_path = self.args.table_save_path + '/'
    #     if not os.path.exists(dirname_path + self.args.exp_name + '_val.csv'):
    #         os.makedirs(dirname_path, exist_ok=True)
    #         args_list = vars(self.args)
    #         args_dict = {k: str(v) for k, v in args_list.items()}
    #         # args_dict加上val_loss_end就是save_dict
    #         save_dict = {**args_dict, 'val_loss_end': self.val_loss_end.item()}
    #         df = pd.DataFrame(save_dict, index=[0])
    #         df.to_csv(self.args.table_save_path + '/' + self.args.exp_name + '_val.csv', index=False, mode='a',
    #                   header=True)
    #         # header=True表示保存列索引,不存在文件，第一次创建保存表头
    #     else:
    #         args_list = vars(self.args)
    #         args_dict = {k: str(v) for k, v in args_list.items()}
    #         save_dict = {**args_dict, 'val_loss_end': self.val_loss_end.item()}
    #         df = pd.DataFrame(save_dict, index=[0])
    #         df.to_csv(self.args.table_save_path + '/' + self.args.exp_name + '_val.csv', index=False, mode='a',
    #                   header=False)
    #         # index=False表示不保存行索引，已有文件，不保存表头


    def test_step(self, batch, batch_idx):
        if batch_idx == 1:
            start_time = time.perf_counter()

        if self.args.TASK == 'anomaly_detection':
            self.Y_orig, self.Y_fore = self.test_step_AD(batch, batch_idx)
        elif self.args.TASK == 'forecast':
            self.Y_orig, self.Y_fore, self.trues, self.preds = self.test_step_FC(batch, batch_idx)
        elif self.args.TASK == 'reconstruct':
            self.Y_orig, self.Y_fore = self.test_step_RE(batch, batch_idx)
        else:
            raise Exception("No such TASK! must be 'anomaly_detection' or 'forecast' or 'reconstruct'!")

        if batch_idx == 1:
            self.test_time = time.perf_counter() - start_time
            print('test_time:', self.test_time)

        return self.Y_orig, self.Y_fore


    def test_step_AD(self, batch, batch_idx):
        """如果是在SPS_Model_PINN时，可以直接使用物理信息的仿真结果"""
        if self.args.model_selection == 'SPS_Model_PINN':
            # X, X_norm, WC, WC_norm, Y, Y_norm, Y_primitive, Y_primitive_norm, Y_dirty, Y_dirty_norm, label, all_label, timestamp_label, T, T_of_y, scaler_info, PI_info_batch = batch
            X, X_norm, Y, Y_norm, Y_primitive, Y_primitive_norm, Y_dirty, Y_dirty_norm, label, all_label, timestamp_label, T, T_of_y, scaler_info, PI_info_batch = batch
            PI_info_batch = PI_info_batch.permute(0, 2, 1)
            'PI_info_batch: (batch_size, node_num, lag)   not be normalized'
        else:
            # X, X_norm, WC, WC_norm, Y, Y_norm, Y_primitive, Y_primitive_norm, Y_dirty, Y_dirty_norm, label, all_label, timestamp_label, T, T_of_y, scaler_info = batch
            X, X_norm, Y, Y_norm, Y_primitive, Y_primitive_norm, Y_dirty, Y_dirty_norm, label, all_label, timestamp_label, T, T_of_y, scaler_info = batch
            PI_info_batch = None
        X = X.permute(0, 2, 1)
        'X: (batch_size, node_num, lag)   not be normalized'
        X_norm = X_norm.permute(0, 2, 1)
        'X_norm: (batch_size, node_num, lag)   normalized'
        if self.args.if_add_work_condition:
            raise Exception("anomaly detection dataset do not return WC data, please check the dataset!")
            WC = WC.permute(0, 2, 1)
            'WC:工况 (batch_size, 4, lag)   not be normalized'
            WC_norm = WC_norm.permute(0, 2, 1)
            'WC_norm:工况 (batch_size, 4, lag)   normalized'
        else:
            WC = torch.zeros_like(X_norm[:, :4, :])
            'WC:工况 (batch_size, 4, lag)   not be normalized'
            WC_norm = torch.zeros_like(X_norm[:, :4, :])
            'WC_norm:工况 (batch_size, 4, lag)   normalized'
        Y = Y.permute(0, 2, 1) if self.args.BaseOn == 'reconstruct' else Y.permute(0, 2, 1)[:, :, -self.args.pred_len:]
        'Y: (batch_size, node_num, lag/pred_len)   not be normalized'
        Y_norm = Y_norm.permute(0, 2, 1) if self.args.BaseOn == 'reconstruct' else Y_norm.permute(0, 2, 1)[:, :, -self.args.pred_len:]
        'Y_norm: (batch_size, node_num, lag/pred_len)   normalized'
        Y_primitive = Y_primitive.permute(0, 2, 1) if self.args.BaseOn == 'reconstruct' else Y_primitive.permute(0, 2, 1)[:, :, -self.args.pred_len:]
        'Y_primitive: (batch_size, node_num, lag/pred_len)   not be normalized  Y没缺失没加噪声前的原始数据，仅用来计算MSE，为避免信息泄露，训练阶段不被允许使用'
        Y_primitive_norm = Y_primitive_norm.permute(0, 2, 1) if self.args.BaseOn == 'reconstruct' else Y_primitive_norm.permute(0, 2, 1)[:, :, -self.args.pred_len:]
        'Y_primitive_norm: (batch_size, node_num, lag/pred_len)   normalized  Y没缺失没加噪声前的原始数据，仅用来计算MSE，为避免信息泄露，训练阶段不被允许使用'
        T = T.permute(0, 2, 1)
        'T: (batch_size, 4, lag)'
        T_of_y = T_of_y.permute(0, 2, 1)
        'T_of_y: (batch_size, 4, label_len+pred_len / lag)'
        # init_sample = init_sample.permute(0, 2, 1)
        # 'init_sample: (batch_size, node_num, lag)   not be normalized'
        # init_sample_norm = init_sample_norm.permute(0, 2, 1)
        # 'init_sample_norm: (batch_size, node_num, lag)   normalized'
        init_sample = None
        'init_sample: (batch_size, node_num, lag)   not be normalized'
        init_sample_norm = None
        'init_sample_norm: (batch_size, node_num, lag)   normalized'
        for key, value in scaler_info.items():
            scaler_info[key] = value[0,:]
        self.scaler_info = scaler_info
        "'scaler_info: dict, {'scale_list': scale_list, 'mean_list': mean_list, 'scale_list_work_condition': scale_list_work_condition, 'mean_list_work_condition': mean_list_work_condition}'"
        A = torch.Tensor(self.A).to(X.device) if self.A is not None else None
        'A: (node_num, node_num)'
        label = label[0] if label[0] != 'None' else None
        'label: (data_len)'
        all_label = all_label[0].permute(1, 0) if all_label[0] != 'None' else None
        'all_label: (node_num, data_len)'
        timestamp_label = timestamp_label[0]
        'timestamp_label: (data_len)'

        H_norm = self.SPS_Model(A, X, X_norm, WC, WC_norm, T, T_of_y, Y, init_sample, init_sample_norm, scaler_info, PI_info_batch)
        'H_norm: (batch_size, node_num, lag/pred_len)   normalized'

        if self.args.BaseOn == 'forecast':
            Y = Y[:, :, -self.args.pred_len:]
            'Y: R(batch_size, node_num, pred_len) or F(batch_size, node_num, pred_len)'
            Y_norm = Y_norm[:, :, -self.args.pred_len:]
            'Y_norm: R(batch_size, node_num, pred_len) or F(batch_size, node_num, pred_len)'
        if H_norm.shape != Y_norm.shape:
            raise Exception("H_norm.shape != Y_norm.shape, please check the model!")
        if self.args.channel_to_channel == 'MS':
            Y_norm = Y_norm[:, self.args.MS_which, :]
            Y_norm = Y_norm.unsqueeze(1)
            'Y_norm: (batch_size, 1, lag)'
            H_norm = H_norm[:, self.args.MS_which, :]
            H_norm = H_norm.unsqueeze(1)
            'H_norm: (batch_size, 1, pred_len)'

        ### 数据连接
        if batch_idx == 0:
            # 判断label是否为numpy数组，若是则转为tensor
            if isinstance(label, np.ndarray):
                label = torch.from_numpy(label)
            self.label = label.squeeze() if label.ndim > 1 else label
            'label: (data_len) epoch_end —> R啥也不做(data_len) or F剪掉前面(data_len - lag)'
            # 判断all_label是否为numpy数组，若是则转为tensor
            if all_label is not None: 
                if isinstance(all_label, np.ndarray):
                    all_label = torch.from_numpy(all_label)
            else:
                if label is not None:
                    all_label = self.label.unsqueeze(0).repeat(self.args.sensor_num, 1)
            self.all_label = all_label
            'all_label: (node_num, data_len) ' \
            'epoch_end —> R啥也不做(node_num, data_len) or F剪掉前面(node_num, data_len - lag)'
            # 判断timestamp_label是否为numpy数组，若是则转为tensor
            if isinstance(timestamp_label, np.ndarray):
                timestamp_label = torch.from_numpy(timestamp_label)
            self.timestamp_label = timestamp_label
            'timestamp_label: (data_len) epoch_end —> R啥也不做(data_len) or F剪掉前面(data_len - lag)'

            self.Y_orig = Y_norm[0, :, :]
            'Y_orig: R(node_num, data_len) or F(node_num, data_len - lag)   normalized'
            self.Y_fore = H_norm[0, :, :]
            'Y_fore: R(node_num, data_len) or F(node_num, data_len - lag)   normalized'

            self.Y_orig = torch.cat((self.Y_orig, Y_norm[1:, :, -self.args.lag_step:].permute(1, 0, 2).reshape(Y_norm.shape[1], -1)), dim=1)
            self.Y_fore = torch.cat((self.Y_fore, H_norm[1:, :, -self.args.lag_step:].permute(1, 0, 2).reshape(H_norm.shape[1], -1)), dim=1)
        else:
            self.Y_orig = torch.cat((self.Y_orig, Y_norm[:, :, -self.args.lag_step:].permute(1, 0, 2).reshape(Y_norm.shape[1], -1)), dim=1)
            self.Y_fore = torch.cat((self.Y_fore, H_norm[:, :, -self.args.lag_step:].permute(1, 0, 2).reshape(H_norm.shape[1], -1)), dim=1)

        return self.Y_orig, self.Y_fore

    def test_step_FC(self, batch, batch_idx):
        """如果是在SPS_Model_PINN时，可以直接使用物理信息的仿真结果"""
        if self.args.model_selection == 'SPS_Model_PINN':
            # X, X_norm, WC, WC_norm, Y, Y_norm, Y_primitive, Y_primitive_norm, Y_dirty, Y_dirty_norm, label, all_label, timestamp_label, T, T_of_y, scaler_info, PI_info_batch = batch
            X, X_norm, Y, Y_norm, Y_primitive, Y_primitive_norm, Y_dirty, Y_dirty_norm, label, all_label, timestamp_label, T, T_of_y, scaler_info, PI_info_batch = batch
            PI_info_batch = PI_info_batch.permute(0, 2, 1)
            'PI_info_batch: (batch_size, node_num, lag)   not be normalized'
        else:
            # X, X_norm, WC, WC_norm, Y, Y_norm, Y_primitive, Y_primitive_norm, Y_dirty, Y_dirty_norm, label, all_label, timestamp_label, T, T_of_y, scaler_info = batch
            X, X_norm, Y, Y_norm, Y_primitive, Y_primitive_norm, Y_dirty, Y_dirty_norm, label, all_label, timestamp_label, T, T_of_y, scaler_info = batch
            PI_info_batch = None
        X = X.permute(0, 2, 1)
        'X: (batch_size, node_num, lag)   not be normalized'
        X_norm = X_norm.permute(0, 2, 1)
        'X_norm: (batch_size, node_num, lag)   normalized'
        if self.args.if_add_work_condition:
            raise Exception("anomaly detection dataset do not return WC data, please check the dataset!")
            WC = WC.permute(0, 2, 1)
            'WC:工况 (batch_size, 4, lag)   not be normalized'
            WC_norm = WC_norm.permute(0, 2, 1)
            'WC_norm:工况 (batch_size, 4, lag)   normalized'
        else:
            WC = torch.zeros_like(X_norm[:, :4, :])
            'WC:工况 (batch_size, 4, lag)   not be normalized'
            WC_norm = torch.zeros_like(X_norm[:, :4, :])
            'WC_norm:工况 (batch_size, 4, lag)   normalized'
        Y = Y.permute(0, 2, 1) if self.args.BaseOn == 'reconstruct' else Y.permute(0, 2, 1)[:, :, -self.args.pred_len:]
        'Y: (batch_size, node_num, lag/pred_len)   not be normalized'
        Y_norm = Y_norm.permute(0, 2, 1) if self.args.BaseOn == 'reconstruct' else Y_norm.permute(0, 2, 1)[:, :, -self.args.pred_len:]
        'Y_norm: (batch_size, node_num, lag/pred_len)   normalized'
        Y_primitive = Y_primitive.permute(0, 2, 1) if self.args.BaseOn == 'reconstruct' else Y_primitive.permute(0, 2, 1)[:, :, -self.args.pred_len:]
        'Y_primitive: (batch_size, node_num, lag/pred_len)   not be normalized  Y没缺失没加噪声前的原始数据，仅用来计算MSE，为避免信息泄露，训练阶段不被允许使用'
        Y_primitive_norm = Y_primitive_norm.permute(0, 2, 1) if self.args.BaseOn == 'reconstruct' else Y_primitive_norm.permute(0, 2, 1)[:, :, -self.args.pred_len:]
        'Y_primitive_norm: (batch_size, node_num, lag/pred_len)   normalized  Y没缺失没加噪声前的原始数据，仅用来计算MSE，为避免信息泄露，训练阶段不被允许使用'
        T = T.permute(0, 2, 1)
        'T: (batch_size, 4, lag)'
        T_of_y = T_of_y.permute(0, 2, 1)
        'T_of_y: (batch_size, 4, label_len+pred_len / lag)'
        # init_sample = init_sample.permute(0, 2, 1)
        # 'init_sample: (batch_size, node_num, lag)   not be normalized'
        # init_sample_norm = init_sample_norm.permute(0, 2, 1)
        # 'init_sample_norm: (batch_size, node_num, lag)   normalized'
        init_sample = None
        'init_sample: (batch_size, node_num, lag)   not be normalized'
        init_sample_norm = None
        'init_sample_norm: (batch_size, node_num, lag)   normalized'
        for key, value in scaler_info.items():
            scaler_info[key] = value[0,:]
        self.scaler_info = scaler_info
        "'scaler_info: dict, {'scale_list': scale_list, 'mean_list': mean_list, 'scale_list_work_condition': scale_list_work_condition, 'mean_list_work_condition': mean_list_work_condition}'"
        A = torch.Tensor(self.A).to(X.device) if self.A is not None else None
        'A: (node_num, node_num)'

        H_norm = self.SPS_Model(A, X, X_norm, WC, WC_norm, T, T_of_y, Y, init_sample, init_sample_norm, scaler_info, PI_info_batch)
        'H_norm: (batch_size, node_num, lag/pred_len)   normalized'

        if self.args.BaseOn == 'forecast':
            Y = Y[:, :, -self.args.pred_len:]
            'Y: R(batch_size, node_num, pred_len) or F(batch_size, node_num, pred_len)'
            Y_norm = Y_norm[:, :, -self.args.pred_len:]
            'Y_norm: R(batch_size, node_num, pred_len) or F(batch_size, node_num, pred_len)'
        if H_norm.shape != Y_norm.shape:
            raise Exception("H_norm.shape != Y_norm.shape, please check the model!")
        ### 如果只预测OT列即最后一列，则只取最后一列，但要保持三维
        if self.args.channel_to_channel == 'MS':
            Y_norm = Y_norm[:, self.args.MS_which, :]
            Y_norm = Y_norm.unsqueeze(1)
            'Y: (batch_size, 1, lag)'
            H_norm = H_norm[:, self.args.MS_which, :]
            H_norm = H_norm.unsqueeze(1)
            'H: (batch_size, 1, pred_len)'

        ### 数据连接，这个会把滑窗滑出来的样本 重复的部分 去掉
        if batch_idx == 0:
            self.Y_orig = X_norm[0, :, :]
            'Y_orig: (node_num, data_len - lag)   normalized'
            self.Y_fore = X_norm[0, :, :]
            'Y_fore: (node_num, data_len - lag)   normalized'

            self.Y_orig = torch.cat((self.Y_orig, Y_norm[:, :, -self.args.lag_step:].permute(1, 0, 2).reshape(Y_norm.shape[1], -1)), dim=1)
            self.Y_fore = torch.cat((self.Y_fore, H_norm[:, :, -self.args.lag_step:].permute(1, 0, 2).reshape(H_norm.shape[1], -1)), dim=1)
        else:
            self.Y_orig = torch.cat((self.Y_orig, Y_norm[:, :, -self.args.lag_step:].permute(1, 0, 2).reshape(Y_norm.shape[1], -1)), dim=1)
            self.Y_fore = torch.cat((self.Y_fore, H_norm[:, :, -self.args.lag_step:].permute(1, 0, 2).reshape(H_norm.shape[1], -1)), dim=1)

        ### 但同类方法似乎大家默认是不去掉重复的部分的
        true = Y_primitive_norm.detach().cpu().numpy()
        'true: (batch_size, node_num, pred_len)'
        pred = H_norm.detach().cpu().numpy()
        'pred: (batch_size, node_num, pred_len)'
        self.trues.append(true)
        'trues: batch_num个 (batch_size, node_num, pred_len)'
        self.preds.append(pred)
        'preds: batch_num个 (batch_size, node_num, pred_len)'

        return self.Y_orig, self.Y_fore, self.trues, self.preds

    def test_step_RE(self, batch, batch_idx):
        if self.args.BaseOn != 'reconstruct':
            raise Exception("if TASK is 'reconstruct', BaseOn must be 'reconstruct'!")
        """如果是在SPS_Model_PINN时，可以直接使用物理信息的仿真结果"""
        if self.args.model_selection == 'SPS_Model_PINN':
            # X, X_norm, WC, WC_norm, Y, Y_norm, Y_primitive, Y_primitive_norm, Y_dirty, Y_dirty_norm, label, all_label, timestamp_label, T, T_of_y, scaler_info, PI_info_batch = batch
            X, X_norm, Y, Y_norm, Y_primitive, Y_primitive_norm, Y_dirty, Y_dirty_norm, label, all_label, timestamp_label, T, T_of_y, scaler_info, PI_info_batch = batch
            PI_info_batch = PI_info_batch.permute(0, 2, 1)
            'PI_info_batch: (batch_size, node_num, lag)   not be normalized'
        else:
            # X, X_norm, WC, WC_norm, Y, Y_norm, Y_primitive, Y_primitive_norm, Y_dirty, Y_dirty_norm, label, all_label, timestamp_label, T, T_of_y, scaler_info = batch
            X, X_norm, Y, Y_norm, Y_primitive, Y_primitive_norm, Y_dirty, Y_dirty_norm, label, all_label, timestamp_label, T, T_of_y, scaler_info = batch
            PI_info_batch = None
        X = X.permute(0, 2, 1)
        'X: (batch_size, node_num, lag)   not be normalized'
        X_norm = X_norm.permute(0, 2, 1)
        'X_norm: (batch_size, node_num, lag)   normalized'
        if self.args.if_add_work_condition:
            raise Exception("anomaly detection dataset do not return WC data, please check the dataset!")
            WC = WC.permute(0, 2, 1)
            'WC:工况 (batch_size, 4, lag)   not be normalized'
            WC_norm = WC_norm.permute(0, 2, 1)
            'WC_norm:工况 (batch_size, 4, lag)   normalized'
        else:
            WC = torch.zeros_like(X_norm[:, :4, :])
            'WC:工况 (batch_size, 4, lag)   not be normalized'
            WC_norm = torch.zeros_like(X_norm[:, :4, :])
            'WC_norm:工况 (batch_size, 4, lag)   normalized'
        Y = Y.permute(0, 2, 1) if self.args.BaseOn == 'reconstruct' else Y.permute(0, 2, 1)[:, :, -self.args.pred_len:]
        'Y: (batch_size, node_num, lag/pred_len)   not be normalized'
        Y_norm = Y_norm.permute(0, 2, 1) if self.args.BaseOn == 'reconstruct' else Y_norm.permute(0, 2, 1)[:, :, -self.args.pred_len:]
        'Y_norm: (batch_size, node_num, lag/pred_len)   normalized'
        Y_primitive = Y_primitive.permute(0, 2, 1) if self.args.BaseOn == 'reconstruct' else Y_primitive.permute(0, 2, 1)[:, :, -self.args.pred_len:]
        'Y_primitive: (batch_size, node_num, lag/pred_len)   not be normalized  Y没缺失没加噪声前的原始数据，仅用来计算MSE，为避免信息泄露，训练阶段不被允许使用'
        Y_primitive_norm = Y_primitive_norm.permute(0, 2, 1) if self.args.BaseOn == 'reconstruct' else Y_primitive_norm.permute(0, 2, 1)[:, :, -self.args.pred_len:]
        'Y_primitive_norm: (batch_size, node_num, lag/pred_len)   normalized  Y没缺失没加噪声前的原始数据，仅用来计算MSE，为避免信息泄露，训练阶段不被允许使用'
        T = T.permute(0, 2, 1)
        'T: (batch_size, 4, lag)'
        T_of_y = T_of_y.permute(0, 2, 1)
        'T_of_y: (batch_size, 4, label_len+pred_len / lag)'
        # init_sample = init_sample.permute(0, 2, 1)
        # 'init_sample: (batch_size, node_num, lag)   not be normalized'
        # init_sample_norm = init_sample_norm.permute(0, 2, 1)
        # 'init_sample_norm: (batch_size, node_num, lag)   normalized'
        init_sample = None
        'init_sample: (batch_size, node_num, lag)   not be normalized'
        init_sample_norm = None
        'init_sample_norm: (batch_size, node_num, lag)   normalized'
        for key, value in scaler_info.items():
            scaler_info[key] = value[0,:]
        self.scaler_info = scaler_info
        "'scaler_info: dict, {'scale_list': scale_list, 'mean_list': mean_list, 'scale_list_work_condition': scale_list_work_condition, 'mean_list_work_condition': mean_list_work_condition}'"
        A = torch.Tensor(self.A).to(X.device) if self.A is not None else None
        'A: (node_num, node_num)'

        H_norm = self.SPS_Model(A, X, X_norm, WC, WC_norm, T, T_of_y, Y, init_sample, init_sample_norm, scaler_info, PI_info_batch)
        'H_norm: (batch_size, node_num, lag/pred_len)   normalized'

        if self.args.BaseOn == 'forecast':
            Y = Y[:, :, -self.args.pred_len:]
            'Y: R(batch_size, node_num, pred_len) or F(batch_size, node_num, pred_len)'
            Y_norm = Y_norm[:, :, -self.args.pred_len:]
            'Y_norm: R(batch_size, node_num, pred_len) or F(batch_size, node_num, pred_len)'
        if H_norm.shape != Y_norm.shape:
            raise Exception("H_norm.shape != Y_norm.shape, please check the model!")
        if self.args.channel_to_channel == 'MS':
            Y_norm = Y_norm[:, self.args.MS_which, :]
            Y_norm = Y_norm.unsqueeze(1)
            'Y: R(batch_size, 1, lag) or F(batch_size, 1, pred_len)'
            H_norm = H_norm[:, self.args.MS_which, :]
            H_norm = H_norm.unsqueeze(1)
            'H: R(batch_size, 1, lag) or F(batch_size, 1, pred_len)'

        if batch_idx == 0:
            self.Y_orig = Y_norm[0, :, :]
            'Y_orig: R(node_num, data_len) or F(node_num, data_len - lag)    normalized'
            self.Y_fore = H_norm[0, :, :]
            'Y_orig: R(node_num, data_len) or F(node_num, data_len - lag)    normalized'
            self.Y_orig = torch.cat((self.Y_orig, Y_norm[1:, :, -self.args.lag_step:].permute(1, 0, 2).reshape(Y_norm.shape[1], -1)), dim=1)
            self.Y_fore = torch.cat((self.Y_fore, H_norm[1:, :, -self.args.lag_step:].permute(1, 0, 2).reshape(H_norm.shape[1], -1)), dim=1)
        else:
            self.Y_orig = torch.cat((self.Y_orig, Y_norm[:, :, -self.args.lag_step:].permute(1, 0, 2).reshape(Y_norm.shape[1], -1)), dim=1)
            self.Y_fore = torch.cat((self.Y_fore, H_norm[:, :, -self.args.lag_step:].permute(1, 0, 2).reshape(H_norm.shape[1], -1)), dim=1)

        return self.Y_orig, self.Y_fore

    def on_test_epoch_end(self):
        "虽然是叫test_epoch_end，但其实测试的epoch只一次，并不会有多个epoch"
        if self.args.TASK == 'anomaly_detection':
            self.on_test_epoch_end_AD(self.Y_orig, self.Y_fore, self.label)
        elif self.args.TASK == 'forecast':
            self.on_test_epoch_end_FC(self.Y_orig, self.Y_fore, self.trues, self.preds)
        elif self.args.TASK == 'reconstruct':
            self.on_test_epoch_end_RE(self.Y_orig, self.Y_fore)
        else:
            raise Exception("No such TASK! must be 'anomaly detection' or 'forecast' or 'reconstruct'!")

        # scale_tensor = torch.tensor(self.scaler_info['scale_list'], dtype=self.Y_orig.dtype, device=self.Y_orig.device).unsqueeze(1)
        scale_tensor = self.scaler_info['scale_list'].unsqueeze(1)
        'scale_tensor: (node_num, 1)'
        # mean_tensor = torch.tensor(self.scaler_info['mean_list'], dtype=self.Y_orig.dtype, device=self.Y_orig.device).unsqueeze(1)
        mean_tensor = self.scaler_info['mean_list'].unsqueeze(1)
        'mean_tensor: (node_num, 1)'
        # 如果是在SPS_Model_Phy时，参数定型后，将所有数据仿真一遍，保存仿真结果，用于PINN加速训练不再需要batch_size设置为1了有了这个结果
        if self.args.if_save_simulate_result and self.args.model_selection == 'SPS_Model_Phy':
            if self.args.SPS_Model_PINN_if_has_Phy_of_BCR:
                file_name = 'physical_simulate_result/' + str(self.args.data_name) + '_' + str(self.args.BaseOn) + '_physical_simulate_result.csv'
                save_path = os.path.join(self.args.root_path, self.args.data_path, file_name)
                # 将self.Y_fore转为pandas并保存为csv
                Y_fore = self.Y_fore * scale_tensor + mean_tensor
                Y_fore = pd.DataFrame(Y_fore.permute(1, 0).detach().cpu().numpy())
                Y_fore.to_csv(save_path, index=False)
            elif not self.args.SPS_Model_PINN_if_has_Phy_of_BCR:
                file_name = 'physical_simulate_result/' + str(self.args.data_name) + '_' + str(self.args.BaseOn) + '_physical_simulate_result_withoutBCR.csv'
                save_path = os.path.join(self.args.root_path, self.args.data_path, file_name)
                # 将self.Y_fore转为pandas并保存为csv
                Y_fore = self.Y_fore * scale_tensor + mean_tensor
                Y_fore = pd.DataFrame(Y_fore.permute(1, 0).detach().cpu().numpy())
                Y_fore.to_csv(save_path, index=False)
            else:
                raise Exception("if_save_simulate_result is True, but SPS_Model_PINN_if_has_Phy_of_BCR is not set!")



    def on_test_epoch_end_AD(self, Y_orig, Y_fore, label):
        """

        Args:
            Y_orig: 原始数据 (node_num, data_len)
            Y_fore: 重建/预测数据 (node_num, data_len)/(node_num, data_len - lag)
            label: 标签 (data_len)

        Returns:

        """
        if self.args.BaseOn == 'forecast':
            # self.Y_orig = Y_orig[:, self.args.lag:]
            # 'Y_orig: (node_num, data_len - lag)'
            self.label = label[self.args.lag:]
            'label: (data_len - lag)'
            if self.all_label is not None: self.all_label = self.all_label[:, self.args.lag:]
            'all_label: (node_num, data_len - lag)'
            self.timestamp_label = self.timestamp_label[self.args.lag:]
            'timestamp_label: (data_len - lag)'
        if self.Y_orig.shape != self.Y_fore.shape:
            raise Exception("Y_orig.shape != Y_fore.shape, please check the code!")
        if self.label.shape[0] != self.Y_orig.shape[1]:
            raise Exception("label.shape[0] != Y_orig.shape[1], please check the code!")
            # '如果报错，可能是因为batch_size太小，导致最后一个batch的数据不够，后面的一段数据被剪掉了'
            # '但一般来说我会设置就算最后一个batch不够，也依然进行训练和测试，所以这个报错应该检查前面'

        # 计算异常分数
        S = self.Y_orig - self.Y_fore
        'S:重建/预测误差，即异常分数矩阵 R(node_num, data_len) / F(node_num, data_len - lag)'
        S = torch.abs(S)
        # 也可以把异常分数滑动平均一下
        if self.args.S_moving_average_window != 1:
            S = moving_average(S, self.args.S_moving_average_window)
        self.S = S
        'S:异常分数矩阵 R(node_num, data_len) / F(node_num, data_len - lag)'

        # 计算异常检测各指标，自动化得到阈值的
        result_AD = performance_AD_auto(S.cpu().numpy(), self.label.cpu().numpy(), threshold=self.args.AD_threshold)
        print(f'F1 score: {result_AD[0]}')
        print(f'accuracy: {result_AD[1]}')
        print(f'precision: {result_AD[2]}')
        print(f'recall: {result_AD[3]}')
        print(f'AUC(ROC下面积): {result_AD[4]}')
        print(f'异常检测推荐选取的阈值: {result_AD[5]}')

        # "/home/dy29/MyWorks_Codes/15_LLM_4_SPS_PHM/AD_repository/model/AD_algorithms_params/all_AD_algorithm_params.yaml"
        # 训练的时候，args是从ckpt权重文件里面导入的，但上面yaml文件里面的AD_threshold可能已经更新了，所以要导入并更新
        with open("/home/dy29/MyWorks_Codes/15_LLM_4_SPS_PHM/AD_repository/model/AD_algorithms_params/all_AD_algorithm_params.yaml", "r") as f:
            loaded_params = yaml.safe_load(f)
        self.args.AD_threshold = loaded_params["Common_configs"]['AD_threshold']['value']
        # args.AD_threshold = loaded_params["Common_configs"]['AD_threshold']['value']

        # 将异常分数与阈值进行比较，判断异常区域，对应timestamp_list得到检测到的异常时间片段列表，并将阈值、异常占比、异常时间片段等各种信息 保存为json文件
        fragments, threshold, anomaly_ratio = detect_AD_fragment(Score_AD=S.cpu().numpy(), 
                                                                threshold=self.args.AD_threshold, 
                                                                json_save_path = self.args.table_save_path, 
                                                                timestamp_label=self.timestamp_label.cpu().numpy(),
                                                                recommend_threshold=result_AD[5],
                                                                mean_or_each='each')

        # 当初标准化和归一化了，有需要的话可以反标准化和反归一化回来
        if self.args.scale and self.args.inverse:
            self.Y_orig = self.trainer.datamodule.data_set.my_inverse_transform(self.Y_orig)
            self.Y_fore = self.trainer.datamodule.data_set.my_inverse_transform(self.Y_fore)

        # 计算对正常样本的重建/预测的 MSE\MAE
        Y_orig_normal = self.Y_orig[:, self.label == 0]
        'Y_orig_normal: (node_num, data_len_normal)'
        Y_fore_normal = self.Y_fore[:, self.label == 0]
        'Y_fore_normal: (node_num, data_len_normal)'
        result_FC = performance_FC( Y_orig_normal.cpu().numpy(), Y_fore_normal.cpu().numpy())
        print(f'AD_normal_MSE: {result_FC[0]}')
        print(f'AD_normal_MAE: {result_FC[1]}')
        # print(f'RMSE: {result_FC[2]}')
        # print(f'MAPE: {result_FC[3]}')
        # print(f'MSPE: {result_FC[4]}')
        if self.args.add_noise_SNR != 0 or self.args.missing_rate != 0:
            print('请注意，如果加入了噪声或者大量缺失，这里的MSE是没有参考意义的，因为用于计算MSE的Y也是加了噪声的,'
                  '请修改代码，在test_step中使用Y_primitive_norm而不是Y_norm，这样计算的MSE才有参考意义')

        self.exam_result = {'F1': result_AD[0], 'precision': result_AD[2], 'recall': result_AD[3],
                            'epoch_train': self.train_epoch_end,
                            'AUC': result_AD[4], 'accuracy': result_AD[1],
                            'normal_MSE': result_FC[0], 'normal_MAE': result_FC[1], 'test_time': self.test_time}

        ray_metric = 100 / result_AD[0]

        self.log('train_end_loss', self.training_loss, prog_bar=True)
        self.log('ray_metric', ray_metric, prog_bar=True)

        self.log('AD_F1', result_AD[0], prog_bar=True)
        self.log('AD_precision', result_AD[2], prog_bar=True)
        self.log('AD_recall', result_AD[3], prog_bar=True)
        self.log('AD_AUC', result_AD[4], prog_bar=True)
        self.log('AD_accuracy', result_AD[1], prog_bar=True)
        # self.log('AD_recommend_threshold', result_AD[5], prog_bar=True)
        self.log('AD_normal_MSE', result_FC[0], prog_bar=True)
        self.log('AD_normal_MAE', result_FC[1], prog_bar=True)

        self.log('FC_MSE', 0, prog_bar=False)
        self.log('FC_MAE', 0, prog_bar=False)
        self.log('FC_RMSE', 0, prog_bar=False)
        self.log('FC_MAPE', 0, prog_bar=False)
        self.log('FC_MSPE', 0, prog_bar=False)

        # 将结果保存到csv文件中
        dirname_path = self.args.table_save_path + '/'
        if not os.path.exists(dirname_path + self.args.exp_name[:66] + '_test.csv'):
            os.makedirs(dirname_path, exist_ok=True)
            args_list = vars(self.args)
            args_dict = {k: str(v) for k, v in args_list.items()}
            save_dict = {**self.exam_result, **args_dict}
            df = pd.DataFrame(save_dict, index=[0])
            df.to_csv(self.args.table_save_path + '/' + self.args.exp_name[:66] + '_test.csv', index=False, mode='a',
                      header=True)
            # header=True表示保存列索引,不存在文件，第一次创建保存表头
        else:
            args_list = vars(self.args)
            args_dict = {k: str(v) for k, v in args_list.items()}
            save_dict = {**self.exam_result, **args_dict}
            df = pd.DataFrame(save_dict, index=[0])
            df.to_csv(self.args.table_save_path + '/' + self.args.exp_name[:66] + '_test.csv', index=False, mode='a',
                      header=False)
            # index=False表示不保存行索引，已有文件，不保存表头

        # 画图
        if self.args.if_plot:
            # 先根据self.scaler_info将数据反标准化和反归一化回来
            if not (self.args.scale and self.args.inverse):
                # scale_tensor = torch.tensor(self.scaler_info['scale_list'], dtype=self.Y_orig.dtype, device=self.Y_orig.device).unsqueeze(1)
                scale_tensor = self.scaler_info['scale_list'].unsqueeze(1)
                'scale_tensor: (node_num, 1)'
                # mean_tensor = torch.tensor(self.scaler_info['mean_list'], dtype=self.Y_orig.dtype, device=self.Y_orig.device).unsqueeze(1)
                mean_tensor = self.scaler_info['mean_list'].unsqueeze(1)
                'mean_tensor: (node_num, 1)'
                Y_orig = self.Y_orig * scale_tensor + mean_tensor
                'Y_orig: (node_num, data_len)     not normalized'
                Y_fore = self.Y_fore * scale_tensor + mean_tensor
                'Y_fore: (node_num, data_len)     not normalized'
            else:
                Y_orig = self.Y_orig
                'Y_orig: (node_num, data_len)     normalized'
                Y_fore = self.Y_fore
                'Y_fore: (node_num, data_len)     normalized'
            # 先计算异常检测01结果矩阵
            # self.detect_01 = torch.where(torch.tensor(S>result_AD[5]).to(S.device),
            #                              torch.tensor([1]).to(S.device),
            #                              torch.tensor([0]).to(S.device))
            self.detect_01 = torch.where(torch.tensor(S>self.args.AD_threshold).to(S.device),
                                         torch.tensor([1]).to(S.device),
                                         torch.tensor([0]).to(S.device))
            MyPlot_AD(self.args,
                      Y_orig, Y_fore, 
                      self.label, self.all_label,
                      self.S, self.detect_01,
                      self.exam_result, args_dict, self.timestamp_label)

        print('this experiment finished')

        return self.exam_result

    def on_test_epoch_end_FC(self, Y_orig, Y_fore, trues, preds):
        """

        Args:
            Y_orig: 原始数据 (node_num, data_len)
            Y_fore: 重建/预测数据 (node_num, data_len)/(node_num, data_len - lag)
            trues: 不去掉滑窗重复部分的 真实数据 (batch_num, batch_size, node_num, pred_len)
            preds: 不去掉滑窗重复部分的 预测数据 (batch_num, batch_size, node_num, pred_len)

        Returns:

        """
        # if self.args.BaseOn == 'forecast':
        #     self.Y_orig = Y_orig[:, self.args.lag:]
        #     'Y_orig: (node_num, data_len - lag)'
        if self.Y_orig.shape != self.Y_fore.shape:
            raise Exception("Y_orig.shape != Y_fore.shape, please check the code!")

        trues = np.concatenate(self.trues, axis=0)
        'trues: (batch_num*batch_size, node_num, pred_len)'
        trues = np.transpose(trues, (1, 0, 2))
        'trues: (node_num, batch_num*batch_size, pred_len)'
        trues = trues.reshape(trues.shape[0], -1)
        'trues: (node_num, batch_num*batch_size*pred_len)'
        preds = np.concatenate(self.preds, axis=0)
        'preds: (batch_num*batch_size, node_num, pred_len)'
        preds = np.transpose(preds, (1, 0, 2))
        'preds: (node_num, batch_num*batch_size, pred_len)'
        preds = preds.reshape(preds.shape[0], -1)
        'preds: (node_num, batch_num*batch_size*pred_len)'

        # 当初标准化和归一化了，有需要的话可以反标准化和反归一化回来
        if self.args.scale and self.args.inverse:
            self.Y_orig = self.trainer.datamodule.data_set.my_inverse_transform(self.Y_orig)
            self.Y_fore = self.trainer.datamodule.data_set.my_inverse_transform(self.Y_fore)
            trues = self.trainer.datamodule.data_set.my_inverse_transform(trues)
            preds = self.trainer.datamodule.data_set.my_inverse_transform(preds)

        # 计算预测的 MSE\MAE
        # 本来我计算MSE啥的是用把滑窗滑出来的样本 重复的部分 去掉的
        # result_FC = performance_FC(self.Y_orig.cpu().numpy(), self.Y_fore.cpu().numpy())
        # 但同类方法似乎大家默认是不去掉重复的部分的，那就只能：
        result_FC = performance_FC(trues, preds)

        print(f'MSE: {result_FC[0]}')
        print(f'MAE: {result_FC[1]}')
        print(f'RMSE: {result_FC[2]}')
        print(f'MAPE: {result_FC[3]}')
        print(f'MSPE: {result_FC[4]}')
        print(f'RSE: {result_FC[5]}')
        print(f'CORR: {result_FC[6]}')

        self.exam_result = {'MSE': result_FC[0], 'MAE': result_FC[1],
                            'epoch_train': self.train_epoch_end,
                            'RMSE': result_FC[2], 'MAPE': result_FC[3], 'MSPE': result_FC[4],
                            'test_time': self.test_time,
                            'RSE': result_FC[5], 'CORR': result_FC[6]}

        ray_metric = result_FC[0]

        self.log('train_end_loss', self.training_loss, prog_bar=True)
        self.log('ray_metric', ray_metric, prog_bar=True)

        self.log('AD_F1', 0, prog_bar=False)
        self.log('AD_precision', 0, prog_bar=False)
        self.log('AD_recall', 0, prog_bar=False)
        self.log('AD_AUC', 0, prog_bar=False)
        self.log('AD_accuracy', 0, prog_bar=False)
        self.log('AD_threshold', 0, prog_bar=False)
        self.log('AD_normal_MSE', 0, prog_bar=False)
        self.log('AD_normal_MAE', 0, prog_bar=False)

        self.log('FC_MSE', result_FC[0], prog_bar=True)
        self.log('FC_MAE', result_FC[1], prog_bar=True)
        self.log('FC_RMSE', result_FC[2], prog_bar=True)
        self.log('FC_MAPE', result_FC[3], prog_bar=True)
        self.log('FC_MSPE', result_FC[4], prog_bar=True)
        self.log('FC_RSE', result_FC[5], prog_bar=True)
        self.log('FC_CORR', result_FC[6], prog_bar=True)

        # 将结果保存到csv文件中
        dirname_path = self.args.table_save_path + '/'
        if not os.path.exists(dirname_path + self.args.exp_name[:66] + '_test.csv'):
            os.makedirs(dirname_path, exist_ok=True)
            args_list = vars(self.args)
            args_dict = {k: str(v) for k, v in args_list.items()}
            save_dict = {**self.exam_result, **args_dict}
            df = pd.DataFrame(save_dict, index=[0])
            df.to_csv(self.args.table_save_path + '/' + self.args.exp_name[:66] + '_test.csv', index=False, mode='a',
                      header=True)
            # header=True表示保存列索引,不存在文件，第一次创建保存表头
        else:
            args_list = vars(self.args)
            args_dict = {k: str(v) for k, v in args_list.items()}
            save_dict = {**self.exam_result, **args_dict}
            df = pd.DataFrame(save_dict, index=[0])
            df.to_csv(self.args.table_save_path + '/' + self.args.exp_name[:66] + '_test.csv', index=False, mode='a',
                      header=False)
            # index=False表示不保存行索引，已有文件，不保存表头

        # 画图
        if self.args.if_plot:
            # 先根据self.scaler_info将数据反标准化和反归一化回来
            if not (self.args.scale and self.args.inverse):
                # scale_tensor = torch.tensor(self.scaler_info['scale_list'], dtype=self.Y_orig.dtype, device=self.Y_orig.device).unsqueeze(1)
                scale_tensor = self.scaler_info['scale_list'].unsqueeze(1)
                'scale_tensor: (node_num, 1)'
                # mean_tensor = torch.tensor(self.scaler_info['mean_list'], dtype=self.Y_orig.dtype, device=self.Y_orig.device).unsqueeze(1)
                mean_tensor = self.scaler_info['mean_list'].unsqueeze(1)
                'scale_tensor: (node_num, 1)'
                Y_orig = self.Y_orig * scale_tensor + mean_tensor
                'Y_orig: (node_num, data_len)     not normalized'
                Y_fore = self.Y_fore * scale_tensor + mean_tensor
                'Y_fore: (node_num, data_len)     not normalized'
            else:
                Y_orig = self.Y_orig
                Y_fore = self.Y_fore
                'Y_orig: (node_num, data_len)     normalized'
                'Y_fore: (node_num, data_len)     normalized'
            MyPlot_FC(self.args,
                      Y_orig, Y_fore,
                      self.exam_result, args_dict)

        print('this experiment finished')

        return self.exam_result

    def on_test_epoch_end_RE(self, Y_orig, Y_fore):
        """

        Args:
            Y_orig: 原始数据 (node_num, data_len)
            Y_fore: 重建/预测数据 (node_num, data_len)/(node_num, data_len - lag)

        Returns:

        """
        # if self.args.BaseOn == 'forecast':
        #     self.Y_orig = Y_orig[:, self.args.lag:]
        #     'Y_orig: (node_num, data_len - lag)'
        if self.Y_orig.shape != self.Y_fore.shape:
            raise Exception("Y_orig.shape != Y_fore.shape, please check the code!")

        # 当初标准化和归一化了，有需要的话可以反标准化和反归一化回来
        if self.args.scale and self.args.inverse:
            self.Y_orig = self.trainer.datamodule.data_set.my_inverse_transform(self.Y_orig)
            self.Y_fore = self.trainer.datamodule.data_set.my_inverse_transform(self.Y_fore)

        # 计算预测的 MSE\MAE
        result_FC = performance_FC(self.Y_orig.cpu().numpy(), self.Y_fore.cpu().numpy())
        print(f'MSE: {result_FC[0]}')
        print(f'MAE: {result_FC[1]}')
        print(f'RMSE: {result_FC[2]}')
        print(f'MAPE: {result_FC[3]}')
        print(f'MSPE: {result_FC[4]}')
        print(f'RSE: {result_FC[5]}')
        print(f'CORR: {result_FC[6]}')

        if self.args.add_noise_SNR != 0 or self.args.missing_rate != 0.0:
            raise Exception("if add_noise_SNR != 0 or missing_rate != 0.0"
                            ", the MSE is not meaningful, "
                            "because the Y which used to calculate MSE is added noise or missing, "
                            "please modeify the code to calculate the MSE use Y_primitive_norm in test_step rather than Y_norm")

        self.exam_result = {'MSE': result_FC[0], 'MAE': result_FC[1],
                            'epoch_train': self.train_epoch_end,
                            'RMSE': result_FC[2], 'MAPE': result_FC[3], 'MSPE': result_FC[4],
                            'test_time': self.test_time,
                            'RSE': result_FC[5], 'CORR': result_FC[6]}

        ray_metric = result_FC[0]

        self.log('train_end_loss', self.training_loss, prog_bar=True)
        self.log('ray_metric', ray_metric, prog_bar=True)

        self.log('AD_F1', 0, prog_bar=False)
        self.log('AD_precision', 0, prog_bar=False)
        self.log('AD_recall', 0, prog_bar=False)
        self.log('AD_AUC', 0, prog_bar=False)
        self.log('AD_accuracy', 0, prog_bar=False)
        self.log('AD_threshold', 0, prog_bar=False)
        self.log('AD_normal_MSE', 0, prog_bar=False)
        self.log('AD_normal_MAE', 0, prog_bar=False)

        self.log('FC_MSE', result_FC[0], prog_bar=True)
        self.log('FC_MAE', result_FC[1], prog_bar=True)
        self.log('FC_RMSE', result_FC[2], prog_bar=True)
        self.log('FC_MAPE', result_FC[3], prog_bar=True)
        self.log('FC_MSPE', result_FC[4], prog_bar=True)
        self.log('FC_RSE', result_FC[5], prog_bar=True)
        self.log('FC_CORR', result_FC[6], prog_bar=True)

        # 将self.SPS_Model模型及其子模型中，所有只有单个元素的参数，保存到csv文件中
        print('saving SPS_Model parameters to csv file...')
        # single_element_params_dict = save_single_element_params_to_csv(self.SPS_Model)
        single_element_params_dict = {}
        for p in self.named_parameters():
            if p[1].numel() == 1:
                single_element_params_dict[p[0]] = p[1].item()

        # 将结果保存到csv文件中
        dirname_path = self.args.table_save_path + '/'
        if not os.path.exists(dirname_path + self.args.exp_name[:66] + '_test.csv'):
            os.makedirs(dirname_path, exist_ok=True)
            args_list = vars(self.args)
            args_dict = {k: str(v) for k, v in args_list.items()}
            # save_dict = {**self.exam_result, **args_dict}
            save_dict = {**self.exam_result, **args_dict, **single_element_params_dict}
            df = pd.DataFrame(save_dict, index=[0])
            df.to_csv(self.args.table_save_path + '/' + self.args.exp_name[:66] + '_test.csv', index=False, mode='a',
                      header=True)
            # header=True表示保存列索引,不存在文件，第一次创建保存表头
        else:
            args_list = vars(self.args)
            args_dict = {k: str(v) for k, v in args_list.items()}
            # save_dict = {**self.exam_result, **args_dict}
            save_dict = {**self.exam_result, **args_dict, **single_element_params_dict}
            df = pd.DataFrame(save_dict, index=[0])
            df.to_csv(self.args.table_save_path + '/' + self.args.exp_name[:66] + '_test.csv', index=False, mode='a',
                      header=False)
            # index=False表示不保存行索引，已有文件，不保存表头

        # 画图
        if self.args.if_plot:
            # 先根据self.scaler_info将数据反标准化和反归一化回来
            if not (self.args.scale and self.args.inverse):
                # scale_tensor = torch.tensor(self.scaler_info['scale_list'], dtype=self.Y_orig.dtype, device=self.Y_orig.device).unsqueeze(1)
                scale_tensor = self.scaler_info['scale_list'].unsqueeze(1)
                'scale_tensor: (node_num, 1)'
                # mean_tensor = torch.tensor(self.scaler_info['mean_list'], dtype=self.Y_orig.dtype, device=self.Y_orig.device).unsqueeze(1)
                mean_tensor = self.scaler_info['mean_list'].unsqueeze(1)
                'mean_tensor: (node_num, 1)'
                Y_orig = self.Y_orig * scale_tensor + mean_tensor
                'Y_orig: (node_num, data_len)     not normalized'
                Y_fore = self.Y_fore * scale_tensor + mean_tensor
                'Y_fore: (node_num, data_len)     not normalized'
            else:
                Y_orig = self.Y_orig
                Y_fore = self.Y_fore
                'Y_orig: (node_num, data_len)     normalized'
                'Y_fore: (node_num, data_len)     normalized'
            MyPlot_RE(self.args,
                      Y_orig, Y_fore,
                      self.exam_result, args_dict)

        print('this experiment finished')

        return self.exam_result

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        if batch_idx ==0:
            print('this experiment finished')






