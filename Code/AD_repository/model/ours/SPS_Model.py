import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from AD_repository.model.ours.SPS_Model_PINN import SPS_Model_PINN
from AD_repository.model.ours.SPS_Model_NN import SPS_Model_NN
from AD_repository.model.ours.SPS_Model_Phy import SPS_Model_Phy
from AD_repository.model.ours.SPS_Model_PhyOpt import SPS_Model_PhyOpt
from AD_repository.model.ours.SPS_Model_Opt import SPS_Model_Opt
from AD_repository.model.ours.SPS_Model_Phy_wo_BCR import SPS_Model_Phy_wo_BCR




class SPS_Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.H_last = None

        # SPS_Model_PINN/SPS_Model_NN/SPS_Model_Phy/SPS_Model_Opt
        if args.model_selection == 'SPS_Model_PINN':
            self.SPS_Model = SPS_Model_PINN(args)
        elif args.model_selection == 'SPS_Model_NN':
            self.SPS_Model = SPS_Model_NN(args)
        elif args.model_selection == 'SPS_Model_Phy' and args.SPS_Model_PINN_if_has_Phy_of_BCR:
            self.SPS_Model = SPS_Model_PhyOpt(exp_frequency=args.exp_frequency,
                                           lag_step=args.lag_step,
                                           BAT_QU_curve_app_order=args.BAT_QU_curve_app_order,
                                           Load_TP_curve_app_order=args.Load_TP_curve_app_order,
                                           SOC_init=args.SOC_init,
                                           SOC_if_trickle_charge=args.SOC_if_trickle_charge)
        elif args.model_selection == 'SPS_Model_Phy' and not args.SPS_Model_PINN_if_has_Phy_of_BCR:
            self.SPS_Model = SPS_Model_Phy_wo_BCR(exp_frequency=args.exp_frequency,
                                                  lag_step=args.lag_step,
                                                  BAT_QU_curve_app_order=args.BAT_QU_curve_app_order,
                                                  Load_TP_curve_app_order=args.Load_TP_curve_app_order,
                                                  SOC_init=args.SOC_init,
                                                  SOC_if_trickle_charge=args.SOC_if_trickle_charge,
                                                  BCR_MLP_hidden_dim=args.BCR_MLP_hidden_dim,
                                                  BCR_MLP_lay_num=args.BCR_MLP_lay_num,
                                                  dropout=args.dropout,
                                                  LeakyReLU_slope=args.LeakyReLU_slope)
        elif args.model_selection == 'SPS_Model_Opt':
            self.SPS_Model = SPS_Model_Opt(exp_frequency=args.exp_frequency,
                                           lag_step=args.lag_step,
                                           BAT_QU_curve_app_order=args.BAT_QU_curve_app_order,
                                           Load_TP_curve_app_order=args.Load_TP_curve_app_order,
                                           SOC_init=args.SOC_init,
                                           SOC_if_trickle_charge=args.SOC_if_trickle_charge)
        else:
            raise Exception("No such model_selection! must be 'SPS_Model_PINN' or 'SPS_Model_NN' or 'SPS_Model_Phy' or 'SPS_Model_Opt'!")

    def forward(self, A, X, X_norm, WC, WC_norm, T, T_of_y, Y, init_sample, init_sample_norm, scaler_info, PI_info_batch):
        """
        :param A: (node_num, node_num)
        :param X: (batch_size, node_num, lag)
        :param X_norm: (batch_size, node_num, lag)
        :param WC: (batch_size, 4, lag), actually are working conditions: irradiance, temperature, wind speed, load
        :param WC_norm: (batch_size, 4, lag), normalized working conditions
        :param T: (batch_size, 4, lag), DAY, HOUR, MINUTE, SECOND
        :param T_of_y: (batch_size, 4, pred_len), DAY, HOUR, MINUTE, SECOND, used for transformers as Autoregressive beginning
        :param Y: R(batch_size, node_num, lag) or F(batch_size, node_num, label_len+pred_len), used for transformers as Autoregressive beginning
        :param init_sample: (batch_size, node_num, lag) the initial state at the beginning of the simulation
        :param init_sample_norm: (batch_size, node_num, lag) the normalized initial state at the beginning of the simulation
        :param scaler_info: dict, {'scale_list': scale_list, 'mean_list': mean_list, 'scale_list_work_condition': scale_list_work_condition, 'mean_list_work_condition': mean_list_work_condition}
        :param PI_info_batch: (batch_size, node_num, lag), PI_info_batch is the PI information, simluation data

        return: H_norm: (batch_size, node_num, lag)   normalized  or  H2_norm: (batch_size, node_num, pred_len)   normalized
        """
        if self.args.BaseOn == "reconstruct":
            H_norm = self.deploy_SPS_Model(A, X, X_norm, WC, WC_norm, T, T_of_y, Y, init_sample, init_sample_norm, scaler_info, PI_info_batch)
            'H_norm: (batch_size, node_num, lag)'
            return H_norm
        elif self.args.BaseOn == "forecast":
            H_norm = self.deploy_SPS_Model(A, X, X_norm, WC, WC_norm, T, T_of_y, Y, init_sample, init_sample_norm, scaler_info, PI_info_batch)
            'H: (batch_size, node_num, lag)'
            H2_norm = H_norm[:, :, -self.args.pred_len:]
            'H2_norm: (batch_size, node_num, pred_len)'
            return H2_norm
        else:
            raise Exception("No such BaseOn! must be 'reconstruct' or 'forecast'!")

    def deploy_SPS_Model(self, A, X, X_norm, WC, WC_norm, T, T_of_y, Y, init_sample, init_sample_norm, scaler_info, PI_info_batch):
        """
        :param A: (node_num, node_num)
        :param X: (batch_size, node_num, lag)
        :param X_norm: (batch_size, node_num, lag)
        :param WC: (batch_size, 4, lag), actually are working conditions: irradiance, temperature, wind speed, load
        :param WC_norm: (batch_size, 4, lag), normalized working conditions
        :param T: (batch_size, 4, lag), DAY, HOUR, MINUTE, SECOND
        :param T_of_y: (batch_size, 4, pred_len), DAY, HOUR, MINUTE, SECOND, used for transformers as Autoregressive beginning
        :param Y: R(batch_size, node_num, lag) or F(batch_size, node_num, label_len+pred_len), used for transformers as Autoregressive beginning
        :param init_sample: (batch_size, node_num, lag) the initial state at the beginning of the simulation
        :param init_sample_norm: (batch_size, node_num, lag) the normalized initial state at the beginning of the simulation
        :param scaler_info: dict, {'scale_list': scale_list, 'mean_list': mean_list, 'scale_list_work_condition': scale_list_work_condition, 'mean_list_work_condition': mean_list_work_condition}
        :param PI_info_batch: (batch_size, node_num, lag), PI_info_batch is the PI information, simluation data

        return: H_norm: (batch_size, node_num, lag)   normalized
        """
        # scaler to tensor
        # scale_tensor = torch.tensor(scaler_info['scale_list'], dtype=X.dtype, device=X.device).unsqueeze(0).unsqueeze(2)
        scale_tensor = scaler_info['scale_list'].unsqueeze(0).unsqueeze(2)
        'scale_tensor: (1, node_num, 1)'
        # mean_tensor = torch.tensor(scaler_info['mean_list'], dtype=X.dtype, device=X.device).unsqueeze(0).unsqueeze(2)
        mean_tensor = scaler_info['mean_list'].unsqueeze(0).unsqueeze(2)
        'mean_tensor: (1, node_num, 1)'

        # SPS_Model_PINN/SPS_Model_NN/SPS_Model_Phy/SPS_Model_Opt
        if self.args.model_selection == 'SPS_Model_PINN':
            H_norm = self.SPS_Model(A, X, X_norm, WC, WC_norm, T, init_sample, init_sample_norm, scaler_info, PI_info_batch)
            'H_norm: (batch_size, node_num, lag)   normalized'

        elif self.args.model_selection == 'SPS_Model_NN':
            H_norm = self.SPS_Model(A, X_norm, WC_norm, T, T_of_y, Y)
            'H_norm: (batch_size, node_num, lag)   normalized'

        elif self.args.model_selection == 'SPS_Model_Phy':
            if not self.SPS_Model.init_sample_mark:
                self.H_last = init_sample.data
            # if self.H_last.size(0) != X.size(0):
            #     self.H_last = self.H_last[:X.size(0), :, :]
            #     # 因为在数据导入时，如果末尾凑不够一个batch_size，就会丢弃，所以这里也要丢弃
            #     # 这样也不行，训练集裁成不完整的传入验证集又会连不起来而报错，根源错误不在这里
            self.SPS_Model.set_init_value(self.H_last)
            H_list = self.SPS_Model(S_irr_SA=WC[:, 0:1, :], T_SA=WC[:, 1:2, :], theta=WC[:, 2:3, :], Load_Signal=WC[:, 3:, :])
            H = torch.cat(H_list, dim=1)
            'H: (batch_size, node_num, lag)   not normalized'
            # self.H_last = H.data
            self.H_last = H.detach()
            'self.H_last: (batch_size, node_num, lag)   not normalized'
            H_norm = (H - mean_tensor) / scale_tensor
            'H_norm: (batch_size, node_num, lag)   normalized'

        elif self.args.model_selection == 'SPS_Model_Opt':
            if self.H_last is None:
                self.H_last = init_sample.data
            # if self.H_last.size(0) != X.size(0):
            #     self.H_last = self.H_last[:X.size(0), :, :]
            #     # 因为在数据导入时，如果末尾凑不够一个batch_size，就会丢弃，所以这里也要丢弃
            #     # 这样也不行，训练集裁成不完整的传入验证集又会连不起来而报错，根源错误不在这里
            self.SPS_Model.set_init_value(self.H_last)
            H_list = self.SPS_Model(S_irr_SA=WC[:, 0:1, :], T_SA=WC[:, 1:2, :], theta=WC[:, 2:3, :], Load_Signal=WC[:, 3:, :])
            H = torch.cat(H_list, dim=1)
            'H: (batch_size, node_num, lag)   not normalized'
            # self.H_last = H.data
            self.H_last = H.detach()
            'self.H_last: (batch_size, node_num, lag)   not normalized'
            H_norm = (H - mean_tensor) / scale_tensor
            'H_norm: (batch_size, node_num, lag)   normalized'

        else:
            raise Exception("No such model_selection! must be 'SPS_Model_PINN' or 'SPS_Model_NN' or 'SPS_Model_Phy' or 'SPS_Model_Opt'!")
        return H_norm




























