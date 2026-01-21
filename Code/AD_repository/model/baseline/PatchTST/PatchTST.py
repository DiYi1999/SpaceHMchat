# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from types import SimpleNamespace
from AD_repository.model.baseline.layers.PatchTST_backbone import PatchTST_backbone
from AD_repository.model.baseline.layers.PatchTST_layers import series_decomp



class PatchTST_4ours(nn.Module):
    """
    PatchTST: A Transformer Model for Time Series Forecasting
    """
    def __init__(self, args):
        super(PatchTST_4ours, self).__init__()
        config = {}
        config['enc_in'] = args.sensor_num  # 输入通道数
        config['seq_len'] = args.lag  # 上下文窗口大小
        config['pred_len'] = args.pred_len if args.BaseOn == "forecast" else args.lag  # 目标窗口大小
        if (config['pred_len'] != args.lag) and (args.spatial_block != 'Nothing'):
            raise Exception("When using spatial block, the pred_len must be equal to lag!"
                            "Or you can change this code remove Temporal moudle after Spatial moudle,"
                            "Or you can change the code in the SPS_Model_NN.py: cat() T and X after clipping T")
        config['e_layers'] = 3  # 编码器层数
        config['d_model'] = 128  # 模型维度
        config['d_ff'] = 256  # 前馈网络维度
        config['n_heads'] = 16  # 注意力头数
        config['dropout'] =args.dropout  # dropout概率
        config['fc_dropout'] = args.dropout  # 全连接层dropout概率
        config['head_dropout'] = args.dropout  # 头dropout概率
        config['individual'] = 0  # 是否独立处理每个变量
        config['patch_len'] = 16  # patch长度
        config['stride'] = 8  # patch步长
        config['padding_patch'] = 'end'  # patch填充方式
        config['revin'] = 1  # 是否使用RevIN
        config['affine'] = 0  # 是否使用仿射变换
        config['subtract_last'] = 0  # 是否减去最后一个值
        config['decomposition'] = 0 # 是否使用分解
        config['kernel_size'] = 25  # 分解核大小

        config = SimpleNamespace(**config)
        self.PatchTST = Model(config)
        # (self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None,
        # d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0.,
        # act:str="gelu", key_padding_mask:bool='auto', padding_var:Optional[int]=None,
        # attn_mask:Optional[Tensor]=None, res_attention:bool=True,
        # pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True,
        # pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, ** kwargs)

    def forward(self, X):
        """

        Args:
            X: (batch_size, sensor_num, lag)

        Returns:
            out: (batch_size, sensor_num, lag/pred_len)

        """
        X = X.permute(0, 2, 1).contiguous()
        # (batch_size, time_step, node_cnt)  /  # (batch_size, lag, sensor_num)
        out = self.PatchTST(X)
        # out: (batch_size, lag/pred_len, sensor_num)
        out = out.permute(0, 2, 1).contiguous()
        # out: (batch_size, sensor_num, lag/pred_len)

        return out





class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None,
                 d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0.,
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None,
                 attn_mask:Optional[Tensor]=None, res_attention:bool=True,
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True,
                 pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters
        c_in = configs.enc_in  # 输入通道数
        context_window = configs.seq_len  # 上下文窗口大小
        target_window = configs.pred_len  # 目标窗口大小
        
        n_layers = configs.e_layers  # 编码器层数
        n_heads = configs.n_heads  # 注意力头数
        d_model = configs.d_model  # 模型维度
        d_ff = configs.d_ff  # 前馈网络维度
        dropout = configs.dropout  # dropout概率
        fc_dropout = configs.fc_dropout  # 全连接层dropout概率
        head_dropout = configs.head_dropout  # 头dropout概率
        
        individual = configs.individual  # 是否独立处理每个变量
    
        patch_len = configs.patch_len  # patch长度
        stride = configs.stride  # patch步长
        padding_patch = configs.padding_patch  # patch填充方式
        
        revin = configs.revin  # 是否使用RevIN
        affine = configs.affine  # 是否使用仿射变换
        subtract_last = configs.subtract_last  # 是否减去最后一个值
        
        decomposition = configs.decomposition  # 是否使用分解
        kernel_size = configs.kernel_size  # 分解核大小
        
        
        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)  # 初始化分解模块
            self.model_trend = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)  # 初始化趋势模型
            self.model_res = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)  # 初始化残差模型
        else:
            self.model = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)  # 初始化模型
    
    
    def forward(self, x):           # x: [Batch, Input length, Channel]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)  # 分解输入x为残差和趋势
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)  # 处理残差部分
            trend = self.model_trend(trend_init)  # 处理趋势部分
            x = res + trend  # 合并残差和趋势
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x = self.model(x)  # 处理输入
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        return x