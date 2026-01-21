import torch
import torch.nn as nn
import torch.nn.functional as F
from AD_repository.model.baseline.layers.masking import TriangularCausalMask, ProbMask
from AD_repository.model.baseline.layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from AD_repository.model.baseline.layers.SelfAttention_Family import FullAttention, ProbAttention, AttentionLayer
from AD_repository.model.baseline.layers.Embed import DataEmbedding,DataEmbedding_wo_pos,DataEmbedding_wo_temp,DataEmbedding_wo_pos_temp
import numpy as np
from types import SimpleNamespace

"https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/models/Informer.py"



class Informer_4ours(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.if_timestamp = args.if_timestamp
        self.timestamp_dim = args.timestamp_dim
        self.label_len = args.label_len
        self.pred_len = args.pred_len if args.BaseOn == "forecast" else args.lag

        config = {}
        config['seq_len'] = args.lag # 输入序列长度，当前输入序列长度（可自定义）,计算attention时每个batch以96行作为最小单位
        config['label_len'] = args.label_len # 标签长度，标签（带预测值的那个东西）长度（可自定义），有标签预测序列长度,label_len小于seq_len
        config['pred_len'] = args.pred_len if args.BaseOn == "forecast" else args.lag  # 目标窗口大小，预测未来序列长度 （可自定义）,预测未来多少个时间点的数据，无标签预测序列长度，通过前label_len个真实值辅助decoder进行预测pred_len个预测值
        config['output_attention'] = False # 是否输出注意力权重
        # config['moving_avg'] = 25 # 移动平均窗口大小
        config['moving_avg'] = args.preMA_win # 移动平均窗口大小，事实上这个也没用到，忘记了Informer要这个干嘛，反正和我预处理的preMA_win保持一致了

        # config['enc_in'] = args.sensor_num # 输入特征维度，解码器输入维度，你数据有多少列,要减去时间那一列
        # config['dec_in'] = args.sensor_num # 解码器输入特征维度，编码器输入维度，你数据有多少列,要减去时间那一列
        # config['c_out'] = args.sensor_num # 输出特征维度，输出预测多少维度，如果features填写的是M那么和上面就一样，如果填写的MS那么这里要输入1因为你的输出只有一列数据
        config['enc_in'] = args.node_num # 输入特征维度，解码器输入维度，你数据有多少列,要减去时间那一列
        config['dec_in'] = args.node_num # 解码器输入特征维度，编码器输入维度，你数据有多少列,要减去时间那一列
        config['c_out'] = args.node_num # 输出特征维度，输出预测多少维度，如果features填写的是M那么和上面就一样，如果填写的MS那么这里要输入1因为你的输出只有一列数据

        config['d_model'] = args.Informer_dim_model # 默认512，模型维度，隐层特征，enc和dec输出维度，数据中列数不能大于该值，必须是偶数
        config['embed'] = 'timeF' # 嵌入方式
        config['embed_type'] = 0 # 嵌入类型
        # freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        freq_map = {4: 'h', 5: 't', 6: 's', 1: 'm', 2: 'w', 3: 'd', 0: 'a', -1: 'b'}
        config['freq'] = freq_map[args.timestamp_dim]
        # config['freq'] = 't' # 时间频率,‘h’就是说T是4列，‘t’就是T是5列
        config['dropout'] = args.dropout # dropout率

        config['factor'] = args.Informer_factor # attn factor 注意力KQV，对Q进行采样，对Q采样的因子数
        config['n_heads'] = args.Informer_num_heads # 注意力头数，多头注意力机制，头越多注意力越好
        config['e_layers'] = args.Informer_enc_layers # 编码器层数,堆叠几层enc
        config['d_layers'] = args.Informer_dec_layers # 解码器层数,堆叠几层dec
        config['d_ff'] = args.Informer_d_ff # 全连接层（多层感知机）输出维度
        config['activation'] = 'gelu' # 激活函数

        config['distil'] = args.Informer_distil # whether to use distilling in encoder, using this argument means not using distilling

        config = SimpleNamespace(**config)
        self.Informer = Model(config)

    def forward(self, X, T, T_of_y, Y):
        """
        https://github.com/thuml/iTransformer/blob/main/experiments/exp_long_term_forecasting.py
        https://github.com/thuml/Autoformer/blob/main/exp/exp_main.py
        https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/data_provider/data_loader.py
        https://github.com/thuml/iTransformer/blob/main/model/Transformer.py

        Args:
            X: (batch_size, sensor_num, lag)
            T: (batch_size, 4, lag)
            T_of_y: (batch_size, 4, label_len+pred_len / lag) 就是标签Y对应的T，给Transformer那些自回归模型的Decoder生成用的一个自回归开头
            Y: (batch_size, sensor_num, label_len+pred_len / lag) 就是标签Y，给Transformer那些自回归模型的Decoder生成用的一个自回归开头
        Returns: (batch_size, sensor_num, pred_len)
        """
        # if self.if_timestamp:
        #     X = X[:, :-self.timestamp_dim, :]  # 去掉时间戳列
        # X: (batch_size, node_num, lag) --> (batch_size, sensor_num, lag)

        "制作Encoder输入X"
        x_enc = X.permute(0, 2, 1).contiguous()
        # x_enc就是elf.data_x[s_begin:s_end] (batch_size, sensor_num, lag) -> (batch_size, lag, sensor_num)

        "制作Encoder输入T"
        x_mark_enc = T.permute(0, 2, 1).contiguous()
        # x_mark_enc其实就是self.data_stamp[s_begin:s_end]  (batch_size, sensor_num, lag) -> (batch_size, lag, sensor_num)

        "制作Decoder输入 回归生成头X/Y"
        if Y.shape[1] != X.shape[1]: Y = torch.cat([Y, T_of_y], dim=1)
        # 如果Y的列数和X的列数不一样，就把T_of_y拼接到Y后面sensor_num-->node_num，(batch_size, sensor_num, label_len+pred_len) -> (batch_size, node_num, label_len+pred_len)
        Y = Y.permute(0, 2, 1).contiguous()
        # Y就是self.data_y[r_begin:r_end] (batch_size, sensor_num, label_len+pred_len) -> (batch_size, label_len+pred_len, sensor_num)
        x_dec = torch.zeros_like(Y[:, -self.pred_len:, :]).float()
        x_dec = torch.cat([Y[:, :self.label_len, :], x_dec], dim=1).float().to(X.device)
        # x_dec其实就是self.data_y[r_begin:r_end]: (batch_size, label_len+pred_len, sensor_num)

        "制作Decoder输入 回归生成头T"
        T_of_y = T_of_y.permute(0, 2, 1).contiguous()
        # T_of_y其实就是self.data_stamp[r_begin:r_end] (batch_size, sensor_num, label_len+pred_len) -> (batch_size, label_len+pred_len, sensor_num)
        x_mark_dec = T_of_y
        # x_mark_dec其实就是self.data_stamp[r_begin:r_end]: (batch_size, label_len+pred_len, sensor_num)

        "开始Encoder和Decoder"
        x = self.Informer(x_enc, x_mark_enc, x_dec, x_mark_dec)
        # x: (B, L, N)
        x = x.permute(0, 2, 1).contiguous()
        # (B, L, N) -> (batch_size, sensor_num, lag/pred_len)
        # if self.if_timestamp:
        #     x = torch.cat([x, T[:, :, -x.shape[2]:]], dim=1)
        # # x: (batch_size, sensor_num, lag/pred_len) --> (batch_size, node_num, lag/pred_len)

        return x




class Model(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        if configs.embed_type == 0:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                            configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        elif configs.embed_type == 1:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 2:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        elif configs.embed_type == 3:
            self.enc_embedding = DataEmbedding_wo_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 4:
            self.enc_embedding = DataEmbedding_wo_pos_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [
                ConvLayer(
                    configs.d_model
                ) for l in range(configs.e_layers - 1)
            ] if configs.distil else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
