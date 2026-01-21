import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import numpy as np



class MLP_dim3(nn.Module):
    def __init__(self, in_dim, hidden_dim, layer_num, out_dim, dropout=0.0, LeakyReLU_slope=0.01):
        """
        MLP，输入维度是in_dim，输出维度是out_dim, 隐藏层维度是hidden_dim, 隐藏层数是layer_num
             dropout是dropout概率，LeakyReLU_slope是LeakyReLU的负半轴斜率

        Args:
            in_dim:
            hidden_dim:
            layer_num:
            out_dim:
            dropout:
            LeakyReLU_slope:
        """
        super(MLP_dim3, self).__init__()
        self.layers = nn.ModuleList()

        # 添加隐藏层
        for i in range(layer_num):
            if i == 0:  # 第一层，接收输入维度
                self.layers.append(nn.Linear(in_dim, hidden_dim))
                self.layers.append(nn.Dropout(p=dropout))
                self.layers.append(nn.LeakyReLU(negative_slope=LeakyReLU_slope))
            else:  # 其他层，接收隐藏层维度
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.layers.append(nn.Dropout(p=dropout))
                self.layers.append(nn.LeakyReLU(negative_slope=LeakyReLU_slope))
        # 添加输出层
        self.layers.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, X):
        """

        Args:
            X: (batch_size, node_num, in_dim)

        Returns:
            X: (batch_size, node_num, out_dim)

        """
        'X: (batch_size, node_num, in_dim)'
        for layer in self.layers:
            X = layer(X)
        'X: (batch_size, node_num, out_dim)'

        return X



class MLP_dim2(nn.Module):
    def __init__(self, in_dim, hidden_dim, layer_num, out_dim, dropout=0.0, LeakyReLU_slope=0.01):
        """
        MLP，输入维度是in_dim，输出维度是out_dim, 隐藏层维度是hidden_dim, 隐藏层数是layer_num
             dropout是dropout概率，LeakyReLU_slope是LeakyReLU的负半轴斜率

        Args:
            in_dim:
            hidden_dim:
            layer_num:
            out_dim:
            dropout:
            LeakyReLU_slope:
        """
        super(MLP_dim2, self).__init__()
        self.layers = nn.ModuleList()

        # 添加隐藏层
        for i in range(layer_num):
            if i == 0:  # 第一层，接收输入维度
                self.layers.append(nn.Linear(in_dim, hidden_dim))
                self.layers.append(nn.Dropout(p=dropout))
                self.layers.append(nn.LeakyReLU(negative_slope=LeakyReLU_slope))
            else:  # 其他层，接收隐藏层维度
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.layers.append(nn.Dropout(p=dropout))
                self.layers.append(nn.LeakyReLU(negative_slope=LeakyReLU_slope))
        # 添加输出层
        self.layers.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, X):
        """

        Args:
            X: (batch_size, in_node_num, lag)

        Returns:
            X: (batch_size, out_node_num, lag)

        """
        'X: (batch_size, in_node_num, lag)'
        X = X.permute(0, 2, 1)
        'X: (batch_size, lag, in_node_num)'
        for layer in self.layers:
            X = layer(X)
        'X: (batch_size, lag, out_node_num)'
        X = X.permute(0, 2, 1)
        'X: (batch_size, out_node_num, lag)'

        return X



### TCN
class Chomp1d(nn.Module):
    """
    这个函数是用来修剪卷积之后的数据的尺寸，让其与输入数据尺寸相同。https://blog.csdn.net/jasminefeng/article/details/117671964
    因为TCN用nn.Conv1d时候为保证输出还是lag而不缩水进行了俩端的填充，这个函数将右端填充裁掉 https://juejin.cn/post/7262269863343079479
    也就是说后续的截取pred_len得截取后面而不是前面，不然截一堆0
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()



class TCNBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, padding_mode='zeros', dropout=0.2):
        super(TCNBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride,
                                           padding=padding, padding_mode=padding_mode, dilation=dilation))
        ", padding_mode='replicate'  默认0填充"
        # https://blog.51cto.com/u_15473842/4882110
        # https://juejin.cn/post/7262269863343079479
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride,
                                           padding=padding, padding_mode=padding_mode, dilation=dilation))
        ", padding_mode='replicate'  默认0填充"
        # https://blog.51cto.com/u_15473842/4882110
        # https://juejin.cn/post/7262269863343079479
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        'https://juejin.cn/post/7262269863343079479  ' \
        '注意因为TCN用nn.Conv1d时候为保证输出还是lag而不缩水进行了 左右俩端 的填充，设置了padding和padding_mode=zeros' \
        '又通过self.chomp这个函数将右端填充裁掉 也就是说后续的截取pred_len得截取后面而不是前面，不然截一堆0'

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)



class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.1):
        """
        利用TCN完成时间信息提取，输入x (batch, node_num, lag)，输出out (batch, TCN_layers_channels[-1], lag)。
        An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling
        Shaojie Bai, J. Zico Kolter, Vladlen Koltun
        https://arxiv.org/abs/1803.01271
        https://zhuanlan.zhihu.com/p/584620088
        默认膨胀系数是按照TCNBlock的层数（也就是num_channels列表长度）来进行2的次方，就如https://mmbiz.qpic.cn/mmbiz_png/8LIHzsJ61ObaxHAlBUvES5kCRug7H3PKMMeMuv8cwUy5Gib2TsDMELUOIiayEB9sgULSQYxV0jsjTafibOrUewtQg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1.

        Args:
            num_inputs:输入数据的通道数，比如MSL是27.
            num_channels:这是一个列表，它同时定义了TCN的层数 和 每层通道数变化结果，比如[24, 8, 1]，就是有3层TCNBlock，每进行一次TCNBlock通道数从原来的27依次变化为24变8变1。
            kernel_size:膨胀卷积核长度
            dropout:随机丢弃，默认0.1
        """
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            """https://mmbiz.qpic.cn/mmbiz_jpg/xhSzPEfYia98LFCydbXXChNQgT92dOKFiaXfKuINK4dqZQx6dFic82IUqdrVbom58vuQrCOLbbsODruwQQbKnrAuQ/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1"""
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TCNBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        out = self.network(x)
        # (batch, node_num, lag) -> (batch, TCN_layers_channels[-1], lag)

        return out



### GRU
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.1):
        """
        GRU完成时间信息提取，输入x (batch, node_num, lag)，输出out (batch, hidden_size, lag)

        Args:
            input_size:输入数据的通道数，比如MSL是27.
            hidden_size:GRU的隐藏层通道数
            num_layers:GRU的层数
            dropout:随机丢弃，默认0.2
        """
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          dropout=dropout,
                          batch_first=True,
                          # bidirectional=True
                          )
        #### batch_first-如果 True ，则输入和输出张量提供为 (batch, seq, feature) 而不是 (seq, batch, feature) 。
        #### 请注意，这不适用于隐藏（第二个输出）或单元状态。记得确认和更改。默认值：False

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # (batch, node_num, lag) -> (batch, lag, node_num)

        # Forward propagate RNN
        out, _ = self.gru(x)
        # (batch, lag, hidden_size)
        out = out.permute(0, 2, 1)
        # (batch, lag, hidden_size) -> (batch, hidden_size, lag)

        return out



class Do_Nothing(nn.Module):
    def __init__(self):
        super(Do_Nothing, self).__init__()

    def forward(self, x):
        return x


















