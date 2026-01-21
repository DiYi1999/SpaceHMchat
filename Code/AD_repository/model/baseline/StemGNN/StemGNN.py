import torch
import torch.nn as nn
import torch.nn.functional as F

"""https://github.com/Siew-soon/StemGNN/blob/main/main.py"""



class StemGNN_4ours(nn.Module):
    def __init__(self, args):
        """
        Args:
            args
        """
        super().__init__()
        horizon = args.pred_len if args.BaseOn == "forecast" else args.lag
        self.StemGNN = Model(# GRU的隐藏层特征维度
                             # units=256,
                             units=args.node_num,
                             stack_cnt=2,
                                time_step=args.lag,
                             multi_layer=5,
                                horizon=horizon,
                                dropout_rate=args.dropout,
                                leaky_rate=args.LeakyReLU_slope)

    def forward(self, X, A=None):
        """

        Args:
            X: (batch_size, node_num, lag)

        Returns:
            out: (batch_size, node_num, lag/pred_len)

        """
        X = X.permute(0, 2, 1).contiguous()
        # (batch_size, time_step, node_cnt)  /  # (batch_size, lag, node_num)
        out, _ = self.StemGNN(X)
        # out: (batch_size, lag/pred_len, node_num)
        out = out.permute(0, 2, 1).contiguous()
        # out: (batch_size, node_num, lag/pred_len)

        return out




# GLU模块：门控线性单元，用于特征选择
class GLU(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(GLU, self).__init__()
        self.linear_left = nn.Linear(input_channel, output_channel)  # 全连接层，输入维度为input_channel，输出维度为output_channel
        self.linear_right = nn.Linear(input_channel, output_channel)  # 同上

    def forward(self, x):
        # x: (batch_size, input_channel)
        return torch.mul(self.linear_left(x), torch.sigmoid(self.linear_right(x)))  
        # 输出维度: (batch_size, output_channel)

# StockBlockLayer模块：用于时间序列预测的核心模块
class StockBlockLayer(nn.Module):
    def __init__(self, time_step, unit, multi_layer, stack_cnt=0):
        super(StockBlockLayer, self).__init__()
        self.time_step = time_step  # 时间步长
        self.unit = unit  # 单元数
        self.stack_cnt = stack_cnt  # 堆叠层数
        self.multi = multi_layer  # 多层因子
        self.weight = nn.Parameter(
            torch.Tensor(
                1, 3 + 1, 1, self.time_step * self.multi, self.multi * self.time_step
            )
        )  # 权重张量，维度: (1, 4, 1, time_step * multi, multi * time_step)
        nn.init.xavier_normal_(self.weight)  # 使用Xavier初始化权重
        self.forecast = nn.Linear(
            self.time_step * self.multi, self.time_step * self.multi
        )  # 预测层，输入输出维度均为(time_step * multi)
        self.forecast_result = nn.Linear(self.time_step * self.multi, self.time_step)  
        # 最终预测结果层，输入维度为(time_step * multi)，输出维度为time_step
        if self.stack_cnt == 0:
            self.backcast = nn.Linear(self.time_step * self.multi, self.time_step)  
            # 反向预测层，仅在stack_cnt为0时定义
            self.backcast_short_cut = nn.Linear(self.time_step, self.time_step)
            # 短路连接层，输入输出维度均为time_step
        self.relu = nn.ReLU()  # 激活函数
        self.GLUs = nn.ModuleList()  # GLU模块列表
        self.output_channel = 4 * self.multi  # 输出通道数

        for i in range(3):  # 定义3组GLU模块
            if i == 0:
                self.GLUs.append(
                    GLU(self.time_step * 4, self.time_step * self.output_channel)
                )  # 第一组GLU，输入维度为(time_step * 4)，输出维度为(time_step * output_channel)
                self.GLUs.append(
                    GLU(self.time_step * 4, self.time_step * self.output_channel)
                )  # 同上
            elif i == 1:
                self.GLUs.append(
                    GLU(
                        self.time_step * self.output_channel,
                        self.time_step * self.output_channel,
                    )
                )  # 第二组GLU，输入输出维度均为(time_step * output_channel)
                self.GLUs.append(
                    GLU(
                        self.time_step * self.output_channel,
                        self.time_step * self.output_channel,
                    )
                )  # 同上
            else:
                self.GLUs.append(
                    GLU(
                        self.time_step * self.output_channel,
                        self.time_step * self.output_channel,
                    )
                )  # 第三组GLU，输入输出维度均为(time_step * output_channel)
                self.GLUs.append(
                    GLU(
                        self.time_step * self.output_channel,
                        self.time_step * self.output_channel,
                    )
                )  # 同上

    def spe_seq_cell(self, input):
        """

        Args:
            input: # input: (batch_size, k, input_channel, node_cnt, time_step)

        Returns:

        """
        batch_size, k, input_channel, node_cnt, time_step = input.size()
        input = input.view(batch_size, -1, node_cnt, time_step)  
        # 重新调整维度: (batch_size, k * input_channel, node_cnt, time_step)
        ffted = torch.fft.fft(input, dim=-1)  
        # 对最后一个维度(time_step)进行傅里叶变换，输出维度与输入相同
        real = (
            ffted.real.permute(0, 2, 1, 3)
            .contiguous()
            .reshape(batch_size, node_cnt, -1)
        )  
        # 提取实部，调整维度: (batch_size, node_cnt, k * input_channel * time_step)
        img = (
            ffted.imag.permute(0, 2, 1, 3)
            .contiguous()
            .reshape(batch_size, node_cnt, -1)
        )  
        # 提取虚部，调整维度: (batch_size, node_cnt, k * input_channel * time_step)
        for i in range(3):  # 通过3组GLU模块处理实部和虚部
            real = self.GLUs[i * 2](real)  
            # 实部经过GLU处理，维度: (batch_size, node_cnt, time_step * output_channel)
            img = self.GLUs[2 * i + 1](img)  
            # 虚部经过GLU处理，维度: (batch_size, node_cnt, time_step * output_channel)
        real =real.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()
        # 调整实部维度: (batch_size, 4, node_cnt, time_step * output_channel / 4)
        img = img.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()  
        # 调整虚部维度: (batch_size, 4, node_cnt, time_step * output_channel / 4)
        time_step_as_inner = torch.cat([real.unsqueeze(-1), img.unsqueeze(-1)], dim=-1)  
        # 合并实部和虚部，维度: (batch_size, 4, node_cnt, time_step * output_channel / 4, 2)
        iffted = torch.fft.irfft(time_step_as_inner, 1)  
        # 对第1维进行逆傅里叶变换，输出维度与输入相同(batch_size, 4, node_cnt, time_step*output_channel/4, 2)
        return iffted

    def forward(self, x, mul_L):
        """

        Args:
            x: x: (batch_size, 1, node_cnt, time_step)
            mul_L: mul_L: (batch_size, k, node_cnt, node_cnt)

        Returns:
            forecast: 预测结果: (batch_size, node_cnt, time_step)
            backcast_source: 反向预测源: (batch_size, 1, node_cnt, time_step)

        """
        mul_L = mul_L.unsqueeze(1)
        # 扩展维度: (batch_size, 1, k, node_cnt, node_cnt)
        x = x.unsqueeze(1)
        # 扩展维度: (batch_size, 1, 1, node_cnt, time_step)
        gfted = torch.matmul(mul_L, x)
        # (batch_size, 1, k, node_cnt, time_step)
        gconv_input = self.spe_seq_cell(gfted)
        # gconv_input: (batch_size, 4, node_cnt, time_step*output_channel/4, 2)
        gconv_input = gconv_input.unsqueeze(2)
        # 扩展维度: (batch_size, 4, 1, node_cnt, time_step*output_channel/4, 2)
        gconv_input_permuted = gconv_input.permute(0, 1, 2, 3, 5, 4)
        # 调整维度顺序: (batch_size, 4, 1, node_cnt, 2, time_step*output_channel/4)

        # Reshape the tensor to move the dimension with size 60 to the last dimension
        gconv_input_reshaped = gconv_input_permuted.reshape(
            gconv_input.shape[:-2] + (-1,)
        )
        # gconv_input_reshaped: (batch_size, 4, 1, node_cnt, time_step*output_channel/4*2)

        # Perform matrix multiplication
        # self.weight: (1, 3 + 1, 1, self.time_step * self.multi, self.multi * self.time_step)
        "要求self.multi*self.time_step = time_step*output_channel/4*2" \
        "前面有：self.output_channel = 4 * self.multi  # 输出通道数"
        igfted = torch.matmul(gconv_input_reshaped, self.weight)
        # igfted: (batch_size, 4, 1, node_cnt, self.multi * self.time_step)

        igfted = torch.sum(igfted, dim=1)
        # 对第1维求和: (batch_size, 1, node_cnt, self.multi * self.time_step)

        forecast_source = torch.sigmoid(self.forecast(igfted).squeeze(1))
        # 预测源: (batch_size, node_cnt, self.multi * self.time_step)
        forecast = self.forecast_result(forecast_source)
        # 预测结果: (batch_size, node_cnt, time_step)
        if self.stack_cnt == 0:
            backcast_short = self.backcast_short_cut(x).squeeze(1)
            # 短路连接: (batch_size, 1, node_cnt, time_step)
            backcast_source = torch.sigmoid(self.backcast(igfted) - backcast_short)
            # 反向预测源: (batch_size, 1, node_cnt, time_step)
        else:
            backcast_source = None
        return forecast, backcast_source


class Model(nn.Module):
    def __init__(
        self,
        units,
        stack_cnt,
        time_step,
        multi_layer,
        horizon=1,
        dropout_rate=0.5,
        leaky_rate=0.2,
        # device="cpu",
    ):
        """

        Args:
            units: GRU的隐藏层维度
            stack_cnt: StockBlockLayer的堆叠数目
            time_step: GRU的输入维度
            multi_layer: StockBlockLayer的多层因子， 5
            horizon: 预测长度
            dropout_rate:
            leaky_rate:
            device: 这个参数被我弃用了
        """
        super(Model, self).__init__()
        self.unit = units
        self.stack_cnt = stack_cnt
        self.unit = units
        self.alpha = leaky_rate
        self.time_step = time_step
        self.horizon = horizon
        self.weight_key = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)
        self.GRU = nn.GRU(self.time_step, self.unit)
        self.multi_layer = multi_layer
        self.stock_block = nn.ModuleList()
        self.stock_block.extend(
            [
                StockBlockLayer(
                    self.time_step, self.unit, self.multi_layer, stack_cnt=i
                )
                for i in range(self.stack_cnt)
            ]
        )
        self.fc = nn.Sequential(
            nn.Linear(int(self.time_step), int(self.time_step)),
            nn.LeakyReLU(),
            nn.Linear(int(self.time_step), self.horizon),
        )
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(p=dropout_rate)
        # self.to(device)

    def get_laplacian(self, graph, normalize):
        """
        return the laplacian of the graph.
        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L

    def cheb_polynomial(self, laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.
        :param laplacian: the graph laplacian, [N, N].
        :return: the multi order Chebyshev laplacian, [K, N, N].
        """
        N = laplacian.size(0)  # [N, N]
        laplacian = laplacian.unsqueeze(0)
        first_laplacian = torch.zeros(
            [1, N, N], device=laplacian.device, dtype=torch.float
        )
        second_laplacian = laplacian
        third_laplacian = (
            2 * torch.matmul(laplacian, second_laplacian)
        ) - first_laplacian
        forth_laplacian = (
            2 * torch.matmul(laplacian, third_laplacian) - second_laplacian
        )
        multi_order_laplacian = torch.cat(
            [first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=0
        )
        return multi_order_laplacian

    def latent_correlation_layer(self, x):
        # x: (batch_size, time_step, node_cnt)
        input, _ = self.GRU(x.permute(2, 0, 1).contiguous())
        # input: (node_cnt, batch_size, units)
        input = input.permute(1, 0, 2).contiguous()
        # input: (batch_size, node_cnt, units)
        attention = self.self_graph_attention(input)
        attention = torch.mean(attention, dim=0)
        degree = torch.sum(attention, dim=1)
        # laplacian is sym or not
        attention = 0.5 * (attention + attention.T)
        degree_l = torch.diag(degree)
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-7))
        laplacian = torch.matmul(
            diagonal_degree_hat, torch.matmul(degree_l - attention, diagonal_degree_hat)
        )
        mul_L = self.cheb_polynomial(laplacian)
        return mul_L, attention

    def self_graph_attention(self, input):
        # input: (batch_size, node_cnt, units)
        input = input.permute(0, 2, 1).contiguous()
        # input: (batch_size, units, node_cnt)
        bat, N, fea = input.size()
        key = torch.matmul(input, self.weight_key)   # self.weight_key: (units, 1)
        query = torch.matmul(input, self.weight_query)
        data = key.repeat(1, 1, N).view(bat, N * N, 1) + query.repeat(1, N, 1)
        data = data.squeeze(2)
        data = data.view(bat, N, -1)
        data = self.leakyrelu(data)
        attention = F.softmax(data, dim=2)
        attention = self.dropout(attention)
        return attention

    def graph_fft(self, input, eigenvectors):
        return torch.matmul(eigenvectors, input)

    def forward(self, x):
        """

        Args:
            x: (batch_size, time_step, node_cnt)

        Returns:
            output: (batch_size, horizon, node_cnt)
            attention: (batch_size, node_cnt, node_cnt)

        """
        mul_L, attention = self.latent_correlation_layer(x)
        # mul_L: (K, N, N)
        # attention: (batch_size, node_cnt, node_cnt)
        X = x.unsqueeze(1).permute(0, 1, 3, 2).contiguous()
        # X: (batch_size, 1, node_cnt, time_step)
        result = []
        for stack_i in range(self.stack_cnt):
            forecast, X = self.stock_block[stack_i](X, mul_L)
            # forecast: (batch_size, node_cnt, time_step)
            # X: (batch_size, 1, node_cnt, time_step)
            result.append(forecast)
        forecast = result[0] + result[1]
        # forecast: (batch_size, node_cnt, time_step)
        forecast = self.fc(forecast)
        # forecast: (batch_size, node_cnt, horizon)
        if forecast.size()[-1] == 1:
            output = forecast.unsqueeze(1).squeeze(-1)
            # output: (batch_size, 1, node_cnt)
        else:
            output = forecast.permute(0, 2, 1).contiguous()
            # output: (batch_size, horizon, node_cnt)
        return output, attention

