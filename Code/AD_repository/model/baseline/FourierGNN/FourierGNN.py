import torch
import torch.nn as nn
import torch.nn.functional as F



class FourierGNN_4ours(nn.Module):
    def __init__(self, args):
        """

        Args:
            args

        forward: input (batch_size, node_num, lag) -> output (batch_size, node_num, lag/pred_len)

        """
        super().__init__()
        out_dim = args.pred_len if args.BaseOn == "forecast" else args.lag
        self.FGN = FGN(pre_length=out_dim,
                       embed_size=256,
                        feature_size=args.node_num,
                        seq_length=args.lag,
                        hidden_size=512,
                        hard_thresholding_fraction=1,
                        hidden_size_factor=1,
                        sparsity_threshold=0.01)

    def forward(self, x, A=None):
        """
        Args:
            x: (batch_size, node_num, lag)

        Returns: (batch_size, node_num, lag/pred_len)

        """
        x = x.permute(0, 2, 1).contiguous()
        # (batch_size, node_num, lag) -> (B, L, N)
        x = self.FGN(x)
        # x: (B, L, N) -> (batch_size, node_num, lag/pred_len)
        return x



class FGN(nn.Module):
    def __init__(self, pre_length, embed_size,
                 feature_size, seq_length, hidden_size, hard_thresholding_fraction=1,
                 hidden_size_factor=1, sparsity_threshold=0.01):
        """

        Args:
            pre_length: 输出预测长度
            embed_size: 嵌入维度
            feature_size: 这参数没啥用
            seq_length: 就是输入x的第三个维度lag
            hidden_size: 隐藏层维度
            hard_thresholding_fraction: 硬阈值比例
            hidden_size_factor: 隐藏层维度因子
            sparsity_threshold: 稀疏性阈值

        forward: input (B, L, N) -> output (B, N, pre_length)

        """
        super().__init__()
        self.embed_size = embed_size  # 嵌入维度
        self.hidden_size = hidden_size  # 隐藏层维度
        self.number_frequency = 1  # 频率数量
        self.pre_length = pre_length  # 输出预测长度
        self.feature_size = feature_size  # 特征维度
        self.seq_length = seq_length  # 序列长度
        self.frequency_size = self.embed_size // self.number_frequency  # 每个频率的嵌入维度
        self.hidden_size_factor = hidden_size_factor  # 隐藏层维度因子
        self.sparsity_threshold = sparsity_threshold  # 稀疏性阈值
        self.hard_thresholding_fraction = hard_thresholding_fraction  # 硬阈值比例
        self.scale = 0.02  # 参数初始化缩放因子
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))  # 嵌入参数，维度: (1, embed_size)

        # 定义权重和偏置参数
        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size, self.frequency_size * self.hidden_size_factor))
        # 维度: (2, frequency_size, frequency_size * hidden_size_factor)
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        # 维度: (2, frequency_size * hidden_size_factor)
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor, self.frequency_size))
        # 维度: (2, frequency_size * hidden_size_factor, frequency_size)
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size))
        # 维度: (2, frequency_size)
        self.w3 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size,
                                     self.frequency_size * self.hidden_size_factor))
        # 维度: (2, frequency_size, frequency_size * hidden_size_factor)
        self.b3 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        # 维度: (2, frequency_size * hidden_size_factor)
        self.embeddings_10 = nn.Parameter(torch.randn(self.seq_length, 8))
        # 嵌入参数，维度: (seq_length, 8)
        self.fc = nn.Sequential(
            nn.Linear(self.embed_size * 8, 64),
            # 全连接层，输入维度: (embed_size * 8)，输出维度: (64)
            nn.LeakyReLU(),
            nn.Linear(64, self.hidden_size),
            # 全连接层，输入维度: (64)，输出维度: (hidden_size)
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length)
            # 全连接层，输入维度: (hidden_size)，输出维度: (pre_length)
        )
        # self.to('cuda:0')
        # # 将模型移动到 GPU 上

    def tokenEmb(self, x):
        x = x.unsqueeze(2)  # 增加一个维度，x: (B, NL) -> (B, NL, 1)
        y = self.embeddings  # 嵌入参数，y: (1, embed_size)
        return x * y  # 广播相乘，输出维度: (B, NL, embed_size)

    # FourierGNN
    def fourierGC(self, x, B, N, L):
        o1_real = torch.zeros([B, (N*L)//2 + 1, self.frequency_size * self.hidden_size_factor],
                              device=x.device)  # 初始化实部张量，维度: (B, (N*L)//2 + 1, frequency_size * hidden_size_factor)
        o1_imag = torch.zeros([B, (N*L)//2 + 1, self.frequency_size * self.hidden_size_factor],
                              device=x.device)  # 初始化虚部张量，维度: (B, (N*L)//2 + 1, frequency_size * hidden_size_factor)
        o2_real = torch.zeros(x.shape, device=x.device)  # 初始化实部张量，维度与输入 x 相同
        o2_imag = torch.zeros(x.shape, device=x.device)  # 初始化虚部张量，维度与输入 x 相同

        o3_real = torch.zeros(x.shape, device=x.device)  # 初始化实部张量，维度与输入 x 相同
        o3_imag = torch.zeros(x.shape, device=x.device)  # 初始化虚部张量，维度与输入 x 相同

        o1_real = F.relu(
            torch.einsum('bli,ii->bli', x.real, self.w1[0]) - \
            torch.einsum('bli,ii->bli', x.imag, self.w1[1]) + \
            self.b1[0]
        )  # 计算第一层实部，维度: (B, (N*L)//2 + 1, frequency_size * hidden_size_factor)

        o1_imag = F.relu(
            torch.einsum('bli,ii->bli', x.imag, self.w1[0]) + \
            torch.einsum('bli,ii->bli', x.real, self.w1[1]) + \
            self.b1[1]
        )  # 计算第一层虚部，维度: (B, (N*L)//2 + 1, frequency_size * hidden_size_factor)

        # 1 layer
        y = torch.stack([o1_real, o1_imag], dim=-1)  # 合并实部和虚部，维度: (B, (N*L)//2 + 1, frequency_size * hidden_size_factor, 2)
        y = F.softshrink(y, lambd=self.sparsity_threshold)  # 稀疏化操作，维度不变

        o2_real = F.relu(
            torch.einsum('bli,ii->bli', o1_real, self.w2[0]) - \
            torch.einsum('bli,ii->bli', o1_imag, self.w2[1]) + \
            self.b2[0]
        )  # 计算第二层实部，维度: (B, (N*L)//2 + 1, frequency_size)

        o2_imag = F.relu(
            torch.einsum('bli,ii->bli', o1_imag, self.w2[0]) + \
            torch.einsum('bli,ii->bli', o1_real, self.w2[1]) + \
            self.b2[1]
        )  # 计算第二层虚部，维度: (B, (N*L)//2 + 1, frequency_size)

        # 2 layer
        x = torch.stack([o2_real, o2_imag], dim=-1)  # 合并实部和虚部，维度: (B, (N*L)//2 + 1, frequency_size, 2)
        x = F.softshrink(x, lambd=self.sparsity_threshold)  # 稀疏化操作，维度不变
        x = x + y  # 残差连接，维度不变

        o3_real = F.relu(
                torch.einsum('bli,ii->bli', o2_real, self.w3[0]) - \
                torch.einsum('bli,ii->bli', o2_imag, self.w3[1]) + \
                self.b3[0]
        )  # 计算第三层实部，维度: (B, (N*L)//2 + 1, frequency_size * hidden_size_factor)

        o3_imag = F.relu(
                torch.einsum('bli,ii->bli', o2_imag, self.w3[0]) + \
                torch.einsum('bli,ii->bli', o2_real, self.w3[1]) + \
                self.b3[1]
        )  # 计算第三层虚部，维度: (B, (N*L)//2 + 1, frequency_size * hidden_size_factor)

        # 3 layer
        z = torch.stack([o3_real, o3_imag], dim=-1)  # 合并实部和虚部，维度: (B, (N*L)//2 + 1, frequency_size * hidden_size_factor, 2)
        z = F.softshrink(z, lambd=self.sparsity_threshold)  # 稀疏化操作，维度不变
        z = z + x  # 残差连接，维度不变
        z = torch.view_as_complex(z)  # 转换为复数形式，维度: (B, (N*L)//2 + 1, frequency_size * hidden_size_factor)
        return z

    def forward(self, x):
        """

        Args:
            x: (B, L, N)

        Returns: (B, N, pre_length)

        """
        x = x.permute(0, 2, 1).contiguous()
        # 调整维度顺序，x: (B, L, N) -> (B, N, L)
        B, N, L = x.shape  # 获取批量大小、节点数和序列长度
        x = x.reshape(B, -1)  # 展平节点和序列维度，x: (B, L, N) -> (B, NL)
        x = self.tokenEmb(x)  # 嵌入操作，x: (B, NL) -> (B, NL, embed_size)

        x = torch.fft.rfft(x, dim=1, norm='ortho')
        # 进行快速傅里叶变换，x: (B, NL, embed_size) -> (B, NL//2+1, embed_size)

        x = x.reshape(B, (N*L)//2+1, self.frequency_size)
        # 调整维度，x: (B, NL//2+1, embed_size) -> (B, (N*L)//2+1, frequency_size)

        bias = x
        # 保存偏置，维度: (B, (N*L)//2+1, frequency_size)

        x = self.fourierGC(x, B, N, L)
        # 通过 FourierGNN 模块，维度: (B, (N*L)//2+1, frequency_size)

        x = x + bias
        # 残差连接，维度: (B, (N*L)//2+1, frequency_size)

        x = x.reshape(B, (N*L)//2+1, self.embed_size)
        # 调整维度，x: (B, (N*L)//2+1, frequency_size) -> (B, (N*L)//2+1, embed_size)

        x = torch.fft.irfft(x, n=N*L, dim=1, norm="ortho")
        # 进行逆快速傅里叶变换，x: (B, (N*L)//2+1, embed_size) -> (B, NL, embed_size)

        x = x.reshape(B, N, L, self.embed_size)
        # 调整维度，x: (B, NL, embed_size) -> (B, N, L, embed_size)
        x = x.permute(0, 1, 3, 2)
        # 调整维度顺序，x: (B, N, L, embed_size) -> (B, N, embed_size, L)

        x = torch.matmul(x, self.embeddings_10)
        # 投影操作，x: (B, N, embed_size, L) -> (B, N, embed_size, 8)
        x = x.reshape(B, N, -1)
        # 展平最后两个维度，x: (B, N, embed_size, 8) -> (B, N, embed_size*8)
        x = self.fc(x)
        # 全连接层，x: (B, N, embed_size*8) -> (B, N, pre_length)

        return x  # 输出维度: (B, N, pre_length)
