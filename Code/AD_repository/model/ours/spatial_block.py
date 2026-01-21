import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F





class GCN_layer(nn.Module):
    def __init__(self, args, input_num, output_num):
        """
        AXW+b, 输入X: (batch_size, node_num, input_num), 输出H: (batch_size, node_num, output_num)

        Args:
            args:
            input_num:
            output_num:
        """
        super(GCN_layer, self).__init__()
        self.args = args
        self.W = Parameter(torch.Tensor(input_num, output_num))
        self.b = Parameter(torch.Tensor(args.node_num, 1))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.W)
        torch.nn.init.normal_(self.b)

    def forward(self, X, A):
        """
        :param X: (batch_size, node_num, input_num)
        :param A: (node_num, node_num)
        """
        H = torch.matmul(A, X)
        'H: (batch_size, node_num, input_num)'
        H = torch.matmul(H, self.W) + self.b
        'H: (batch_size, node_num, output_num)'
        H = F.leaky_relu(H, negative_slope=self.args.LeakyReLU_slope)
        'H: (batch_size, node_num, output_num)'
        return H, A


class GCN_s(nn.Module):
    def __init__(self, args):
        """
        叠多层AXW+b，输入X: (batch_size, node_num, lag), 输出H: (batch_size, node_num, lag)

        Args:
            args:
        """
        super(GCN_s, self).__init__()
        self.args = args

        GCN_list = []
        layer_num = len(args.GCN_layer_nums)
        for i in range(layer_num):
            if i == 0:
                GCN_list.append(GCN_layer(args, args.lag, args.GCN_layer_nums[i]))
            else:
                GCN_list.append(GCN_layer(args, args.GCN_layer_nums[i-1], args.GCN_layer_nums[i]))

        self.GCN_list = nn.ModuleList(GCN_list)

        if self.args.GCN_layer_nums[-1] != self.args.lag:
            self.fc = nn.Linear(args.GCN_layer_nums[-1], self.args.lag)
            self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)

    def forward(self, X, A):
        """
        :param X: (batch_size, node_num, lag)
        :param A: (node_num, node_num)
        """
        H = X
        'H: (batch_size, node_num, lag)'
        for GCN_layer in self.GCN_list:
            H, A = GCN_layer(H, A)
            'H: (batch_size, node_num, GCN_layer_nums[-1])'
        if self.args.GCN_layer_nums[-1] != self.args.lag:
            H = self.fc(H)
            'H: (batch_size, node_num, lag)'
            # H = F.leaky_relu(H, negative_slope=self.args.LeakyReLU_slope)
            # 'H: (batch_size*node_num, lag)'

        return H


class Nothing_to_do_S(nn.Module):
    def __init__(self):
        super(Nothing_to_do_S, self).__init__()

    def forward(self, X, A):
        return X


class FeatureAttentionLayer(nn.Module):
    """Single Graph Feature/Spatial Attention Layer from https://github.com/ML4ITS/mtad-gat-pytorch/blob/main/modules.py
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param dropout: percentage of nodes to dropout
    :param alpha: negative slope used in the leaky rely activation function
    :param embed_dim: embedding dimension (output dimension of linear transformation)
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param use_bias: whether to include a bias term in the attention layer
    """

    def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None, use_gatv2=True, use_bias=True):
        """
        GATv2和GAT, 输入X: (batch_size, node_num, lag), 输出H: (batch_size, node_num, lag)

        Args:
            n_features:
            window_size:
            dropout:
            alpha:
            embed_dim:
            use_gatv2:
            use_bias:
        """
        super(FeatureAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.embed_dim = embed_dim if embed_dim is not None else window_size
        self.use_gatv2 = use_gatv2
        self.num_nodes = n_features
        self.use_bias = use_bias

        # Because linear transformation is done after concatenation in GATv2
        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * window_size
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = window_size
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(n_features, n_features))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape (b, k, n): b - batch size, n - window size, k - number of features
        # For feature attention we represent a node as the values of a particular feature across all timestamps

        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf) Section 3.3
        # Linear transformation applied after concatenation and attention layer applied after leakyrelu
        if self.use_gatv2:
            a_input = self._make_attention_input(x)                 # (b, k, k, 2*window_size)
            a_input = self.leakyrelu(self.lin(a_input))             # (b, k, k, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)            # (b, k, k, 1)

        # Original GAT attention
        else:
            Wx = self.lin(x)                                                  # (b, k, k, embed_dim)
            a_input = self._make_attention_input(Wx)                          # (b, k, k, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)      # (b, k, k, 1)

        if self.use_bias:
            e += self.bias

        # Attention weights
        attention = torch.softmax(e, dim=2)                          # (b, k, k, 1)
        attention = torch.dropout(attention, self.dropout, train=self.training)

        # Computing new node features using the attention
        h = self.sigmoid(torch.matmul(attention, x))                 # (b, k, n)

        return h

    def _make_attention_input(self, v):
        """
        Preparing the feature attention mechanism.
        Creating matrix with all possible combinations of concatenations of node.
        Each node consists of all values of that node within the window
            v1 || v1,
            ...
            v1 || vK,
            v2 || v1,
            ...
            v2 || vK,
            ...
            ...
            vK || v1,
            ...
            vK || vK,

        Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf) Section 3.3
        """

        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)  # (b, K*K, 2*window_size)

        if self.use_gatv2:
            return combined.view(v.size(0), K, K, 2 * self.window_size)
        else:
            return combined.view(v.size(0), K, K, 2 * self.embed_dim)


class Muti_S_GAT(nn.Module):
    def __init__(self, args):
        """
        GATv2和GAT在空间维度的多头, 输入X: (batch_size, node_num, lag), 输出H: (batch_size, node_num, lag)

        Args:
            args:
        """
        super(Muti_S_GAT, self).__init__()
        self.args = args

        self.GAT_list = nn.ModuleList([FeatureAttentionLayer(n_features=args.node_num,
                                                             window_size=args.lag,
                                                             dropout=args.dropout,
                                                             alpha=args.LeakyReLU_slope,
                                                             embed_dim=args.S_GAT_embed_dim,
                                                             use_gatv2=args.use_gatv2,
                                                             use_bias=False)
                                       for _ in range(args.S_GAT_K)])

    def forward(self, X, A):
        """
        :param X: (batch_size, node_num, lag)
        :param A: (node_num, node_num)
        """
        H = [self.GAT_list[i](X) for i in range(self.args.S_GAT_K)]
        'H: list(K) of each(batch_size, node_num, lag)'
        # 此处使用取平均的方式，就不拼接后再线性变换了
        H = torch.stack(H, dim=0).mean(dim=0)
        'H: (batch_size, node_num, lag)'
        return H



class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, layer_num, out_dim, dropout=0.0, LeakyReLU_slope=0.01):
        """
        MLP，输入维度是E_in_dim，输出维度是E_out_dim, 隐藏层维度是E_hidden_dim, 隐藏层数是E_layer_num
             dropout是dropout概率，LeakyReLU_slope是LeakyReLU的负半轴斜率

        Args:
            in_dim:
            hidden_dim:
            layer_num:
            out_dim:
            dropout:
            LeakyReLU_slope:
        """
        super(MLP, self).__init__()
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



"""
    GIN: Graph Isomorphism Networks
    HOW POWERFUL ARE GRAPH NEURAL NETWORKS? (Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka, ICLR 2019)
    https://arxiv.org/pdf/1810.00826.pdf
"""
class GIN_layer(nn.Module):
    def __init__(self, args, in_features, out_features, eps, GIN_MLP_layer_num):
        """
        GIN单层, 输入X: (batch_size, node_num, in_features), 输出H: (batch_size, node_num, out_features)

        Args:
            args:
            in_features:
            out_features:
            eps: GIN那个eps
            GIN_MLP_layer_num: GIN公式里面那个MLP的层数
        """
        super(GIN_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.MLP = MLP(in_dim=in_features,
                       hidden_dim=out_features,
                       layer_num=GIN_MLP_layer_num,
                       out_dim=out_features,
                       dropout=args.dropout,
                       LeakyReLU_slope=args.LeakyReLU_slope)

    def forward(self, X, A):
        """
        :param X: (batch_size, node_num, in_features)
        :param A: (node_num, node_num)
        """
        H = (1 + self.eps) * X + torch.matmul(A, X)
        'H: (bjatch_size, node_num, in_features)'
        H = self.MLP(H)
        'H: (batch_size, node_num, out_features)'

        return H



class GIN(nn.Module):
    def __init__(self, args):
        """
        GIN, 输入X: (batch_size, node_num, lag), 输出H: (batch_size, node_num, lag)

        Args:
            args:
        """
        super(GIN, self).__init__()
        self.args = args

        GIN_layer_nums = args.GIN_layer_nums
        # self.GIN_eps = torch.nn.Parameter(torch.zeros(len(GIN_layer_nums)))
        self.GIN_eps = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1)) for _ in range(len(GIN_layer_nums))])

        GIN_list = []
        layer_num = len(GIN_layer_nums)
        for i in range(layer_num):
            if i == 0:
                GIN_list.append(GIN_layer(args, args.lag, GIN_layer_nums[i], self.GIN_eps[i], args.GIN_MLP_layer_num))
            else:
                GIN_list.append(GIN_layer(args, GIN_layer_nums[i-1], GIN_layer_nums[i], self.GIN_eps[i], args.GIN_MLP_layer_num))
        self.GIN_list = nn.ModuleList(GIN_list)

        if GIN_layer_nums[-1] != args.lag:
            self.fc = nn.Linear(GIN_layer_nums[-1], args.lag)
            self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)

    def forward(self, X, A):
        """
        :param X: (batch_size, node_num, lag)
        :param A: (node_num, node_num)
        """
        H = X
        'H: (batch_size, node_num, lag)'
        for GIN_layer in self.GIN_list:
            H = GIN_layer(H, A)
            'H: (batch_size, node_num, GIN_layer_nums[-1])'
        if self.args.GIN_layer_nums[-1] != self.args.lag:
            H = self.fc(H)
            'H: (batch_size, node_num, lag)'
            # H = F.leaky_relu(H, negative_slope=self.args.LeakyReLU_slope)
            # 'H: (batch_size*node_num, lag)'

        return H



class SGC(nn.Module):
    def __init__(self, args):
        """
        SGC, 输入X: (batch_size, node_num, lag), 输出H: (batch_size, node_num, lag)

        Args:
            args:
        """
        super(SGC, self).__init__()
        self.args = args

        self.SGC_K = args.SGC_K
        self.SGC_hidden_dim = args.SGC_hidden_dim
        self.lag = args.lag

        self.SGC_W = torch.nn.Parameter(torch.Tensor(self.lag, self.SGC_hidden_dim))

        if self.SGC_hidden_dim != self.lag:
            self.fc = nn.Linear(self.SGC_hidden_dim, self.lag)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.SGC_W)
        if self.SGC_hidden_dim != self.lag:
            torch.nn.init.xavier_uniform_(self.fc.weight)
            torch.nn.init.zeros_(self.fc.bias)

    def forward(self, X, A):
        """
        输入X: (batch_size, node_num, lag), 输出H: (batch_size, node_num, lag)

        Args:
            X:
            A:

        Returns:

        """
        for i in range(self.SGC_K):
            X = torch.matmul(A, X)
            'X: (batch_size, node_num, lag)'
        H = torch.matmul(X, self.SGC_W)
        'H: (batch_size, node_num, SGC_hidden_dim)'

        H = F.leaky_relu(H, negative_slope=self.args.LeakyReLU_slope)
        'H: (batch_size, node_num, SGC_hidden_dim)'

        if self.SGC_hidden_dim != self.lag:
            H = self.fc(H)
            'H: (batch_size, node_num, lag)'

        return H






class GPRGNN(nn.Module):
    def __init__(self, GPRGNN_K, if_self_edge=True):
        """
        GPRGNN, 输入X: (batch_size, node_num, lag), 输出H: (batch_size, node_num, lag)
        ADAPTIVE UNIVERSAL GENERALIZED PAGERANK GRAPH NEURAL NETWORK
        Fig 5 in Graph Neural Networks for Graphs with Heterophily: A Survey
        https://github.com/jianhao2016/GPRGNN/blob/master/src/GNN_models.py#L225

        Args:
            GPRGNN_K: GPRGNN的K值，表示GPRGNN的深度/层数
        """
        super(GPRGNN, self).__init__()
        self.if_self_edge = if_self_edge

        # 定义一个长度为GPRGNN_K的可训练向量
        self.GPRGNN_W = torch.nn.Parameter(torch.Tensor(GPRGNN_K))

        self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.GPRGNN_W)
        torch.nn.init.normal_(self.GPRGNN_W, mean=0.0, std=1.0)

    def forward(self, X, A):
        """
        输入X: (batch_size, node_num, lag), 输出H: (batch_size, node_num, lag)
        Args:
            X: (batch_size, node_num, lag)
            A: (node_num, node_num)

        Returns: (batch_size, node_num, lag)

        """
        if self.if_self_edge:
            H = self.GPRGNN_W[0] * X
            for i in range(1, len(self.GPRGNN_W)):
                H += self.GPRGNN_W[i] * torch.matmul(A, H)
                'H: (batch_size, node_num, lag)'
        else:
            H = torch.zeros_like(X)  # 如果没有自环，则初始化H为全0
            'H: (batch_size, node_num, lag)'
            for i in range(0, len(self.GPRGNN_W)):
                H += self.GPRGNN_W[i] * torch.matmul(A, H)
                'H: (batch_size, node_num, lag)'
        
        return H





class H2GCN(nn.Module):
    def __init__(self, lag, feature_dim, output_lag, H2GCN_round_K=2, dropout=0.3, if_self_edge=True):
        """
        H2GCN, 输入X: (batch_size, node_num, lag), 输出H: (batch_size, node_num, lag)
        Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs
        https://blog.csdn.net/qq_42103091/article/details/126242752?spm=1001.2014.3001.5501
        https://github.com/sxwee/GNNsIMPL/blob/main/H2GCN/model/h2gcn.py
        https://github.com/GitEventhandler/H2GCN-PyTorch/blob/master/model.py

        Args:
            lag: 输入序列的长度
            feature_dim: 输入序列被编码后的维度/长度, lag-->feature_dim
            output_lag: 输出序列的长度, feature_dim-->output_lag
            H2GCN_round_K: H2GCN拼接几轮，就当作K阶数层数用
            dropout: 随机失活概率
        """
        super(H2GCN, self).__init__()
        self.if_self_edge = if_self_edge
        self.H2GCN_round_K = H2GCN_round_K
        self.feature_embedding = nn.Sequential(
            nn.Linear(lag, feature_dim),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(dropout)
        if if_self_edge:
            self.output_embedding = nn.Linear(feature_dim * (2**(H2GCN_round_K+1)-1), output_lag)
        else:
            self.output_embedding = nn.Linear(feature_dim * (2**(H2GCN_round_K+1)-2), output_lag)

        # self.reset_parameters()
    # def reset_parameters(self):
    #     torch.nn.init.xavier_uniform_(self.output_embedding.weight)
    #     torch.nn.init.zeros_(self.output_embedding.bias)

    def norm_adj(self, A, add_self_loops):
        """
        归一化邻接矩阵, 原始H2GCN因为是用于引文网络分类等大规模图计算，所以使用稀疏矩阵实现的，但我这里不需要，直接稠密矩阵计算就行：https://github.com/sxwee/GNNsIMPL/blob/main/H2GCN/utils.py#L23

        Args:
            A: (node_num, node_num)
            add_self_loops: bool 是否添加自环

        return: (node_num, node_num)
        """
        if add_self_loops:
            I = torch.eye(A.shape[-2], A.shape[-1], dtype=A.dtype, device=A.device)
            A = A * (1 - I) + I
        D = torch.sum(A, dim=1)
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt = torch.diag_embed(D_inv_sqrt)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0
        A = torch.matmul(torch.matmul(D_inv_sqrt, A), D_inv_sqrt)
        return A
    
    def hopNeighbor(self, A):
        """
        H2GCN需要计算第二跳的A2 = A*A - A. 整体来讲：A0 = I, A1 = A - I,  A2 = A*A - A - I
        原始H2GCN因为是用于引文网络分类等大规模图计算，所以使用稀疏矩阵实现的，但我这里不需要，直接稠密矩阵计算就行：https://github.com/sxwee/GNNsIMPL/blob/main/H2GCN/utils.py#L16
        Args:
            A: (node_num, node_num)

            return: (node_num, node_num)
        """
        A0 = torch.eye(A.shape[-2], A.shape[-1], dtype=A.dtype, device=A.device)
        A = A * (1 - A0) + A0
        A1 = A - A0
        A2 = torch.matmul(A, A) - A - A0
        A2 = torch.where(A2 > 0, torch.tensor(1.0, device=A.device), torch.tensor(0.0, device=A.device))

        A1 = self.norm_adj(A1, add_self_loops=self.if_self_edge)
        A2 = self.norm_adj(A2, add_self_loops=self.if_self_edge)

        return A0, A1, A2

    def forward(self, X, A):
        """
        输入X: (batch_size, node_num, lag), 输出H: (batch_size, node_num, output_lag)
        Args:
            X: (batch_size, node_num, lag)
            A: (node_num, node_num)  注意H2GCN的输入A必须是【没有】归一化过的A

        Returns: (batch_size, node_num, output_lag)
        """
        'X: (batch_size, node_num, lag)'
        'A: (node_num, node_num)'

        A0, A1, A2 = self.hopNeighbor(A)
        'A0: (node_num, node_num), A1: (node_num, node_num), A2: (node_num, node_num)'

        reps  = []
        'reps: list(K) of each(batch_size, node_num, 2*feature_dim)'

        H = self.feature_embedding(X)
        'H: (batch_size, node_num, feature_dim)'
        if self.if_self_edge: 
            reps.append(H)

        for i in range(self.H2GCN_round_K):
            r1 = torch.matmul(A1, H)
            r2 = torch.matmul(A2, H)
            'r1: (batch_size, node_num, feature_dim), r2: (batch_size, node_num, feature_dim)'
            H = torch.cat([r1, r2], dim=-1)
            reps.append(H)
            'reps: list(K) of each(batch_size, node_num, 2*feature_dim)'

        H = self.dropout(torch.cat(reps, dim=-1))
        'H: (batch_size, node_num, 2*feature_dim*(K+1))'

        H = self.output_embedding(H)
        'H: (batch_size, node_num, output_lag)'

        return H































