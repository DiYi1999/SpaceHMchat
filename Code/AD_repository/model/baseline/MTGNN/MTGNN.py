from AD_repository.model.baseline.MTGNN.MTGNN_layer import *


class MTGNN(nn.Module):
    def __init__(self, args, predefined_A=None, static_feat=None):
        """

        Args:
            args:
            predefined_A: 预定义的A矩阵
            static_feat: 用来计算邻接矩阵的特征，若为None，它会用nn.Embedding随机生成，相当于学习特征用于计算邻接矩阵
        """
        super(MTGNN, self).__init__()

        gcn_true = args.MTGNN_gcn_true
        '是否增加GCN层'
        buildA_true = args.MTGNN_buildA_true
        '是否构建自适应的A矩阵'
        gcn_depth = args.MTGNN_gcn_depth
        'mixprop里面的图卷积深度'
        num_nodes = args.node_num
        '节点数'
        dropout = args.dropout
        '丢弃率'
        subgraph_size = args.MTGNN_graph_k
        '每个节点几个邻居'
        node_dim = args.MTGNN_node_embedding
        '节点embedding的维度'
        dilation_exponential = args.MTGNN_dilation_exponential
        '膨胀指数'
        conv_channels = args.MTGNN_conv_channels
        'TC模块输入维度是residual_channels，输出维度是conv_channels'
        residual_channels = args.MTGNN_residual_channels
        'GC模块输入维度是conv_channels，输出维度是residual_channels'
        skip_channels = args.MTGNN_skip_channels
        'skip_channels是skip connection的输出维度，是图二下半部分的Skip Connection，输入维度是conv_channels，输出维度是skip_channels'
        end_channels = args.MTGNN_end_channels
        'end_conv_1将skip_channels映射到end_channels，end_conv_2将end_channels映射到下面的out_dim'
        seq_length = args.lag
        'seq_length是seq_in_len，是输入序列的长度，其实就是lag'
        in_dim = args.MTGNN_in_dim
        'in_dim是输入特征的维度/MTGNN特有的那个多出来的维度的通道数'
        out_dim = args.pred_len if args.BaseOn == "forecast" else args.lag
        'end_conv_2将end_channels映射到下面的out_dim，MTGNN很神奇新增一个维度扩增到out_dim也就是pred_len'
        layers = args.MTGNN_layers
        '多少层GC+TC'
        propalpha = args.MTGNN_propalpha
        'Prop alpha，即在混合跳传播中保留根节点原始状态的比例，取值范围在 0 到 1 之间。'
        tanhalpha = args.MTGNN_tanhalpha
        '生成邻接矩阵时的双曲正切 alpha 值，alpha 控制饱和率。'
        layer_norm_affline = args.MTGNN_layer_norm_affline
        '在层归一化中是否进行逐元素仿射操作。'


        self.gcn_true = gcn_true
        '是否增加GCN层'
        self.buildA_true = buildA_true
        '是否构建自适应的A矩阵'
        self.num_nodes = num_nodes
        '节点数'
        self.dropout = dropout
        self.predefined_A = predefined_A
        '预定义的A矩阵'
        self.filter_convs = nn.ModuleList()
        'TC模块的左边卷积路，输入维度是residual_channels，输出维度是conv_channels'
        self.gate_convs = nn.ModuleList()
        'TC模块的右边卷积路，输入维度是residual_channels，输出维度是conv_channels'
        self.residual_convs = nn.ModuleList()
        '只有在gcn_true为False时才会用到，跨过GC模块直接的残差卷积，输入维度是conv_channels，输出维度是residual_channels'
        self.skip_convs = nn.ModuleList()
        '好像是图二下半部分的Skip Connection，输入维度是conv_channels，输出维度是skip_channels'
        self.gconv1 = nn.ModuleList()
        'GC模块的左边卷积路，输入维度是conv_channels，输出维度是residual_channels'
        self.gconv2 = nn.ModuleList()
        'GC模块的右边卷积路，输入维度是conv_channels，输出维度是residual_channels'
        self.norm = nn.ModuleList()
        '每个TC\GC层的最后一步，LayerNorm层，输入维度是residual_channels，输出维度是residual_channels'
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))

        self.args = args
        self.node_num = args.node_num
        'node_num: 传感器个数*分解 + 年月日'
        self.node_num_output = args.node_num - args.timestamp_dim if args.if_timestamp else args.node_num
        self.node_num_output = self.node_num_output - 4 if args.if_add_work_condition else self.node_num_output
        'node_num_output: node_num-timestamp_dim是空间模块的输出维度' \
        '，单输出sensor_num个传感器的数据，或者sensor_num*3如果数据被分解了的话'

        ##### 邻接矩阵
        self.gc = graph_constructor(nnodes=args.node_num, k=subgraph_size, dim=node_dim,
                                    alpha=tanhalpha, static_feat=static_feat)
        '构建图结构' \
        'static_feat: 用来计算邻接矩阵的特征，若为None，它会用nn.Embedding随机生成，相当于学习特征用于计算邻接矩阵'

        self.seq_length = seq_length
        'seq_length是seq_in_len，是输入序列的长度，其实就是lag。' \
        'out_dim是pred_len/lag，是end_conv_2从end_channels映射到out_dim的，MTGNN很神奇新增一个维度扩增到out_dim也就是pred_len' \
        'in_dim是in_dim，是输入特征的维度/MTGNN特有的那个多出来的维度的通道数。'

        kernel_size = 7
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1

        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                if not self.gcn_true:
                    self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                        out_channels=residual_channels,
                                                     kernel_size=(1, 1)))
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)))

                if self.gcn_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                             out_channels=out_dim,
                                             kernel_size=(1,1),
                                             bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)


        self.idx = torch.arange(num_nodes)


    def forward(self, X, A, idx=None):
        """
        :param X: (batch_size, sensor_num(+5)+4, lag)
        :param A: (node_num, node_num)
        :param idx: 只用部分节点的话，传入这些节点的索引

        return: H: (batch_size, sensor_num(+5)+4, lag)
        """
        self.idx = self.idx.to(X.device)

        # in_dim = 1
        input = X.unsqueeze(1)
        'input: (batch_size, in_dim=1, sensor_num(+5)+4, lag)'
        if self.seq_length<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))

        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx)
                    'adp 邻接矩阵: (sensor_num(+5)+4, sensor_num(+5)+4)'
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A

        x = self.start_conv(input)
        'x: (batch_size, in_dim=1, sensor_num(+5)+4, lag) -> (batch_size, residual_channels, sensor_num(+5)+4, lag)'
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        'skip: (batch_size, in_dim=1, sensor_num(+5)+4, lag) -> (batch_size, skip_channels, sensor_num(+5)+4, lag)'

        for i in range(self.layers):
            residual = x
            'residual: (batch_size, residual_channels, sensor_num(+5)+4, lag)'

            # TC模块，输入维度是residual_channels，输出维度是conv_channels
            filter = self.filter_convs[i](x)
            'filter: (batch_size, conv_channels, sensor_num(+5)+4, lag)'
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            'gate: (batch_size, conv_channels, sensor_num(+5)+4, lag)'
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            '(batch_size, conv_channels, sensor_num(+5)+4, lag)'

            # skip connection，输入维度是conv_channels，输出维度是skip_channels
            s = x
            's: (batch_size, skip_channels, sensor_num(+5)+4, lag)'
            s = self.skip_convs[i](s)
            's: (batch_size, skip_channels, sensor_num(+5)+4, 1)'
            skip = s + skip

            # GC模块，输入维度是conv_channels，输出维度是residual_channels
            if self.gcn_true:
                x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1,0))
                'x: (batch_size, residual_channels, sensor_num(+5)+4, lag)'
            else:
                x = self.residual_convs[i](x)

            # residual、LayerNorm层，输入维度是residual_channels，输出维度是residual_channels
            x = x + residual[:, :, :, -x.size(3):]
            'x: (batch_size, residual_channels, sensor_num(+5)+4, lag)'
            if idx is None:
                x = self.norm[i](x,self.idx)
                'x: (batch_size, residual_channels, sensor_num(+5)+4, lag)'
            else:
                x = self.norm[i](x,idx)

        # skip connection end，输入维度是residual_channels，输出维度是skip_channels
        skip = self.skipE(x) + skip
        'skip: (batch_size, skip_channels, sensor_num(+5)+4, 1)'
        x = F.relu(skip)
        ### 最后一层卷积，输入维度是skip_channels，输出维度是end_channels
        x = F.relu(self.end_conv_1(x))
        'x: (batch_size, end_channels, sensor_num(+5)+4, 1)'
        ### 最后一层卷积，输入维度是skip_channels，输出维度是out_dim
        x = self.end_conv_2(x)
        'x: (batch_size, out_dim=pred_len/lag, sensor_num(+5)+4, 1)'

        # x替换1、3维度，并去掉第1维度
        H_S = x.transpose(1, 3).squeeze(1)
        'H_S: (batch_size, sensor_num(+5)+4, out_dim=pred_len/lag)'

        return H_S