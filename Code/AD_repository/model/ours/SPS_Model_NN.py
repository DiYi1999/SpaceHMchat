from AD_repository.model.ours.spatial_block import *
from AD_repository.model.ours.temporal_block import *
from AD_repository.model.baseline.MTGNN.MTGNN import MTGNN
from AD_repository.model.baseline.Autoformer.Autoformer import Autoformer_4ours
from AD_repository.model.baseline.Transformer.Transformer import Transformer_4ours
from AD_repository.model.baseline.Informer.Informer import Informer_4ours
from AD_repository.model.baseline.PatchTST.PatchTST import PatchTST_4ours
from AD_repository.model.baseline.Graph_WaveNet.Graph_WaveNet import Graph_WaveNet_4ours
from AD_repository.model.baseline.FourierGNN.FourierGNN import FourierGNN_4ours
from AD_repository.model.baseline.DLinear.DLinear import DLinear_4ours
from AD_repository.model.baseline.StemGNN.StemGNN import StemGNN_4ours



class SPS_Model_NN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.node_num = args.node_num
        self.sensor_num = args.sensor_num

        ### temporal_block for temporal modeling
        if args.temporal_block == 'MLP':
            self.temporal_block = MLP_dim2(in_dim=self.sensor_num,
                                                 hidden_dim=args.transf_MLP_hidden_dim,
                                                 out_dim=self.sensor_num,
                                                 layer_num=args.transf_MLP_layer_num,
                                                 dropout=args.dropout,
                                                 LeakyReLU_slope=args.LeakyReLU_slope)
            # (batch_size, in_node_num, lag) -> (batch_size, out_node_num, lag)
        elif args.temporal_block == 'TCN':
            self.temporal_block = TCN(num_inputs=self.sensor_num,
                                            num_channels=args.transf_TCN_num_channels,
                                            kernel_size=args.transf_TCN_kernel_size,
                                            dropout=args.dropout)
            # (batch, node_num, lag) -> (batch, TCN_layers_channels[-1], lag)
            if args.transf_TCN_num_channels[-1] != self.sensor_num:
                self.temporal_block_end = MLP_dim2(in_dim=args.transf_TCN_num_channels[-1],
                                                         hidden_dim=args.temporal_block_end_mlp_hidden_dim,
                                                         layer_num=args.temporal_block_end_mlp_layer_num,
                                                         out_dim=self.sensor_num,
                                                         dropout=args.dropout,
                                                         LeakyReLU_slope=args.LeakyReLU_slope)
        elif args.temporal_block == 'GRU':
            self.temporal_block = GRU(input_size=self.sensor_num,
                                            hidden_size=args.transf_GRU_hidden_size,
                                            num_layers=args.transf_GRU_layers,
                                            dropout=args.dropout)
            # (batch, node_num, lag) -> (batch, hidden_size, lag)
            if args.transf_GRU_hidden_size != self.sensor_num:
                self.temporal_block_end = MLP_dim2(in_dim=args.transf_GRU_hidden_size,
                                                         hidden_dim=args.temporal_block_end_mlp_hidden_dim,
                                                         layer_num=args.temporal_block_end_mlp_layer_num,
                                                         out_dim=self.sensor_num,
                                                         dropout=args.dropout,
                                                         LeakyReLU_slope=args.LeakyReLU_slope)
        elif args.temporal_block == 'Autoformer':
            self.temporal_block = Autoformer_4ours(args)
            # (batch, node_num, lag) -> (batch, node_num, pred_len)
        elif args.temporal_block == 'Transformer':
            self.temporal_block = Transformer_4ours(args)
            # (batch, node_num, lag) -> (batch, node_num, pred_len)
        elif args.temporal_block == 'Informer':
            self.temporal_block = Informer_4ours(args)
            # (batch, node_num, lag) -> (batch, node_num, pred_len)
        elif args.temporal_block == 'PatchTST':
            self.temporal_block = PatchTST_4ours(args)
            # (batch, node_num, lag) -> (batch, node_num, pred_len)
        elif args.temporal_block == 'DLinear':
            self.temporal_block = DLinear_4ours(args)
            # (batch, node_num, lag) -> (batch, node_num, pred_len)            
        elif args.temporal_block == 'Nothing':
            self.temporal_block = Do_Nothing()
        else:
            raise Exception("No such temporal_block! must be 'MLP' or 'TCN' or 'GRU'!")

        ### spatial_block for spatial modeling
        if args.graph_ca_meth == 'Training':
            self.A = Parameter(torch.Tensor(args.node_num, args.node_num))
            nn.init.xavier_uniform_(self.A)
            # 'self.A: (node_num, node_num)'
        else:
            self.A = None

        if args.spatial_block == 'MTGNN':
            self.spatial_block = MTGNN(args=args)
            # (batch_size, node_num, lag) -> (batch_size, node_num, lag/pred_len)
        elif args.spatial_block == 'StemGNN':
            self.spatial_block = StemGNN_4ours(args)
            # (batch_size, node_num, lag) -> (batch_size, node_num, lag/pred_len)
        elif args.spatial_block == 'FourierGNN':
            self.spatial_block = FourierGNN_4ours(args)
            # (batch_size, node_num, lag) -> (batch_size, node_num, lag/pred_len)
        elif args.spatial_block == 'Graph_WaveNet':
            self.spatial_block = Graph_WaveNet_4ours(args)
            # (batch_size, node_num, lag) -> (batch_size, node_num, lag/pred_len)
        # elif args.spatial_block == 'G3CN':
        #     self.spatial_block = CMTS_GCN(CMTS_GCN_K_nums=args.CMTS_GCN_K_nums,
        #                                   node_num=args.node_num,
        #                                   CMTS_GCN_residual=args.CMTS_GCN_residual,
        #                                   LeakyReLU_slope=args.LeakyReLU_slope)
        #     # (batch_size, node_num, lag) -> (batch_size, node_num, lag)
        elif args.spatial_block == 'GCN':
            self.spatial_block = GCN_s(GCN_layer_nums=args.GCN_layer_nums,
                                       node_num=args.node_num,
                                       lag=args.lag,
                                       LeakyReLU_slope=args.LeakyReLU_slope)
            # (batch_size, node_num, lag) -> (batch_size, node_num, lag)
        elif args.spatial_block == 'GAT':
            self.spatial_block = Muti_S_GAT(Muti_S_GAT_K=args.Muti_S_GAT_K,
                                            Muti_S_GAT_embed_dim=args.Muti_S_GAT_embed_dim,
                                            node_num=args.node_num,
                                            lag=args.lag,
                                            use_gatv2=args.use_gatv2,
                                            dropout=args.dropout,
                                            LeakyReLU_slope=args.LeakyReLU_slope)
            # (batch_size, node_num, lag) -> (batch_size, node_num, lag)
        elif args.spatial_block == 'GIN':
            self.spatial_block = GIN(GIN_layer_nums=args.GIN_layer_nums,
                                     GIN_MLP_layer_num=args.GIN_MLP_layer_num,
                                     lag=args.lag,
                                     dropout=args.dropout,
                                     LeakyReLU_slope=args.LeakyReLU_slope)
            # (batch_size, node_num, lag) -> (batch_size, node_num, lag)
        elif args.spatial_block == 'SGC':
            self.spatial_block = SGC(SGC_K=args.SGC_K,
                                     SGC_hidden_dim=args.SGC_hidden_dim,
                                     lag=args.lag,
                                     LeakyReLU_slope=args.LeakyReLU_slope)
            # (batch_size, node_num, lag) -> (batch_size, node_num, lag)
        elif args.spatial_block == 'Nothing':
            self.spatial_block = Do_Nothing()
            # (batch_size, node_num, lag) -> (batch_size, node_num, lag)
        else:
            raise Exception("No such spatial_block! must be 'G3CN' or 'GCN' or 'GAT' or 'GIN' or 'SGC'!")


    def forward(self, A, X, WC, T, T_of_y, Y):
        """
        :param A: (node_num, node_num)
        :param X: (batch_size, sensor_num, lag)
        :param WC: (batch_size, 4, lag), actually are working conditions: irradiance, temperature, wind speed, load
        :param T: (batch_size, 5, lag) DAY, HOUR, MINUTE, SECOND, TIMESTAMP
        :param T: (batch_size, 4, lag), DAY, HOUR, MINUTE, SECOND
        :param T_of_y: (batch_size, 4, pred_len), DAY, HOUR, MINUTE, SECOND, used for transformers as Autoregressive beginning
        :param Y: R(batch_size, node_num, lag) or F(batch_size, node_num, label_len+pred_len), used for transformers as Autoregressive beginning

        return: H: (batch_size, sensor_num, lag)
        """
        if self.A is not None:
            A = self.A
            'A: (node_num, node_num)'


        # temporal_block
        if 'former' in self.args.temporal_block:
            X2 = self.temporal_block(X, T, T_of_y, Y)
        else:
            X2 = self.temporal_block(X)
        'X2: (batch_size, TCN_layers_channels[-1], lag)'
        if X2.size(1) != self.sensor_num:
            X2 = self.temporal_block_end(X2)
        'X2: (batch_size, sensor_num, lag)'

        # spatial_block
        X2 = torch.cat((X, T), dim=1) if self.args.if_timestamp else X2
        'X2: (batch_size, sensor_num+5, lag)'
        X2 = torch.cat((X2, WC), dim=1) if self.args.if_add_work_condition else X2
        'X2: (batch_size, sensor_num(+5)+4, lag)'
        H = self.spatial_block(X2, A)
        'H: (batch_size, sensor_num(+5+4), lag)'
        if self.args.if_add_work_condition:
            H = H[:, :-WC.size(1), :]
            'H: (batch_size, sensor_num(+5), lag)'
        if self.args.if_timestamp:
            H = H[:, :-T.size(1), :]
            'H: (batch_size, sensor_num, lag)'

        return H


















