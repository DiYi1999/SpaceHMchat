import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys


"https://github.com/nnzhan/Graph-WaveNet/blob/master/model.py"


class Graph_WaveNet_4ours(nn.Module):
    def __init__(self, args):
        super(Graph_WaveNet_4ours, self).__init__()
        # self.BaseOn = args.BaseOn
        # self.pred_len = args.pred_len
        self.gwnet = gwnet(num_nodes=args.node_num,
                            dropout=args.dropout,
                            supports=None,
                            gcn_bool=True,
                            addaptadj=True,
                            aptinit=None,
                            in_dim=1,
                            out_dim=1,
                            residual_channels=32,
                            dilation_channels=32,
                            skip_channels=256,
                            end_channels=512,
                            kernel_size=2,
                            blocks=4,
                            layers=2)

    def forward(self, X, A):
        """

        Args:
            X: (batch_size, node_num, lag)
            A: (num_node, num_node)

        Returns:

        """
        x = X.unsqueeze(1)
        # x: (batch_size, in_dim=1, num_nodes, seq_len=lag)
        out = self.gwnet(x)
        # out: (batch_size, out_dim=1, num_nodes, seq_len=lag)
        out = out.squeeze(1)
        # out: (batch_size, num_nodes, seq_len=lag)
        # if self.BaseOn == "forecast":
        #     out = out[:, :, -self.pred_len:]
        #     # out: (batch_size, num_nodes, pred_len)

        return out



class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()



class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        # x: (batch_size, c_in, num_nodes, seq_len)
        return self.mlp(x)
        # output x: (batch_size, c_out, num_nodes, seq_len)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        # input: (batch_size, c_in, num_nodes, seq_len)
        # output: (batch_size, c_in, num_nodes, seq_len)
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        # x: (batch_size, num_channels, num_nodes, seq_len)
        # support: list of adjacency matrices, each (num_nodes, num_nodes)
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            # x1: (batch_size, num_channels, num_nodes, seq_len)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                # x2: (batch_size, num_channels, num_nodes, seq_len)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        # h: (batch_size, (order * support_len + 1) * num_channels, num_nodes, seq_len)
        h = self.mlp(h)
        # h: (batch_size, c_out, num_nodes, seq_len)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gwnet(nn.Module):
    def __init__(self,
                 # device,
                 num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True,
                 aptinit=None, in_dim=2, out_dim=12, residual_channels=32, dilation_channels=32,
                 skip_channels=256, end_channels=512, kernel_size=2, blocks=4, layers=2):
        """

        Args:
            device:    被我弃用了这个参数
            num_nodes: number of nodes
            dropout:
            supports: None, list of adjacency matrices, each (num_nodes, num_nodes)
            gcn_bool: True
            addaptadj: True
            aptinit: None
            in_dim: 1
            out_dim: 1
            residual_channels: 32
            dilation_channels: 32
            skip_channels: 256
            end_channels:  512
            kernel_size: 2
            blocks: 4
            layers: 2
        """
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        # input: (batch_size, in_dim, num_nodes, seq_len)
        # output: (batch_size, residual_channels, num_nodes, seq_len)
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                # self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10), requires_grad=True)
                # nodevec1: (num_nodes, 10)
                # self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes), requires_grad=True)
                # nodevec2: (10, num_nodes)
                self.supports_len += 1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                # initemb1: (num_nodes, 10)
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                # initemb2: (10, num_nodes)
                # self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True)
                # self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))
                # filter_convs: input (batch_size, residual_channels, num_nodes, seq_len)
                # output: (batch_size, dilation_channels, num_nodes, seq_len)

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))
                # gate_convs: input (batch_size, residual_channels, num_nodes, seq_len)
                # output: (batch_size, dilation_channels, num_nodes, seq_len)

                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))
                # residual_convs: input (batch_size, dilation_channels, num_nodes, seq_len)
                # output: (batch_size, residual_channels, num_nodes, seq_len)

                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                # skip_convs: input (batch_size, dilation_channels, num_nodes, seq_len)
                # output: (batch_size, skip_channels, num_nodes, seq_len)

                self.bn.append(nn.BatchNorm2d(residual_channels))
                # batch normalization for residual_channels

                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))
                    # gconv: input (batch_size, dilation_channels, num_nodes, seq_len)
                    # output: (batch_size, residual_channels, num_nodes, seq_len)

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)
        # end_conv_1: input (batch_size, skip_channels, num_nodes, seq_len)
        # output: (batch_size, end_channels, num_nodes, seq_len)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)
        # end_conv_2: input (batch_size, end_channels, num_nodes, seq_len)
        # output: (batch_size, out_dim, num_nodes, seq_len)

        self.receptive_field = receptive_field

    def forward(self, input):
        # input: (batch_size, in_dim, num_nodes, seq_len)
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
            # x: (batch_size, in_dim, num_nodes, receptive_field)
        else:
            x = input
        x = self.start_conv(x)
        # x: (batch_size, residual_channels, num_nodes, seq_len)
        skip = 0

        new_supports = None
        # 下面我加的
        self.nodevec1.to(input.device)
        self.nodevec2.to(input.device)
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            # adp: (num_nodes, num_nodes)
            new_supports = self.supports + [adp]

        for i in range(self.blocks * self.layers):
            residual = x
            filter = self.filter_convs[i](residual)
            # filter: (batch_size, dilation_channels, num_nodes, seq_len)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            # gate: (batch_size, dilation_channels, num_nodes, seq_len)
            gate = torch.sigmoid(gate)
            x = filter * gate
            # x: (batch_size, dilation_channels, num_nodes, seq_len)

            s = x
            s = self.skip_convs[i](s)
            # s: (batch_size, skip_channels, num_nodes, seq_len)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip
            # skip: (batch_size, skip_channels, num_nodes, seq_len)

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)
            # x: (batch_size, residual_channels, num_nodes, seq_len)

            x = x + residual[:, :, :, -x.size(3):]
            # x: (batch_size, residual_channels, num_nodes, seq_len)

            x = self.bn[i](x)
            # x: (batch_size, residual_channels, num_nodes, seq_len)

        x = F.relu(skip)
        # x: (batch_size, skip_channels, num_nodes, seq_len)
        x = F.relu(self.end_conv_1(x))
        # x: (batch_size, end_channels, num_nodes, seq_len)
        x = self.end_conv_2(x)
        # x: (batch_size, out_dim, num_nodes, seq_len)
        return x
