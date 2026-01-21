import json
import yaml
import torch








# 参数数据
methods_params = {
    "Example_method": {
        "num_int": {"value": 96, "description_CN": "步长", "description_EN": "Lag"},
        "num_folat": {"value": 6.5, "description_CN": "步长", "description_EN": "Lag"},
        "module": {"value": None, "description_CN": "阈值", "description_EN": "Threshold"},
        "list": {"value": [0.01, 0.2], "description_CN": "学习率", "description_EN": "Learning Rate"},
        "path": {"value": '/data/DiYi/DATA', "description_CN": "学习率", "description_EN": "Learning Rate"},
        "str": {"value": 'Nothing', "description_CN": "窗口大小", "description_EN": "Window Size"},
        "if_false": {"value": False, "description_CN": "窗口大小", "description_EN": "Window Size"},
        "if_true": {"value": True, "description_CN": "窗口大小", "description_EN": "Window Size"},
        # "torch_adam": {"value": torch.optim.Adam, "description_CN": "学习率", "description_EN": "Learning Rate"},
    },
    "Nothing": {
    },
    "Common_configs":{
        "Version": {"value": "V1.0", "description_CN": "实验编号", "description_EN": "Experiment Version"},
        "spatial_block": {"value": "MTGNN", "description_CN": "空间信息捕捉模块，目前可选项有GCN、GAT、GIN、SGC、MTGNN、FourierGNN、StemGNN、GraphWaveNet、Nothing", "description_EN": "Spatial Information Mining Module, currently available options include GCN, GAT, GIN, SGC, MTGNN, FourierGNN, StemGNN, GraphWaveNet, and Nothing."},
        "temporal_block": {"value": "Nothing", "description_CN": "时间信息捕捉模块，目前可选项有TCN、GRU、Transformer、Informer、Autoformer、PatchTST、DLinear。Nothing", "description_EN": "Temporal Information Mining Module, currently available options include TCN, GRU, Transformer, Informer, Autoformer, PatchTST, DLinear, and Nothing."},
        
        "AD_threshold": {"value": 0.5, "description_CN": "异常分数阈值，误差超过这个值就认为是异常", "description_EN": "Anomaly Score Threshold, errors exceeding this value are considered anomalies"},
        "gradient_clip_val": {"value": 100.0, "description_CN": "梯度裁剪，防止梯度爆炸", "description_EN": "Gradient Clipping to prevent gradient explosion"},
        # "optimizer": {"value": torch.optim.Adam, "description_CN": "使用的优化器，默认torch.optim.Adam", "description_EN": "Optimizer used, default is torch.optim.Adam"},
        "lr": {"value": 0.0001, "description_CN": "学习率，默认0.0001", "description_EN": "Learning Rate, default is 0.0001"},
        "scheduler": {"value": "ReduceLROnPlateau", "description_CN": "使用的学习率衰减策略，可选项有ReduceLROnPlateau、StepLR、ExponentialLR、CosineAnnealingLR、CosineAnnealingWarmRestarts", "description_EN": "Learning Rate Decay Strategy used, options include ReduceLROnPlateau, StepLR, ExponentialLR, CosineAnnealingLR, CosineAnnealingWarmRestarts"},
        "patience": {"value": 5, "description_CN": "连续多少个epoch内loss都没有减下去就停止训练", "description_EN": "Number of epochs without loss decrease to stop training"},
        "devices": {"value": [0], "description_CN": "使用的设备，GPU编号列表", "description_EN": "Device used, GPU number list"},

        "TASK": {"value": "anomaly_detection", "description_CN": "任务类型", "description_EN": "Task Type"},
        "Method": {"value": "SPS_AD_LLM_Project", "description_CN": "项目名称", "description_EN": "Project Name"},
        "data_name": {"value": "XJTU-SPS for AD", "description_CN": "数据集名称", "description_EN": "Dataset Name"},
        "Decompose": {"value": "None", "description_CN": "预处理多尺度分解方法", "description_EN": "Preprocessing Multiscale Decomposition Method"},
        "Wavelet_level": {"value": 2, "description_CN": "小波分解层数", "description_EN": "Wavelet Decomposition Level"},
        "if_timestamp": {"value": True, "description_CN": "是否使用时间戳也作为输入信息的一部分", "description_EN": "Whether to use timestamp as part of input information"},
        "if_add_work_condition": {"value": False, "description_CN": "是否使用工况信息也作为输入信息的一部分", "description_EN": "Whether to use working condition information as part of input information"},
        "reco_form": {"value": "all_to_all", "description_CN": "此参数已弃用", "description_EN": "This parameter is deprecated"},
        "result_root_path": {"value": "/data/DiYi/MyWorks_Results", "description_CN": "结果保存路径", "description_EN": "Result Save Path"},

        "dataset_tra_d_val": {"value": 8, "description_CN": "训练集与验证集划分比例", "description_EN": "Train and Validation Set Split Ratio"},
        "batch_size": {"value": 1024, "description_CN": "批处理大小", "description_EN": "Batch Size"},
        "lag": {"value": 96, "description_CN": "序列长度", "description_EN": "Sequence Length"},
        "lag_step": {"value": 1, "description_CN": "滑动窗口步长", "description_EN": "Sliding Window Step"},
        "label_len": {"value": 1, "description_CN": "Transformers自回归起始标签长度", "description_EN": "Transformers Autoregressive Starting Label Length"},
        "pred_len": {"value": 1, "description_CN": "预测长度", "description_EN": "Prediction Length"},
        "scale": {"value": True, "description_CN": "是否执行标准化预处理", "description_EN": "Whether to perform normalization preprocessing"},
        "dropout": {"value": 0.05, "description_CN": "Dropout率", "description_EN": "Dropout Rate"},
        "LeakyReLU_slope": {"value": 0.2, "description_CN": "LeakyReLU斜率", "description_EN": "LeakyReLU Slope"},
        "block_residual": {"value": False, "description_CN": "是否对空间模块使用残差连接", "description_EN": "Whether to use residual connections for the spatial block"},
    },
    "GCN":{
        "GCN_layer_nums": {"value": [96,96], "description_CN": "GCN每层的特征维度列表", "description_EN": "List of dimensions for each GCN layer"},
    },
    "GAT":{
        "Muti_S_GAT_K": {"value": 4, "description_CN": "多头GAT的头数", "description_EN": "Number of heads in Multi-head GAT"},
        "Muti_S_GAT_embed_dim": {"value": 96, "description_CN": "GAT的嵌入维度", "description_EN": "Embedding dimension for GAT"},
        "use_gatv2": {"value": True, "description_CN": "是否使用GATv2", "description_EN": "Whether to use GATv2"},
    },
    "GIN":{
        "GIN_layer_nums": {"value": [96, 96], "description_CN": "GIN每层的特征维度列表", "description_EN": "List of dimensions for each GIN layer"},
        "GIN_MLP_layer_num": {"value": 1, "description_CN": "GIN公式中的MLP个数", "description_EN": "Number of MLPs in GIN formula"},
    },
    "SGC":{
        "SGC_hidden_dim": {"value": 96, "description_CN": "SGC的隐藏层维度", "description_EN": "Hidden dimension for SGC"},
        "SGC_K": {"value": 3, "description_CN": "SGC的K值", "description_EN": "K value for SGC"},
    },
    "MLP":{
        "MLP_hidden_dim": {"value": 96, "description_CN": "MLP的隐藏层维度", "description_EN": "Hidden dimension for MLP"},
        "MLP_layer_num": {"value": 2, "description_CN": "MLP的隐藏层层数", "description_EN": "Number of hidden layers in MLP"},
    },
    "TCN":{
        "TCN_num_channels": {"value": [48, 48], "description_CN": "TCN每层的通道数列表", "description_EN": "List of channels for each TCN layer"},
        "TCN_kernel_size": {"value": 2, "description_CN": "TCN的卷积核大小", "description_EN": "Kernel size for TCN"},
        "temporal_block_end_mlp_hidden_dim": {"value": 128, "description_CN": "最后一层用于统一维度的MLP的隐藏层维度", "description_EN": "Hidden dimension for MLP mapping to unified dimension after last dimension change"},
        "temporal_block_end_mlp_layer_num": {"value": 2, "description_CN": "最后一层用于统一维度的MLP的层数", "description_EN": "Number of layers in MLP mapping to unified dimension after last dimension change"},
    },
    "GRU":{
        "transf_GRU_hidden_size": {"value": 64, "description_CN": "GRU的h的维度", "description_EN": "Hidden dimension for GRU"},
        "transf_GRU_layers": {"value": 1, "description_CN": "GRU层数", "description_EN": "Number of GRUs"},
        "temporal_block_end_mlp_hidden_dim": {"value": 128, "description_CN": "最后一层用于统一维度的MLP的隐藏层维度", "description_EN": "Hidden dimension for MLP mapping to unified dimension after last dimension change"},
        "temporal_block_end_mlp_layer_num": {"value": 2, "description_CN": "最后一层用于统一维度的MLP的层数", "description_EN": "Number of layers in MLP mapping to unified dimension after last dimension change"},
    },
    "MTGNN":{
        "MTGNN_gcn_true": {"value": True, "description_CN": "是否增加GCN层", "description_EN": "Whether to add GCN layers"},
        "MTGNN_buildA_true": {"value": True, "description_CN": "是否构建自适应的A矩阵", "description_EN": "Whether to build adaptive A matrix"},
        "MTGNN_gcn_depth": {"value": 2, "description_CN": "mixprop里面的图卷积深度", "description_EN": "Graph convolution depth in mixprop"},
        "MTGNN_graph_k": {"value": 24, "description_CN": "每个节点几个邻居节点", "description_EN": "Number of neighbors for each node"},
        "MTGNN_node_embedding": {"value": 192, "description_CN": "节点嵌入的维度", "description_EN": "Dimension of node embedding"},
        "MTGNN_dilation_exponential": {"value": 1, "description_CN": "膨胀指数", "description_EN": "Dilation exponential"},
        "MTGNN_conv_channels": {"value": 32, "description_CN": "TC模块输入维度是residual_channels，输出维度是conv_channels", "description_EN": "Input dimension for TC module is MTGNN_residual_channels, output dimension is MTGNN_conv_channels"},
        "MTGNN_residual_channels": {"value": 32, "description_CN": "GC模块输入维度是conv_channels，输出维度是residual_channels", "description_EN": "Input dimension for GC module is MTGNN_conv_channels, output dimension is MTGNN_residual_channels"},
        "MTGNN_skip_channels": {"value": 64, "description_CN": "跳跃连接的输出维度。（输入维度是conv_channels）", "description_EN": "Output dimension for skip connection (input dimension is MTGNN_conv_channels)"},
        "MTGNN_end_channels": {"value": 128, "description_CN": "end_conv_1的输出维度", "description_EN": "Output dimension for end_conv_1"},
        "MTGNN_in_dim": {"value": 1, "description_CN": "输入维度，设置为1即可", "description_EN": "Input dimension, set to 1"},
        "MTGNN_layers": {"value": 3, "description_CN": "多少层GC + TC", "description_EN": "Number of layers for GC + TC"},
        "MTGNN_propalpha": {"value": 0.05, "description_CN": "Prop alpha，即在混合跳传播中保留根节点原始状态的比例，取值范围在0到1之间", "description_EN": "Prop alpha, the proportion of the root node's original state retained in mixed jump propagation, range from 0 to 1"},
        "MTGNN_tanhalpha": {"value": 3, "description_CN": "生成邻接矩阵时的双曲正切alpha值，alpha控制饱和率", "description_EN": "Hyperbolic tangent alpha value for generating adjacency matrix, controlling saturation"},
        "MTGNN_layer_norm_affline": {"value": True, "description_CN": "在层归一化中是否进行逐元素仿射操作", "description_EN": "Whether to perform element-wise affine operation in layer normalization"},
    },
    "FourierGNN":{
    },
    "StemGNN":{
    },
    "GraphWaveNet":{
    },
    "Transformer":{
    },
    "Informer":{
    },
    "Autoformer":{
    },
    "PatchTST":{
    },
    "DLinear":{
    },
}





# # 写入 JSON 文件
# with open("/home/dy29/MyWorks_Codes/15_LLM_4_SPS_PHM/AD_repository/model/AD_algorithms_params/all_AD_algorithm_params.json", "w") as f:
#     json.dump(methods_params, f, indent=4)
# # 读取 JSON 文件
# with open("methods_params.json", "r") as f:
#     loaded_params = json.load(f)
#     print(loaded_params["method1"][0]["value"])  # 输出：96



# 写入 YAML 文件
with open("/home/dy29/MyWorks_Codes/15_LLM_4_SPS_PHM/AD_repository/model/AD_algorithms_params/all_AD_algorithm_params.yaml", "w") as f:
    yaml.dump(methods_params, f, default_flow_style=False, allow_unicode=True)
# # 读取 YAML 文件
# with open("/home/dy29/MyWorks_Codes/15_LLM_4_SPS_PHM/AD_repository/model/AD_algorithms_params/all_AD_algorithm_params.yaml", "r") as f:
#     loaded_params = yaml.safe_load(f)
# # 使用读取的数据
# print(loaded_params["Example_method"][0]["value"])  # 输出：96