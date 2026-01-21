import os
from argparse import ArgumentParser
import math
from AD_repository.model.mycallback import MyEarlyStopping
from AD_repository.model.MyModel import *
import torch
"Don't touch the following codes"
def import_lightning():
    try:
        import lightning.pytorch as pl
    except ModuleNotFoundError:
        import pytorch_lightning as pl
    return pl
pl = import_lightning()

from AD_repository.data.lightingdata import MyLigDataModule
import numpy as np
from ray import air, tune
from ray.tune.search.ax import AxSearch
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
# from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
### 这个TuneReportCheckpointCallback官方没更新，还用的是pytorch_lightning，而我的lighting2.0用的是pytorch.lightning
# import ptwt, pywt
from AD_repository.main_sub import *


def set_args():
    parser = ArgumentParser()

    ### 任务设置
    parser.add_argument('--TASK', type=str, default='anomaly_detection',
                        help='anomaly_detection or forecast or reconstruct'
                             '此处TASK和下面的BaseOn的关系是：'
                             '如果TASK是anomaly_detection，完成异常检测任务可以是基于重建，也可以是基于预测，'
                             '    本研究默认是基于预测，若效果不好可改成baseon预测，并且必须确保有label数据文件'
                             '如果TASK是forecast，那么BaseOn只能是forecast，不需要label数据文件'
                             '如果TASK是reconstruct，那么BaseOn只能是reconstruct，不需要label数据文件只重建'
                             '    之所以设置这个reconstruct，是主要用于MIC_simulate数据集，重建该仿真数据')
    parser.add_argument('--BaseOn', type=str, default='forecast',
                        help='reconstruct or forecast，建议设置reconstruct，当前工况用于预测未来的传感器观测量是不合适的')
    parser.add_argument('--data_name', type=str, default='XJTU-SPS for AD',
                        help='数据集名字XJTU-SPS for AD')
    parser.add_argument('--Decompose', type=str, default='None', help='None 不分解/STL/WaveletPacket/Wavelet')
    # if TASK == 'forecast':
    parser.add_argument('--channel_to_channel', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, '
                             'MS:multivariate predict univariate, S:univariate predict univariate')
    parser.add_argument('--MS_which', type=int, default=1,
                        help='如果上面选了MS，那么要指定用哪个单列，'
                             '1一般用在MSL和SMAP等数据集，表示用第一列，忽略其他控制量列'
                             '-1一般用在ETT等数据集，表示用最后一列，就是那个OT列')
    # if TASK == 'reconstruct':
    parser.add_argument('--reco_form', type=str, default='all_to_all',
                        help='有些重建数据集是解微分方程的，会出现用前4维数据重建后4维的情况，而不是全体重建，视数据集决定'
                             ', options:[all_to_all, half_to_half]，注意分解策略和half_to_half不兼容，在dataset的图结构计算那里')



    ### 实验设置
    parser.add_argument('--Version', type=str, default='V0.000', help='代码版本 V1.00')
    parser.add_argument('--Method', type=str, default='SPS_AD_LLM_Project', help='项目名称，最大一级文件目录')
    exp_name = parser.parse_known_args()[0].Version + '_' + \
               parser.parse_known_args()[0].Method + '_' + \
               parser.parse_known_args()[0].data_name + '_' + \
               parser.parse_known_args()[0].Decompose + '_' + \
               parser.parse_known_args()[0].TASK
    parser.add_argument('--exp_name', type=str, default=exp_name, help='实验名称/excel文件名/保存文件夹名等')



    ### 数据导入路径
    """f = open(os.path.join(self.root_path, self.data_path, '{}_train.csv'.format(self.data_name)), "rb")"""

    parser.add_argument('--root_path', type=str, default='/data/DiYi/DATA',
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='Our_Exp_Data/XJTU-SPS Dataset/XJTU-SPS for AD',
                        help='Our_Exp_Data/XJTU-SPS Dataset/XJTU-SPS for AD')



    ### 路径相关
    parser.add_argument('--result_root_path', type=str, default='/data/DiYi/MyWorks_Results',
                        help='实验结果的保存路径')
    save_path = parser.parse_known_args()[0].result_root_path + '/' + \
                parser.parse_known_args()[0].Method + '/' + \
                parser.parse_known_args()[0].data_name + '/' + \
                parser.parse_known_args()[0].exp_name
    parser.add_argument('--ckpt_save_path', type=str, default= save_path + '/ckpt',
                        help='检查点缓存路径，同时也是日志保存路径')
    parser.add_argument('--table_save_path', type=str, default=save_path + '/table',
                        help='调试结果的csv/excel文件保存路径')
    parser.add_argument('--report_save_path', type=str, default=save_path + '/report',
                        help='报告的保存路径')
    parser.add_argument('--plot_save_path', type=str, default=save_path + '/plot',
                        help='画图的保存路径')



    ### node_num相关策略制定
    # 如果上面的Decompose不是None，STL/Wavelet分解参数
    parser.add_argument('--STL_seasonal', type=int, default=7,
                        help='Length of the seasonal smoother. Must be an odd integer, and should normally be >= 7 (default).')
    parser.add_argument('--Wavelet_wave', type=str, default='db4',
                        help='小波基函数，默认db4')
    parser.add_argument('--Wavelet_level', type=int, default=2,
                        help='小波分解层数，如是2则同STL，分解成3层；如是5则分解成5层')
    parser.add_argument('--if_timestamp', type=bool, default=True,
                        help='是否在预测或重建时利用上timestamp数据，即年周月，一定设置False')
    parser.add_argument('--if_add_work_condition', type=bool, default=False,
                        help='是否在预测或重建时加入工况数据也作为节点')
    sensor_num, node_num, timestamp_dim = set_node_num(parser.parse_known_args()[0].data_name,
                                                       parser.parse_known_args()[0].Decompose,
                                                       parser.parse_known_args()[0].Wavelet_level,
                                                       parser.parse_known_args()[0].if_timestamp,
                                                       parser.parse_known_args()[0].if_add_work_condition,
                                                       parser.parse_known_args()[0].reco_form)
    parser.add_argument('--sensor_num', type=int, default=sensor_num, help='sensor_num')
    parser.add_argument('--node_num', type=int, default=node_num, help='node_num')
    parser.add_argument('--timestamp_dim', type=int, default=timestamp_dim,
                        help='dataset会另外返回timestamp，存储年、月、天、周等'
                             'BIRDS没写、MSL是6、SMAP是6、'
                             'ETT_h是4、ETT_m是5'
                             'MIC_simulate是1')



    ### 数据
    parser.add_argument('--Dataset', default= 'XJTU_SPS_for_AD_Dataset',
                        help='使用MyDataset文件里编写的哪个Dataset')
    parser.add_argument('--exp_frequency', type=float, default=1, help='实验数据的采样频率，1Hz/0.5Hz/0.1Hz')
    parser.add_argument('--dataset_split_ratio', type=int, default=0.8,
                        help='数据集划分比例，训练集和验证集共同占比总数据集，一般是6：2：2划分的话也就是0.8了')
    parser.add_argument('--dataset_tra_d_val', type=int, default=8,
                        help='训练集和验证集的划分比例，一般是8：1')
    # 滑窗和滑窗步
    parser.add_argument('--batch_size', type=int, default=1, help='input data batch size')##
    parser.add_argument('--lag', type=int, default=96, help='lag时延，即seq_len/slide_win滑窗长/x_channels每层GRU输入通道数')  #
    lag = parser.parse_known_args()[0].lag
    parser.add_argument('--lag_step', type=int, default=1, help='那些数据集的 滑窗在取样本时 几步几步地滑，默认为1，也不用改')  #
    # if TASK == 'forecast':
    parser.add_argument('--label_len', type=int, default=1, help='如果是预测任务，informer那种模型需要个提示段，其他模型不用管')  #
    parser.add_argument('--pred_len', type=int, default=1, help='如果是预测任务，预测几步，默认为1')
    parser.add_argument('--pred_step', type=int, default=24,
                        help='该参数只在预测长度pred_len大于输入时间步长度lag时发挥作用，因为需要多次循环RNN以生成足够长的pred_len'
                             '每次循环预测pred_step时间步，具体来说，pred_len除pred_step向上取整就是循环次数，默认值和lag保持一致，或取lag的一半')
    # 上述三者的设置可以参考https://github.com/thuml/Time-Series-Library/blob/main/run.py
    # 和https://github.com/thuml/Autoformer/blob/main/predict.ipynb
    # 其他一些数据上的设置
    parser.add_argument('--num_workers', type=int, default=2, help='data loader num workers工作进程数，不该是0，慢')
    parser.add_argument('--features', type=str, default='M', help='features [S, M],前者是只用部分传感器的数据，M是用全部')
    parser.add_argument('--target', type=str, default='OT', help='target feature，选择要用的传感器/特征')



    ### 预处理
    parser.add_argument('--scale', type=bool, default=True,
                        help='是否对数据进行标准化归一化,这里为了和预测领域同类方法比较MSE，统一采用了标准化，'
                             '但那些异常检测任务其实说不定归一化效果更好，后续可以调整，但预测的数据集就只能标准化')
    parser.add_argument('--inverse', type=bool, default=False,
                        help='计算test_loss也就是表格里的MSE时是否恢复原尺度，是否逆标准化，'
                             '注意在画图时，不管这里设置什么都会恢复尺度，这个只影响LOSS计算建议False，因为更能反映各通道的预测效果不受尺度影响')
    # parser.add_argument('--preIDW', type=bool, default=True, help='是否对缺失的数据进行IDW插值 的预处理')



    ### 模型

    # 图结构
    parser.add_argument('--self_edge', type=bool, default=True,
                        help='是否使用自连接图')
    parser.add_argument('--graph_ca_meth', type=str, default='MutualInfo',
                        help='MIC：最大信息系数 / Copent：Copent熵 / Cosine：余弦相似度 / MutualInfo：互信息 / Kendall：肯德尔相关系数')
    parser.add_argument('--graph_ca_len', type=int, default=10000,
                        help='用多长数据来计算图，MIC/Cosine/Copent时需要，Prior/Training时不需要')
    parser.add_argument('--graph_ca_thre', type=float, default=0.6,
                        help='图结构的阈值，MIC/Cosine/Copent时需要，Prior/Training时不需要')
    parser.add_argument('--MIC_c', type=float, default=15,
                        help='MIC阈值，MIC时需要，Prior/Training/Cosine/Copent时不需要')
    parser.add_argument('--MIC_alpha', type=float, default=0.6,
                        help='MIC阈值，MIC时需要，Prior/Training/Cosine/Copent时不需要')

    # 架构
    parser.add_argument('--model_selection', type=str, default='SPS_Model_NN',
                        help='历史遗留无用参数、弃用但要保留、设置SPS_Model_NN不要更改。SPS_Model_PINN/SPS_Model_NN/SPS_Model_Phy/SPS_Model_Opt')

    # 各method共享的参数
    parser.add_argument('--SOC_init', type=float, default=0.99,
                        help='仿真开始时，电池初始SOC，默认39.63193175/40=0.99079829375')
    parser.add_argument('--SOC_if_trickle_charge', type=float, default=0.9999999704,
                        help='这是一个非常重要需要不断调试的参数'
                             '电池充电分为快充和涓流充电，当充电SOC达到多少时，就认为从快充转为涓流充电，默认0.99')
    parser.add_argument('--dropout', type=float, default=0.02, help='随即丢弃，默认0/0.1')
    parser.add_argument('--LeakyReLU_slope', type=float, default=1e-3,
                        help='LeakyReLU激活函数的负斜度，默认1e-2，如果0则是ReLU')
    parser.add_argument('--BAT_QU_curve_app_order', type=int, default=3,
                        help='用几次多项式拟合蓄电池的电量-电压曲线，不用改，改了也没用')
    parser.add_argument('--Load_TP_curve_app_order', type=int, default=6,
                        help='用几次多项式拟合负载的温度-功率曲线，就6，别改，改了也没用')

    # if SPS_Model_PINN
    parser.add_argument('--SPS_Model_PINN_if_simplified', type=bool, default=True,
                        help='H_out = H_gnn还是H_out = H_gnn + H_Phy_norm，默认False，就设置False就行')
    parser.add_argument('--if_adapt_denoise', type=bool, default=True,
                        help='物理信息在进入GNN前的adapt环节是否进行平滑')
    parser.add_argument('--how_deploy_phy', type=str, default='sim',
                        help='cal/sim，cal是根据物理公式现场输入数据计算，sim是实现仿真好了直接用仿真数据'
                             '推荐使用sim，当运行sim前得先用SPS_Model_Phy设置save_sim_data为True跑一次，得到仿真数据')

    # if SPS_Model_PINN or SPS_Model_NN
    "if_timestamp已在上面设置，是否在预测或重建时利用上timestamp数据也作为节点"
    "if_add_work_condition已在上面设置，是否在预测或重建时利用上工况数据也作为节点"
            # spatial_block
    parser.add_argument('--spatial_block', type=str, default='G3CN',
                        help='G3CN/GCN/GAT/GIN/SGC/Nothing')
    parser.add_argument('--block_residual', type=float, default=0,
                        help='在空间卷积block之前后，是否 残差连接 的系数，默认0/1就行')
                    # if G3CN
    parser.add_argument('--CMTS_GCN_K_nums', type=list, default=[node_num * 2],
                        help='CMTS_GCN是多层的MAdjGCN，这个是每层的隐藏神经元个数K值组成列表，一般是用不到，一层就够直接用MAdjGCN' \
                             '主要是根据通用近似定理，加深网络深度有益于减轻网络宽度负担，默认[node_num*3, node_num*3]')
    parser.add_argument('--CMTS_GCN_residual', type=float, default=0,
                        help='在CMTS_GCN的layers之间是否连接残差，的系数，默认0')
                    # if GCN_s
    parser.add_argument('--GCN_layer_nums', type=list, default=[lag * 2, lag * 2],
                        help='默认[lag, lag] 两层不变维度，道理同下面TCN_layers_channels，'
                             '***这个不是节点通道的变化，而是时间维度lag的变换'
                             '注意，如果是预测任务，GCN_layer_nums[-1]不能小于pred_len')
                    # if Muti_S_GAT
    parser.add_argument('--Muti_S_GAT_K', type=int, default=2, help='几个GAT头，默认1')
    parser.add_argument('--Muti_S_GAT_embed_dim', type=int, default=192, help='GAT用于计算权重的embed_dim')
    parser.add_argument('--use_gatv2', type=bool, default=True, help='是使用GATV2还是GAT，GATV2是GAT的改进版')
                    # if GIN
    parser.add_argument('--GIN_layer_nums', type=list, default=[lag * 2, lag * 2],
                        help='GIN的 层数 和 特征 的维度变化，默认[GIN_hidden_dim, GIN_hidden_dim, GIN_hidden_dim, ...]，' \
                             '这变的是【lag】那个维度，相当于将时间序列观测信号投影到其他维度的空间上，'
                             '虽然我采用了GIN_layer_nums这种列表方式，支持多层 不同 隐藏层维度，[32,16,1]都是可以的' \
                             '但是根据GIN的源代码https://github.com/weihua916/powerful-gnns/blob/master/models/graphcnn.py，' \
                             '实际使用时没必要搞这么复杂，多层GIN层设置 一样的 数就行了，[lag*N, lag*N, ..]，这也正是原GIN作者的设计')
    parser.add_argument('--GIN_MLP_layer_num', type=int, default=1,
                        help='每层GIN 公式里的MLP的层数，即叠几个上面的hidden_dim，默认1')
                    # if SGC
    parser.add_argument('--SGC_hidden_dim', type=int, default=int(lag * 2),
                        help='SGC的隐藏层维度，默认lag,这变的是lag那个维度，相当于将时间序列观测信号投影到其他维度的空间上')
    parser.add_argument('--SGC_K', type=int, default=3, help='SGC的K值，默认3')

    # if SPS_Model_NN
    parser.add_argument('--temporal_block', type=str, default='TCN',
                        help='MLP/TCN/GRU/Nothing')
            # temporal_block
                    # if MLP
    parser.add_argument('--transf_MLP_hidden_dim', type=int, default=128,
                        help='MLP的隐藏层维度，默认128')
    parser.add_argument('--transf_MLP_layer_num', type=int, default=2,
                        help='MLP的隐藏层层数，默认2')
                    # if TCN
    parser.add_argument('--transf_TCN_num_channels', type=list, default=[node_num, node_num],
                        help='默认[node_num, node_num, node_num]，'
                             '***这个就是节点通道即特征维度node_num在变，时间维度lag不变'
                             '从工况的4维逐步transform到node_num维，'
                             '得是一个列表如[8, 16, 33]，几个数字就有几层TCN，但事实上一层TCN就已经俩次膨胀因果卷积了'
                             '定义了Temporal_block（TCN）的内部层数 和 每层通道数变化结果，'
                             '比如[8, 16, 33]，就是有3层，各层通道数从原来的4依次变化为8变16变33。')
    parser.add_argument('--transf_TCN_kernel_size', type=int, default=2, help='TCN的卷积核大小，默认2')
                    # if GRU
    parser.add_argument('--transf_GRU_hidden_size', type=int, default=64,
                        help='GRU的h的维度，默认就等于传感器数或其倍数，***这变得是节点通道即特征维度node_num那个维度')
    parser.add_argument('--transf_GRU_layers', type=int, default=1, help='how many GRU')
                    # if TCN or GRU
                    # 的最后一次维度变化即TCN_num_channels[-1]或GRU_hidden_size != node_num
                    # 没变成预计的33维度
    parser.add_argument('--temporal_block_end_mlp_hidden_dim', type=int, default=128,
                        help='最后一次维度变化即TCN_num_channels[-1]或GRU_hidden_size不等于node_num时，'
                             '用MLP将最后一维映射到33维，用的MLP的隐藏层维度，默认128')
    parser.add_argument('--temporal_block_end_mlp_layer_num', type=int, default=2,
                        help='最后一次维度变化即TCN_num_channels[-1]或GRU_hidden_size不等于node_num时，'
                             '用MLP将最后一维映射到33维，用的MLP的隐藏层层数，默认2')
            # spatial_block
                    # if MTGNN
    parser.add_argument('--MTGNN_gcn_true', type=bool, default=True, help='是否增加GCN层')
    parser.add_argument('--MTGNN_buildA_true', type=bool, default=True, help='是否构建自适应的A矩阵')
    parser.add_argument('--MTGNN_gcn_depth', type=int, default=2, help='mixprop里面的图卷积深度')
    parser.add_argument('--MTGNN_graph_k', type=int, default=48, help='每个节点几个邻居')
    parser.add_argument('--MTGNN_node_embedding', type=int, default=192, help='节点embedding的维度')
    parser.add_argument('--MTGNN_dilation_exponential', type=int, default=1, help='膨胀指数')
    parser.add_argument('--MTGNN_conv_channels', type=int, default=32,
                        help='TC模块输入维度是residual_channels，输出维度是conv_channels')
    parser.add_argument('--MTGNN_residual_channels', type=int, default=32,
                        help='GC模块输入维度是conv_channels，输出维度是residual_channels')
    parser.add_argument('--MTGNN_skip_channels', type=int, default=64,
                        help='skip_channels是skip connection的输出维度，是图二下半部分的Skip Connection，输入维度是conv_channels，输出维度是skip_channels')
    parser.add_argument('--MTGNN_end_channels', type=int, default=128,
                        help='end_conv_1将skip_channels映射到end_channels，end_conv_2将end_channels映射到下面的out_dim')
    # parser.add_argument('--MTGNN_seq_length', type=int, default=lag,
    #                     help='seq_length是seq_in_len，是输入序列的长度，其实就是lag')
    parser.add_argument('--MTGNN_in_dim', type=int, default=1,
                        help='in_dim是MTGNN特有的那个多出来的维度的通道数，它贴了一层时间标签所以是2，我这时间标签在node维度，所以我这里就1')
    """MTGNN很神奇新增一个维度从1逐步扩充扩增到out_dim也就是pred_len，而不是把lag那个维度慢慢变成pred_len
    之前的lag那个维度只用来进行并行进行这样的计算，并行pred_len维个观测数据映射到1个预测数据
    ，其实也合理，将所有历史数据加权成一个元素单只预测一步，pred_len个维度如此这般加权得到预测pred_len步，太妙了"""
    # parser.add_argument('--MTGNN_out_dim', type=int, default=12,
    #                     help='end_conv_2将end_channels映射到的out_dim，其实就是pred_len或者lag'
    #                          '，对应的是MTGNN增的那个维度最终变成out_dim'
    #                          '，MTGNN很神奇新增一个维度扩增到out_dim也就是pred_len，而不是把lag那个维度慢慢变成pred_len')
    parser.add_argument('--MTGNN_layers', type=int, default=3, help='多少层GC + TC')
    parser.add_argument('--MTGNN_propalpha', type=float, default=0.05,
                        help='Prop alpha，即在混合跳传播中保留根节点原始状态的比例，取值范围在0到1之间')
    parser.add_argument('--MTGNN_tanhalpha', type=float, default=3,
                        help='生成邻接矩阵时的双曲正切alpha值，alpha控制饱和率')
    parser.add_argument('--MTGNN_layer_norm_affline', type=bool, default=True,
                        help='在层归一化中是否进行逐元素仿射操作')

    # if SPS_Model_Phy
    parser.add_argument('--if_save_simulate_result', type=bool, default=False,
                        help='是否保存仿真结果,只在SPS_Model_Phy时，参数定型后，将所有数据仿真一遍，保存仿真结果'
                             '，用于PINN加速训练不再需要batch_size设置为1了有了这个结果')



    ### 消融、对比试验
    # 数据量：只使用一部分数据，用于对比试验证明PINN在数据需求上的优点
    parser.add_argument('--only_use_data_ratio', type=float, default=1,
                        help='只使用数据集的多少比例，用于对比试验证明PINN在数据需求上的优点')
    # 缺失：对数据添加数据缺失，用于对比试验证明PINN在物理指导下的鲁棒性优势
    parser.add_argument('--missing_rate', type=float, default=0.0, help='数据缺失率')  #
    parser.add_argument('--missvalue', default=np.nan, help='一开始人为制造缺失的时候缺失位置补np.nan还是0')
    # 噪声：对数据添加噪声，用于对比试验证明PINN在物理指导下的鲁棒性优势
    parser.add_argument('--add_noise_SNR', type=float, default=0,
                        help='1%、2%、5%、10%、20%、30%、50% 对应的 信噪比SNR的 分贝数分别是：'
                             '20dB、17dB、13dB、10dB、7dB、5.2dB、3dB'
                             '数据添加噪声的信噪比。注意这里用的信噪比不是比例而是dB，因为领域内常用的是dB，'
                             '设置时要注意  https://blog.csdn.net/qq_58860480/article/details/140583800'
                             'https://blog.csdn.net/xiahouzuoxin/article/details/10949887')  #
    parser.add_argument('--add_outliers', type=bool, default=False,
                        help='是否添加异常值，默认False')
    parser.add_argument('--outliers_rate', type=float, default=0.05,
                        help='异常值的比例，默认0.05，表示5%的数据是异常值。outliers = outliers * (np.random.rand(*dirty_data.shape) < self.args.outlier_rate)')
    parser.add_argument('--remove_outliers', type=bool, default=False, help='是否去除异常值')
    parser.add_argument('--preMA', type=bool, default=False, help='是否对含噪数据进行滑动平均 的预处理')
    parser.add_argument('--preMA_win', type=int, default=5, help='滑动平均窗口大小，BIRDS用50，MSL和SMAP用5')
    # 知识：SA和BAT建模知识充足，但BCR多变且可能难建模，因此假设BCR未知，用于对比试验
    parser.add_argument('--SPS_Model_PINN_if_has_Phy_of_BCR', type=bool, default=True,
                        help='SPS_Model_PINN是否有BCR的物理模型，如果没有，那么就是SPS_Model_Phy_wo_BCR'
                             '【注：前提条件要设置使用SPS_Model_PINN】')
    parser.add_argument('--BCR_MLP_hidden_dim', type=int, default=128,
                        help='没有BCR知识时，拟合BCR的MLP的隐藏层维度，默认128')
    parser.add_argument('--BCR_MLP_lay_num', type=int, default=2,
                        help='没有BCR知识时，拟合BCR的MLP的隐藏层层数，默认2')
    # 旧Loss型PINN：SPS_Model_NN是否使用传统PINN的loss，用于对比试验，相当于传统的Loss型PINN
    parser.add_argument('--SPS_Model_NN_if_use_traditional_PINNloss', type=bool, default=False,
                        help='历史遗留无用参数、保留、不需更改'
                        'SPS_Model_NN是否使用传统PINN的loss，用于对比试验，相当于传统的Loss型PINN,【注：前提条件要设置使用SPS_Model_NN】')



    ### 异常分数计算
    parser.add_argument('--AD_threshold', type=float, default=0.5, help='异常分数阈值，误差超过这个值就认为是异常，默认0.5')
    parser.add_argument('--how_precision', type=bool, default=False, help='计算精度时是否精确到具体传感器')
    parser.add_argument('--focus_on', type=str, default='F1', help='F1')
    parser.add_argument('--S_moving_average_window', type=int, default=1,
                        help='对异常分数进行滑动平均，前后几个数字的平均？如果是1，就不滑动平均')



    ### 后处理
    parser.add_argument('--if_plot', type=bool, default=True, help='是否画图')
    parser.add_argument('--if_plot_data_save', type=bool, default=True, help='是否保存画图数据')



    ### 优化器
    parser.add_argument('--gradient_clip_val', type=float, default=100.0,
                        help='梯度裁剪，防止梯度爆炸')
    parser.add_argument('--optimizer', default=torch.optim.Adam, help='使用的优化器，默认torch.optim.Adam')
    parser.add_argument('--lr', default=0.0001, type=float, help='最后决定整个模型同一个学习率')
    parser.add_argument('--scheduler', default='ReduceLROnPlateau', type=str,
                        help='使用哪种学习率衰减策略'
                             'ReduceLROnPlateau是一种动态调整学习率的方法，当某个指标不再变化（下降或升高）时，调整学习率。'
                             'StepLR是一种学习率衰减策略，每隔step_size个epoch，学习率乘以gamma。'
                             'ExponentialLR是一种学习率衰减策略，每个epoch，学习率乘以gamma。'
                             'CosineAnnealingLR是一种学习率衰减策略，学习率按余弦函数下降，到达最低值后又回升。'
                             'CosineAnnealingWarmRestarts是一种学习率衰减策略，学习率按余弦函数下降，到达最低值后又直接回升。')



    ### 训练配置
    parser.add_argument('--random_seed', type=int, default=42, help='random seed')
    parser.add_argument('--patience', type=int, default=20, help='连续10个epoch内loss_A都没有减下去就停止训练')
    parser.add_argument('--max_epoch', type=int, default=500, help='训练多少个epoch')



    ### Ray Tune
    # 超参搜索计划ASHAScheduler, Trial 是一次尝试
    # parser.add_argument('--max_trail', type=int, default=500, help='最大的trail数量，默认为100，但是ray.tune没法设置最大的trail数量，弃了')
    parser.add_argument('--trail_grace_period', type=int, default=40,
                        help='开始减少trial的数量之前要运行的最小trial数量，默认为')
    # parser.add_argument('--trail_time_out', type=int, default=600, help='trial运行的最大时间（秒），默认为600')
    parser.add_argument('--trail_reduction_factor', type=int, default=3,
                        help='reduction_factor表示每次减少trial数量的比例，'
                             '就是说：比如该参数的默认值是3，这意味着在每次trial之后，'
                             '只有三分之一的试验将被保留到下一次trial中，该策略具体'
                             '见https://blog.ml.cmu.edu/2018/12/12/massively-parallel-hyperparameter-optimization/')
    parser.add_argument('--grid_num_samples', type=int, default=100,
                        help='从提供的Search_config中采样多少个参数组，进行多少次trail'
                             '但如果Search_config用的grid而不是choice，那这个就变成网格会被重复多少次搜索，默认是1。'
                             '如果是-1，会无限重复搜索，直到达到停止条件')


    # parser.set_defaults(max_epochs=100)
    args = parser.parse_args()

    return args




"""
定义主函数,单独运行版
"""
def main(devices, args):
    print("请确认传感器数是否是：", args.sensor_num, "请确认节点数目即通道数是否是：", args.node_num)

    datamodule = MyLigDataModule(args)
    # model = torch.compile(MyLigModel(args))
    model = MyLigModel(args)

    trainer = pl.Trainer(strategy="auto",
                         # accelerator="gpu",
                         # devices="auto",
                         # devices=[0, 1, 2],
                         # devices=[0, 3],
                         devices=devices,
                         fast_dev_run=False,
                         max_epochs=args.max_epoch,
                         # callbacks=[pl.callbacks.EarlyStopping(monitor="training_loss", patience=args.patience, check_on_train_epoch_end=True, mode="min")],
                         callbacks=[pl.callbacks.EarlyStopping(monitor='validation_epoch_loss',
                                                               patience=args.patience,
                                                               check_on_train_epoch_end=True,
                                                               mode="min")],
                         # limit_val_batches=0.1,
                         num_sanity_val_steps=0,
                         # deterministic=True,
                         deterministic="warn",
                         default_root_dir=args.ckpt_save_path,
                         check_val_every_n_epoch=1,
                         gradient_clip_val=args.gradient_clip_val,
                         gradient_clip_algorithm='norm',
                         # gradient_clip_algorithm='value',
                         # inference_mode=False,
                         )

    pl.seed_everything(args.random_seed, workers=True)

    trainer.fit(model, datamodule=datamodule)

    trainer.test(model, datamodule=datamodule)

    # trainer.predict(model, datamodule=datamodule)

# "单个main运行"
# if __name__ == '__main__':
#     args = set_args()

#     os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
#     # devices="auto",
#     # devices=[0, 1, 2],
#     # devices=[0, 3],
#     devices = [3]

#     main(devices=devices, args=args)







"""
定义主函数,单独运行版
"""
def main_4_LLM_calling(devices, args, train_or_test="train", ckpt_path_4_test=None, X_predict=None):
    from API_server.AD_api_utils import find_matching_ckpt
    print("请确认传感器数是否是：", args.sensor_num, "请确认节点数目即通道数是否是：", args.node_num)

    datamodule = MyLigDataModule(args)
    # model = torch.compile(MyLigModel(args))

    if train_or_test == "train":
        model = MyLigModel(args)
    else:
        if ckpt_path_4_test is None:
            ckpt_path_4_test = args.ckpt_save_path
        # # 如果ckpt_path_4_test的结尾不是.ckpt文件，则查找ckpt_path_4_test这个目录下所有子文件夹里面的ckpt文件，并选择创建时间最晚的那个
        # if not ckpt_path_4_test.endswith('.ckpt'):
        #     ckpt_files = []
        #     for root, dirs, files in os.walk(ckpt_path_4_test):
        #         for file in files:
        #             if file.endswith('.ckpt'):
        #                 ckpt_files.append(os.path.join(root, file))
        #     if len(ckpt_files) == 0:
        #         raise ValueError("No ckpt file found in {}".format(ckpt_path_4_test))
        #     ckpt_path_4_test = max(ckpt_files, key=os.path.getctime)
        ckpt_path_4_test = find_matching_ckpt(ckpt_root=ckpt_path_4_test, current_args=args)

        from argparse import Namespace
        # torch.serialization.add_safe_globals([Namespace])
        # model = MyLigModel.load_from_checkpoint(ckpt_path_4_test)
        with torch.serialization.safe_globals([argparse.Namespace, torch.optim.Adam]):
            model = MyLigModel.load_from_checkpoint(ckpt_path_4_test)
        # model = MyLigModel.load_from_checkpoint(ckpt_path_4_test, weights_only=False)

    trainer = pl.Trainer(strategy="auto",
                         # accelerator="gpu",
                         # devices="auto",
                         # devices=[0, 1, 2],
                         # devices=[0, 3],
                         devices=devices,
                         fast_dev_run=False,
                         max_epochs=args.max_epoch,
                         # callbacks=[pl.callbacks.EarlyStopping(monitor="training_loss", patience=args.patience, check_on_train_epoch_end=True, mode="min")],
                         callbacks=[pl.callbacks.EarlyStopping(monitor='validation_epoch_loss',
                                                               patience=args.patience,
                                                               check_on_train_epoch_end=True,
                                                               mode="min")],
                         # limit_val_batches=0.1,
                         num_sanity_val_steps=0,
                         # deterministic=True,
                         deterministic="warn",
                         default_root_dir=args.ckpt_save_path,
                         check_val_every_n_epoch=1,
                         gradient_clip_val=args.gradient_clip_val,
                         gradient_clip_algorithm='norm',
                         # gradient_clip_algorithm='value',
                         # inference_mode=False,
                         )

    pl.seed_everything(args.random_seed, workers=True)

    if train_or_test == "train":
        trainer.fit(model, datamodule=datamodule)
    elif train_or_test == "test":
        trainer.test(model, datamodule=datamodule)
    elif  train_or_test == "predict":
        return model(X_predict)
    else:
        raise ValueError("train_or_test must be 'train' or 'test' or 'predict'")
    
    print("Done")







