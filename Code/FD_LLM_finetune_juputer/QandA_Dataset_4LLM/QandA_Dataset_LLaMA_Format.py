import pandas as pd
from MSL_SMAP.plot import *
import os
from XJTU_SPS.XJTU_SPS_Dataset.QandA_Dataset_4LLM.feature_extraction import *
import json



fault_dict = {
    0: 'normal',
    1: 'SA_partial component or branch open circuit',
    2: 'SA_partial component or branch short circuit',
    3: 'SA_component deterioration',
    4: 'BCR_open circuit',
    5: 'BCR_short circuit',
    6: 'BCR_increased power loss',
    7: 'BAT_degradation',
    8: 'BAT_open circuit',
    9: 'Bus_open circuit',
    10: 'Bus_short circuit',
    11: 'Bus_insulation breakdown',
    12: 'PDM1_open circuit or short circuit',
    13: 'PDM2_open circuit or short circuit',
    14: 'PDM3_open circuit or short circuit',
    15: 'Load1_open circuit',
    16: 'Load2_open circuit',
    17: 'Load3_open circuit'
}

fault_dict_chinese = {
    0: '正常',
    1: '太阳能电池板部分元件或支路开路',
    2: '太阳能电池板部分元件或支路短路',
    3: '太阳能电池板元件老化',
    4: 'BCR开路',
    5: 'BCR短路',
    6: 'BCR功率损耗增加',
    7: '电池组老化',
    8: '电池组开路',
    9: '母线开路',
    10: '母线短路',
    11: '母线绝缘击穿',
    12: 'PDM1开路或短路',
    13: 'PDM2开路或短路',
    14: 'PDM3开路或短路',
    15: '负载1开路',
    16: '负载2开路',
    17: '负载3开路'
}

read_path = 'E:\OneDrive\DATA\Our_Exp_Data\XJTU-SPS Dataset\original data'
read_file_name = '2024.9.18_normal\\2024.9.18_normal.csv'
save_path = 'E:\OneDrive\DATA\Our_Exp_Data\XJTU-SPS Dataset\QandA_Dataset_4LLM\LLaMA_Format'
save_file_name = 'SPS_PHM.json'
# 如果文件夹不存在则创建
if not os.path.exists(os.path.join(save_path, save_file_name)):
    # os.makedirs(os.path.join(save_path, save_file_name))
    os.makedirs(os.path.join(save_path))



"""
使用json文件创建问答数据集,一般有Alpcaca和ShareGPT两种格式,我们这里使用Alpcaca格式: 
SPSPHM_json = [
    {
        "instruction": "人类指令（必填）",
        "input": "人类输入（选填）",
        "output": "模型回答（必填）",
        "system": "系统提示词(选填),
        "domain": "领域标签(选填),
    },
    {
        "instruction": "分析以下症状可能的疾病",
        "input": "患者持续发热3天,伴有咳嗽和胸痛",
        "output": "可能诊断为肺炎,建议进行胸部X光检查",
        "system": "请根据症状提供可能的疾病诊断",
        "domain": "医疗",
    },
    ...
]
"""
SPSPHM_train_json = []
SPSPHM_test_json = []
SPSPHM_normalized_train_json = []
SPSPHM_normalized_test_json = []


"""
一些重要的参数
"""
lag = 8
# 每次对话输入的时间序列的长度



def feature_exact_and_str(sample, sensor_dict, sensor_list, if_normalized):
    """

    Args:
        sample: pd.DataFrame, shape (lag, n_features)
        sensor_dict: dict, {传感器名称: 中文释义}
        sensor_list: list, ['U_SA', 'I_SA', 'P_SA', 'U_Load_output', 'I_Load_output', 'P_Load_output',...]
        if_normalized: bool, 是否是归一化的数据

    Returns:

    """
    temporal_feature_dict = temporal_feature_extract(sample)
    frequency_feature_dict = frequency_feature_extract(sample)
    observed_feature_dict = {sensor_dict[sensor]: sample[sensor].values for sensor in sensor_list}

    # 字典里面那些numpy全部转化为str，每个数字之间用,连接
    if not if_normalized:
        # 如果是非归一化的数据，原始数据不用更改，特征数据保留三位小数
        format_to_3_decimal = np.vectorize(lambda x: f"{x:.3f}")  # 批量将数值转换为保留三位小数的字符串

        for key in observed_feature_dict.keys():
            observed_feature_dict[key] = format_to_3_decimal(observed_feature_dict[key])
            # observed_feature_dict[key] = observed_feature_dict[key].tolist()
        for key in temporal_feature_dict.keys():
            temporal_feature_dict[key] = format_to_3_decimal(temporal_feature_dict[key])
        for key in frequency_feature_dict.keys():
            frequency_feature_dict[key] = format_to_3_decimal(frequency_feature_dict[key])
    else:
        # 如果是归一化的数据，全部*1000并保留成整数
        format_to_1000int = np.vectorize(lambda x: f"{round(x * 1000)}" if not np.isnan(x) else x) # 批量将数值转换为整数的字符串

        for key in observed_feature_dict.keys():
            observed_feature_dict[key] = format_to_1000int(observed_feature_dict[key])
            # observed_feature_dict[key] = observed_feature_dict[key].tolist()
        for key in temporal_feature_dict.keys():
            temporal_feature_dict[key] = format_to_1000int(temporal_feature_dict[key])
        for key in frequency_feature_dict.keys():
            frequency_feature_dict[key] = format_to_1000int(frequency_feature_dict[key])

    for key in temporal_feature_dict.keys():
        temporal_feature_dict[key] = ','.join(np.char.mod('%s', temporal_feature_dict[key]))
    for key in frequency_feature_dict.keys():
        frequency_feature_dict[key] = ','.join(np.char.mod('%s', frequency_feature_dict[key]))
    for key in observed_feature_dict.keys():
        observed_feature_dict[key] = ','.join(np.char.mod('%s', observed_feature_dict[key]))

    return temporal_feature_dict, frequency_feature_dict, observed_feature_dict



def from_data_get_json(data, i, JSON_output, lag, fault_dict, sensor_dict, sensor_list, if_normalized):
    """
    SPSPHM_train_json = from_data_get_json(train_data, lag, fault_dict_chinese)

    Args:
        data:
        i: 第i种故障
        JSON_output: 要把处理好的字典加入哪个JSON
        lag:
        fault_dict:
        sensor_dict:
        sensor_list:
        if_normalized: 是否是归一化的数据

    Returns:

    """
    for j in range(len(data) - lag):
        sample = data.iloc[j:j + lag]

        temporal_feature_dict, frequency_feature_dict, observed_feature_dict \
            = feature_exact_and_str(sample, sensor_dict, sensor_list, if_normalized)
        temporal_feature_str = '\n'.join(
            [key + '特征为: ' + temporal_feature_dict[key] for key in temporal_feature_dict.keys()]
        )
        frequency_feature_str = '\n'.join(
            [key + '特征为: ' + frequency_feature_dict[key] for key in frequency_feature_dict.keys()]
        )
        observed_feature_str = '\n'.join(
            [key + '数据为: ' + observed_feature_dict[key] for key in observed_feature_dict.keys()]
        )
        input_information = temporal_feature_str + '\n' + frequency_feature_str + '\n' + observed_feature_str

        sample_json = {
            "instruction": "请为我执行异常定位任务。即基于我提供的信息，分析得到异常发生的位置和类型。"
                           "\n已知该段数据采集于一个航天器电源系统，其由太阳能电池板、3组蓄电池组、3路负载、"
                           "充电控制器BCR、母线和功率分配模块PDM等子系统和部件组成。"
                           "\n而且根据工程经验，可能的异常发生位置有：太阳能电池板部分元件或支路开路、"
                           "太阳能电池板部分元件或支路短路、太阳能电池板元件老化、BCR开路、BCR短路、BCR功率损耗增加、"
                           "电池组老化、电池组开路、母线开路、母线短路、母线绝缘击穿、PDM1开路或短路、PDM2开路或短路、"
                           "PDM3开路或短路、负载1开路、负载2开路、负载3开路等。",
                           # "\n 请遵循以下格式回答问题，替换或补充[]中的内容："
                           # "output: 经过对数据的分析，异常定位结果为：[太阳能电池板部分元件或支路开路 或 "
                           # "太阳能电池板部分元件或支路短路 或 太阳能电池板元件老化 或 BCR开路 或 BCR短路 或 "
                           # "BCR功率损耗增加 或 电池组老化 或 电池组开路 或 母线开路 或 母线短路 或 母线绝缘击穿 或 "
                           # "PDM1开路或短路 或 PDM2开路或短路 或 PDM3开路或短路 或 负载1开路 或 负载2开路 或 负载3开路 或 "
                           # "其他]，补充信息：[得到结果的思考分析]。",
                            # "The data were collected from a spacecraft power system comprising subsystems and "
                            # "components such as solar panels, three battery groups, three loads, a battery charge regulator (BCR), "
                            # "a bus, and power distribution modules (PDMs)."
                            # "Based on engineering experience, potential fault locations include: open circuits in certain solar "
                            # "panel components or branches, short circuits in solar panel components or branches, solar panel "
                            # "component aging, BCR open circuit, BCR short circuit, increased BCR power loss, battery group aging, "
                            # "battery group open circuit, bus open circuit, bus short circuit, bus insulation breakdown, open or short "
                            # "circuits in PDM1, PDM2, or PDM3, and open circuits in Load 1, Load 2, or Load 3."
            "input": "提取特征和原始数据如下：" + '\n' + input_information,
            "output": fault_dict[i],
            "system": "请进行异常定位",
            "domain": "工业系统智能运维",
        }
        JSON_output.append(sample_json)

    return JSON_output



def from_csv_get_json(json_list, read_path, if_train, if_normalized):
    """

    Args:
        json_list: 现在正在处理哪个json文件，往哪个列表中加问答字典
        read_path: 读数据的路径
        if_train: 是否正在处理训练的json文件，即SPSPHM_train_json和SPSPHM_normalized_train_json
        if_normalized: 是否真正处理归一化的json文件，即SPSPHM_normalized_train_json和SPSPHM_normalized_test_json

    Returns:

    """
    """开始处理数据"""
    for i in range(1, len(fault_dict)):
        read_file_name = fault_dict[i] + '\\' + fault_dict[i] + '.csv'

        # 读取csv文件
        data = pd.read_csv(os.path.join(read_path, read_file_name), sep=',', index_col=False)
        data['Time'] = pd.to_datetime(data['Time'])
        anomaly_label = pd.read_csv(os.path.join(read_path, read_file_name.replace('.csv', '_AnomalyLabel.csv')),
                                                    sep=',', index_col=False)
        # anomaly_label里面['AnomalyLabel']为1的  对应data里面的异常数据 提取出来
        fault_data = data[anomaly_label['AnomalyLabel'] == 1]
        # 周期数
        period_num = int(round((fault_data['Time'].iloc[-1] - fault_data['Time'].iloc[0]).total_seconds() / 60 / 95))
        # 如果有4个周期，只取后三个周期
        if period_num == 4:
            fault_data = fault_data[fault_data['Time'] >= fault_data['Time'].iloc[0] + pd.Timedelta(minutes=95)]

        # 时间列不要了
        fault_data = fault_data.drop(columns=['Time'])
        # 表头提取出来，是各个传感器的名称：
        sensor_list = fault_data.columns.tolist()
        # 替换成中文释义并描述：
        # U_SA	I_SA	P_SA	U_Load_output	I_Load_output	P_Load_output	U_BCR	I_BCR	P_BCR
        # U_BAT2	I_BAT2	T_BAT2	U_BAT3	I_BAT3	T_BAT3	U_BAT4	I_BAT4	T_BAT4
        # U_Bus	I_Bus	P_Bus	U_Load1	I_Load1	T_Load1	P_Load1	U_Load2	I_Load2	T_Load2	P_Load2
        # U_Load3	I_Load3	T_Load3	P_Load3
        sensor_dict = {'U_SA': '太阳能电池板电压', 'I_SA': '太阳能电池板电流', 'P_SA': '太阳能电池板功率',
                       'U_Load_output': '负载总电压', 'I_Load_output': '负载总电流', 'P_Load_output': '负载总功率',
                       'U_BCR': 'BCR电压', 'I_BCR': 'BCR电流', 'P_BCR': 'BCR功率',
                         'U_BAT2': '电池组2电压', 'I_BAT2': '电池组2电流', 'T_BAT2': '电池组2温度',
                         'U_BAT3': '电池组3电压', 'I_BAT3': '电池组3电流', 'T_BAT3': '电池组3温度',
                            'U_BAT4': '电池组4电压', 'I_BAT4': '电池组4电流', 'T_BAT4': '电池组4温度',
                         'U_Bus': '母线电压', 'I_Bus': '母线电流', 'P_Bus': '母线功率',
                            'U_Load1': '负载1电压', 'I_Load1': '负载1电流', 'T_Load1': '负载1温度', 'P_Load1': '负载1功率',
                            'U_Load2': '负载2电压', 'I_Load2': '负载2电流', 'T_Load2': '负载2温度', 'P_Load2': '负载2功率',
                            'U_Load3': '负载3电压', 'I_Load3': '负载3电流', 'T_Load3': '负载3温度', 'P_Load3': '负载3功率'}
        sensor_name_list = [sensor_dict[sensor] for sensor in sensor_list]

        # 2/3的周期作为训练集，1/3的周期作为测试集
        # train_data = fault_data.iloc[:period_num * 2 // 3 * 95 * 60]
        # test_data = fault_data.iloc[period_num * 2 // 3 * 95 * 60:]
        if if_train:
            output_data = fault_data.iloc[:period_num * 2 // 3 * 95 * 60]
        else:
            output_data = fault_data.iloc[period_num * 2 // 3 * 95 * 60:]
        # 归一化到-1到1之间的fault_data
        # normalized_train_data = (train_data - train_data.min()) / (train_data.max() - train_data.min()) * 2 - 1
        # normalized_test_data = (test_data - test_data.min()) / (test_data.max() - test_data.min()) * 2 - 1
        if if_normalized:
            output_data = (output_data - output_data.min()) / (output_data.max() - output_data.min()) * 2 - 1

        # 变成json，加入json文件
        json_list = from_data_get_json(output_data, i, json_list, lag, fault_dict_chinese, sensor_dict, sensor_list, if_normalized)

    return json_list


# 保存json文件
SPSPHM_train_json = from_csv_get_json(SPSPHM_train_json, read_path, if_train=True, if_normalized=False)
with open(os.path.join(save_path, save_file_name.replace('.json', '_Train.json')), 'w', encoding='utf-8') as f:
    json.dump(SPSPHM_train_json, f, ensure_ascii=False, indent=4)
SPSPHM_train_json = []    # 保存后清除缓存

SPSPHM_test_json = from_csv_get_json(SPSPHM_test_json, read_path, if_train=False, if_normalized=False)
with open(os.path.join(save_path, save_file_name.replace('.json', '_Test.json')), 'w', encoding='utf-8') as f:
    json.dump(SPSPHM_test_json, f, ensure_ascii=False, indent=4)
SPSPHM_test_json = []    # 保存后清除缓存

SPSPHM_normalized_train_json = from_csv_get_json(SPSPHM_normalized_train_json, read_path, if_train=True, if_normalized=True)
with open(os.path.join(save_path, save_file_name.replace('.json', '_normalized_Train.json')), 'w', encoding='utf-8') as f:
    json.dump(SPSPHM_normalized_train_json, f, ensure_ascii=False, indent=4)
SPSPHM_normalized_train_json = []    # 保存后清除缓存

SPSPHM_normalized_test_json = from_csv_get_json(SPSPHM_normalized_test_json, read_path, if_train=False, if_normalized=True)
with open(os.path.join(save_path, save_file_name.replace('.json', '_normalized_Test.json')), 'w', encoding='utf-8') as f:
    json.dump(SPSPHM_normalized_test_json, f, ensure_ascii=False, indent=4)
SPSPHM_normalized_test_json = []    # 保存后清除缓存




    # # 保存为csv文件
    # train_data.to_csv(os.path.join(save_path, save_file_name.replace('.csv', '_Train.csv')), index=False)
    # test_data.to_csv(os.path.join(save_path, save_file_name.replace('.csv', '_Test.csv')), index=False)
    # # 保存为pkl文件
    # train_data.to_pickle(os.path.join(save_path, save_file_name.replace('.csv', '_Train.pkl')))
    # test_data.to_pickle(os.path.join(save_path, save_file_name.replace('.csv', '_Test.pkl')))
    # # 画图PDF
    # channel_plot(plot_dirname_path=os.path.join(save_path, save_file_name.replace('.csv', '_Train.pdf')),
    #              data=train_data,
    #              channel_list=train_data.columns.tolist(),
    #              data_all_label=None,
    #              fig_size=(6, 1.5))
    # channel_plot(plot_dirname_path=os.path.join(save_path, save_file_name.replace('.csv', '_Test.pdf')),
    #              data=test_data,
    #              channel_list=test_data.columns.tolist(),
    #              data_all_label=None,
    #              fig_size=(6, 1.5))


print('done')













