import re
import pandas as pd
from datetime import datetime
from API_server.Other_api_utils import get_ADfile_path_from_history, get_user_time_from_question




def get_MD_data_from_question_or_history(question, data):
    """
    用户执行工况识别任务时，有三种可能的提问方式：
    1. 用户直接提问，“请执行工况识别任务。需要分析的各传感器数据如下：。。。”
    2. 用户提问，“既然你检测到<2024/10/18  18:42:01 - 2024/10/18  18:43:01>这段时间发生了异常，那么异常发生之前航天器正处于什么工况？请执行工况识别任务。”
    3. 用户提问，“在 /data/xxxx/xx_Test.csv 文件中，<2024/10/18  18:42:01 - 2024/10/18  18:43:01>这段时间（之前/之后）航天器正处于什么工况？请执行工况识别任务。”

    对于第一种，直接将question返回就得了。
    对于第二种，就需要从data的历史对话信息中，从后到前遍历每个'role'为'user'的对话记录，从中提取出含有test或者Test的csv文件的路径，然后读取、截取用户指定的段落
    对于第三种，直接从中提取出含有test或者Test的csv文件的路径，然后读取、截取用户指定的段落
    """
    # 返回处理后的数据
    sensor_str_list = ['太阳能电池板电压数据为:', '太阳能电池板电流数据为:', '太阳能电池板功率数据为:', 
                        '负载总电压数据为:', '负载总电流数据为:', '负载总功率数据为:', 
                        'BCR电压数据为:', 'BCR电流数据为:', 'BCR功率数据为:', 
                        '电池组2电压数据为:',  '电池组2电流数据为:',  '电池组2温度数据为:', 
                        '电池组3电压数据为:', '电池组3电流数据为:', '电池组3温度数据为:', 
                        '电池组4电压数据为:', '电池组4电流数据为:', '电池组4温度数据为:', 
                        '母线电压数据为:', '母线电流数据为:', '母线功率数据为:', 
                        '负载1电压数据为:', '负载1电流数据为:', '负载1温度数据为:', '负载1功率数据为:',
                        '负载2电压数据为:', '负载2电流数据为:', '负载2温度数据为:', '负载2功率数据为:',
                        '负载3电压数据为:', '负载3电流数据为:', '负载3温度数据为:', '负载3功率数据为:']

    if '数据如下' in question or '数据为' in question or '数据是' in question or '数据:' in question or '数据：' in question:
        return question
    else:
        csv_path_may_in_question = get_ADfile_path_from_history(question)
        if not "File Path not Provided" in csv_path_may_in_question:
            # 用户直接在question中提供了csv文件路径
            csv_path = csv_path_may_in_question
        else:
            # 从data的历史对话信息中，查找最后一个'role'为'user'的对话记录
            if 'prompt' in data:
                # 原始格式 - 直接使用提示
                raise NotImplementedError("Original prompt format not supported. Please use 'messages' format.")
            elif 'messages' in data:
                # Cherry Studio/OpenAI格式 - 处理对话历史
                csv_path = get_ADfile_path_from_history(data)
        print(f"CSV file path: {csv_path}")

        # 读取csv文件并截取用户指定的段落
        try:
            df = pd.read_csv(csv_path, sep=',', index_col=False)
        except FileNotFoundError:
            return f"CSV file path not be provided or found: {csv_path}"

        # 定义正则表达式，匹配横杠前的时间字符串，假定用户提问形式是<2024/10/18  18:42:01 - 2024/10/18  18:43:01>
        start_time, end_time = get_user_time_from_question(question)
        # 转化为时间对象
        # start_time = datetime.strptime(start_time_str, "%Y/%m/%d %H:%M:%S")
        start_time = pd.to_datetime(start_time)
        # end_time = datetime.strptime(end_time_str, "%Y/%m/%d %H:%M:%S")
        end_time = pd.to_datetime(end_time)

        # 以下是一行数据的处理方法
        # df['Time'] = pd.to_datetime(df['Time'])
        # # 如果question里面有‘之前’、‘之后’等字样，则需要截取start_time之前或者之后 一步 的数据, 否则只截取start_time对应的数据
        # if '之前' in question or 'before' in question:
        #     matched_index = df[df['Time'] == start_time].index
        #     row = df.iloc[matched_index[0] - 3] if matched_index[0] - 3 >= 0 else df.iloc[0]
        # elif '之后' in question or 'after' in question:
        #     matched_index = df[df['Time'] == end_time].index
        #     row = df.iloc[matched_index[0] + 3] if matched_index[0] + 3 < len(df) else df.iloc[-1]
        # else:
        #     matched_index = df[df['Time'] == start_time].index
        #     row = df.iloc[matched_index[0]]

        # data_str_list = row.apply(str).tolist()

        # 上面的截取数据只截取一行，不够，应该截三行
            # 以下是多行数据的处理方法
        df['Time'] = pd.to_datetime(df['Time'])
        # 如果question里面有‘之前’、‘之后’等字样，则需要截取start_time之前或者之后 一步 的数据, 否则只截取start_time对应的数据
        if '之前' in question or 'before' in question:
            matched_index = df[df['Time'] == start_time].index
            row = df.iloc[matched_index[0]-6: matched_index[0]-3] if matched_index[0]-6 >= 0 else df.iloc[0:3]
        elif '之后' in question or 'after' in question:
            matched_index = df[df['Time'] == end_time].index
            row = df.iloc[matched_index[0]+3: matched_index[0]+6] if matched_index[0]+6 < len(df) else df.iloc[-3:]
        else:
            matched_index = df[df['Time'] == start_time].index
            row = df.iloc[matched_index[0]-1: matched_index[0]+2] if matched_index[0]-1 >= 0 and matched_index[0]+2 < len(df) else (df.iloc[0:3] if matched_index[0] < 2 else df.iloc[-3:])
        
        data_str_list = row.astype(str).agg(','.join, axis=0).tolist()

        data_str_list = data_str_list if len(data_str_list) == len(sensor_str_list) else data_str_list[1:]
        result_str = "请执行工况识别任务。需要分析的各传感器数据如下：" + ";\n".join([f"{sensor}{data}" for sensor, data in zip(sensor_str_list, data_str_list)])
        
        return result_str


# 太阳能电池板电压是否在0到35V之间？太阳能电池板电流是否在0到2.7A之间？太阳能电池板功率是否在0到70W之间？# 负载总电压是否在0到13V之间？负载总电流是否在0到3A之间？负载总功率是否在0到27W之间？# BCR电压是否在16到17V之间？BCR电流是否在0到3.5A之间？BCR功率是否在0到58W之间？# 各组电池组电压是否在16到17V之间？各组电池组电流是否在-0.7到+0.5A之间？各组电池组温度是否在20到30摄氏度之间？# 母线电压是否在16到17V之间？母线电流是否在0到2A之间？母线功率是否在0到32W之间？# 负载1电压是否在11到13V之间？负载1电流是否在0到3A之间？负载1温度是否在-20到+250摄氏度之间？负载1功率是否在0到30W之间？# 负载2电压是否在4到6V之间？负载2电流是否在0到3A之间？负载2温度是否在-20到+250摄氏度之间？负载2功率是否在0到16W之间？# 负载3电压是否在4到6V之间？负载3电流是否在0到3A之间？负载3温度是否在-20到+250摄氏度之间？负载3功率是否在0到16W之间？


# 电池组2电压是否在16到17V之间？电池组2电流是否在-3到3A之间？电池组2温度是否在-10到50摄氏度之间？
# 电池组3电压是否在16到17V之间？电池组3电流是否在-3到3A之间？电池组3温度是否在-10到50摄氏度之间？
# 电池组4电压是否在16到17V之间？电池组4电流是否在-3到3A之间？电池组4温度是否在-10到50摄氏度之间？




