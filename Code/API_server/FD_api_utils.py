import pandas as pd
import os
from API_server.feature_extraction import *
# from feature_extraction import *
import json
from API_server.Other_api_utils import get_ADfile_path_from_history, get_user_time_from_question
# from Other_api_utils import get_ADfile_path_from_history, get_user_time_from_question



def load_local_finetuned_model(
    if_load_LoRA_or_merged: str = "merged",  # 是否加载LoRA适配器或合并模型
    model_name: str = "Qwen/Qwen2-7B-Chat",  # 模型名称
    max_seq_length: int = 2048,  # 最大序列长度
    dtype: str = "bfloat16",  # 数据类型
    load_in_4bit: bool = True,  # 是否使用4位量化
    cache_dir: str = "./cache",  # 缓存目录
    proxies: dict = None,  # 代理设置
    adapter_path: str = "./checkpoint/adapter",  # LoRA适配器路径
    checkpoint_path: str = "./checkpoint/merged_model",  # 检查点路径
    checkpoint_max_step: int = 75,  # 检查点最大步数
):
    """
    加载模型以供API使用。
    
    参数:
        if_load_LoRA_or_merged (str): lora表示加载基础模型后补上LoRA适配器，merged表示加载合并后的模型，from_checkpoint表示从检查点加载。
        model_name (str): 模型名称。
        max_seq_length (int): 最大序列长度。
        dtype (str): 数据类型。
        load_in_4bit (bool): 是否使用4位量化。
        cache_dir (str): 缓存目录。
        proxies (dict): 代理设置。
        adapter_path (str): LoRA适配器路径。
        checkpoint_path (str): 检查点路径。
        checkpoint_max_step (int): 检查点最大步数。
    """

    from unsloth import FastLanguageModel
    from peft import PeftModel

    if if_load_LoRA_or_merged == "lora":
        # 先加载基础模型
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,  # 使用前面设置的模型名称
            max_seq_length=max_seq_length,  # 使用前面设置的最大长度
            dtype=dtype,  # 使用前面设置的数据类型
            load_in_4bit=load_in_4bit,  # 使用4位量化
            # token="hf_...",  # 如果需要访问授权模型，可以在这里填入密钥
            cache_dir=cache_dir,  # 使用前面设置的缓存目录
            device_map="auto",  # 自动分配设备，以使用GPU或CPU
            # device_map="cuda:2",  # 自动分配设备，以使用GPU或CPU
            local_files_only = True,  # 只使用本地文件，不再重新下载模型
            # proxies=proxies,  # 使用前面设置的代理
            use_safetensors=True,
        )
        # 加载LoRA适配器
        model = PeftModel.from_pretrained(
            model,                  # 基础模型
            adapter_path,           # 适配器路径
            is_trainable=False      # 设置为推理模式
            )
    elif if_load_LoRA_or_merged == "merged":
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,  # 使用前面设置的模型名称
            max_seq_length=max_seq_length,  # 使用前面设置的最大长度
            dtype=dtype,  # 使用前面设置的数据类型
            load_in_4bit=load_in_4bit,  # 使用4位量化
            # token="hf_...",  # 如果需要访问授权模型，可以在这里填入密钥
            # cache_dir=cache_dir,  # 使用前面设置的缓存目录
            device_map="auto",  # 自动分配设备，以使用GPU或CPU
            # device_map="cuda:2",  # 自动分配设备，以使用GPU或CPU
            # local_files_only = True,  # 只使用本地文件，不再重新下载模型
            # proxies=proxies,  # 使用前面设置的代理
            use_safetensors=True,
        )
    elif if_load_LoRA_or_merged == "from_checkpoint":
        # 先加载基础模型
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,  # 使用前面设置的模型名称
            max_seq_length=max_seq_length,  # 使用前面设置的最大长度
            dtype=dtype,  # 使用前面设置的数据类型
            load_in_4bit=load_in_4bit,  # 使用4位量化
            # token="hf_...",  # 如果需要访问授权模型，可以在这里填入密钥
            cache_dir=cache_dir,  # 使用前面设置的缓存目录
            device_map="auto",  # 自动分配设备，以使用GPU或CPU
            # device_map="cuda:2",  # 自动分配设备，以使用GPU或CPU
            local_files_only = True,  # 只使用本地文件，不再重新下载模型
            # proxies=proxies,  # 使用前面设置的代理
            use_safetensors=True,
        )
        # 加载检查点
        FastLanguageModel.for_training(model)
        model = FastLanguageModel.get_peft_model(
            model,  # 传入已经加载好的预训练模型
            r = 16,  # 设置 LoRA 的秩，决定添加的可训练参数数量   ### 一般调试LoRA都是调这个传输，代表保留前多少个奇异值维度的信息
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",  # 指定模型中需要微调的关键模块
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha = 32,  # 设置 LoRA 的超参数，影响可训练参数的训练方式  W' = W + (α/r)·A·B，通常设置r的2倍，常见组合r=16, lora_alpha=32，较小的 lora_alpha：保留更多原始模型知识、较大的 lora_alpha：允许模型更快地适应新任务
            lora_dropout = 0,  # 设置防止过拟合的参数，这里设置为 0 表示不丢弃任何参数
            bias = "none",    # 设置是否添加偏置项，这里设置为 "none" 表示不添加
            use_gradient_checkpointing = "unsloth",  # 使用优化技术节省显存并支持更大的批量大小
            random_state = 3407,  # 设置随机种子，确保每次运行代码时模型的初始化方式相同
            use_rslora = False,  # 设置是否使用 Rank Stabilized LoRA 技术，这里设置为 False 表示不使用
            loftq_config = None,  # 设置是否使用 LoftQ 技术，这里设置为 None 表示不使用
        )
        from datasets import Dataset
        dataset_train = Dataset.from_list([{"text": "这是一个占位符文本，创建一个虚拟dataset_train，因为我根本不是要开启训练而只是导入checkpoint。"}])
        from trl import SFTTrainer  # 导入 SFTTrainer，用于监督式微调
        from transformers import TrainingArguments  # 导入 TrainingArguments，用于设置训练参数
        from unsloth import is_bfloat16_supported  # 导入函数，检查是否支持 bfloat16 数据格式
        # https://huggingface.co/docs/trl/v0.16.1/en/sft_trainer
        trainer = SFTTrainer(  # 创建一个 SFTTrainer 实例
            model=model,  # 传入要微调的模型
            tokenizer=tokenizer,  # 传入 tokenizer，用于处理文本数据
            train_dataset=dataset_train.shuffle(seed=42),  # 传入训练数据集
            dataset_text_field="text",  # 指定数据集中文本字段的名称，也就是说虽然数据集还有'input'啥的特征，但微调其实只用到'text'
            max_seq_length=max_seq_length,  # 设置最大序列长度
            dataset_num_proc=6,  # 设置数据处理的并行进程数
            packing=False,  # 是否启用打包功能（这里设置为 False，打包可以让训练更快，但可能影响效果）
            args=TrainingArguments(  # 定义训练参数
                per_device_train_batch_size=2,  # 每个设备（如 GPU）上的批量大小
                # per_device_train_batch_size=1,  # 每个设备（如 GPU）上的批量大小
                gradient_accumulation_steps=4,  # 梯度累积步数，用于模拟大批次训练
                ##### 公式 1：每个epoch的步数=数据集大小/（每个设备的批量大小*梯度累积步数）   steps_per_epoch = ceil(dataset_size / (per_device_train_batch_size * gradient_accumulation_steps))
                ####### 我的训练集共193664条数据，193664/(24*4)=2018
                # num_train_epochs=10,  # 训练轮数，表示数据集被完整训练的次数。
                num_train_epochs=3,  # 训练轮数，表示数据集被完整训练的次数。
                ##### 公式 2：总步数 如果设置了 num_train_epochs：总步数=每个epoch的步数*epoch数   total_steps = steps_per_epoch * num_train_epochs
                ####### 我2018 * 3 = 6054
                max_steps=checkpoint_max_step,  # 最大训练步数，优先级高于 num_train_epochs。
                ##### 公式 3：总步数 如果设置了 max_steps：总步数=max_steps和总步数中的最小值   total_steps = min(max_steps, steps_per_epoch * num_train_epochs)
                #######  我6052 * 30 = 181560，max_steps=75，所以总步数=75
                lr_scheduler_type="linear",  # 学习率调度器类型，根据训练的进度，从初始学习率线性下降到 0，训练结束时，学习率降为 0。
                warmup_steps=50,  # 预热步数，线性学习率调度器通常会与 warmup steps（预热步数）结合使用。预热阶段的学习率会从 0 增加到初始学习率，然后再开始线性下降。
                # max_grad_norm=1.0,  # 梯度裁剪的最大范数，防止梯度爆炸
                learning_rate=2e-5,  # 学习率，模型学习新知识的速度
                fp16=not is_bfloat16_supported(),  # 是否使用 fp16 格式加速训练（如果环境不支持 bfloat16）
                bf16=is_bfloat16_supported(),  # 是否使用 bfloat16 格式加速训练（如果环境支持）
                logging_steps=1,  # 每隔多少步记录一次训练日志
                optim="adamw_8bit",  # 使用的优化器，用于调整模型参数
                weight_decay=0.001,  # 权重衰减，防止模型过拟合
                seed=3407,  # 随机种子，确保训练结果可复现
                output_dir=checkpoint_path,  # 训练结果保存的目录
                save_strategy = "no",  # 检查点保存策略，这里设置为每步保存一次
                # save_steps = 50,  # 每隔多少步保存一次模型checkpoint,训练可以从断点续训
                report_to="none",  # 是否将训练结果报告到外部工具（如 WandB），这里设置为不报告
            ),
        )
        trainer_stats = trainer.train(resume_from_checkpoint = True)
    else:
        raise ValueError("if_load_LoRA_or_merged must be 'lora' or 'merged'")

    return model, tokenizer






def custom_format(x):
    """自定义格式化函数：
    - |x| > 100: 保留整数
    - 10 <= |x| < 100: 保留一位小数(末尾为0则省略)
    - |x| < 10: 保留两位小数(末尾为00则省略)
    """
    # 获取绝对值和符号
    abs_x = abs(x)
    sign = -1 if x < 0 else 1

    try:
        # 根据范围选择格式化方式
        if np.isnan(x) or np.isinf(x) or x in [None, "None", "nan", "inf", "-inf"]:
            return str(x)
        elif abs_x >= 100:
            # 大于100，四舍五入到整数
            return str(int(round(x)))
        elif abs_x >= 10:
            # 10-100，保留一位小数
            rounded = round(abs_x * 10) / 10 * sign
            # 检查小数部分是否为0
            if rounded == int(rounded):
                return str(int(rounded))
            else:
                return f"{rounded:.1f}"
        else:
            # 0-10，保留两位小数
            rounded = round(abs_x * 100) / 100 * sign

            # 检查小数部分是否为0或0.x0
            if rounded == int(rounded):
                return str(int(rounded))
            elif rounded * 10 == int(rounded * 10):
                return f"{rounded:.1f}"
            else:
                return f"{rounded:.2f}"
    except Exception as e:
        # 如果发生异常，返回原始值
        print(f"Error formatting {x}: {e}")
        return str(x)


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
        format_to_3_decimal = np.vectorize(lambda x: custom_format(x))  # 批量将数值转换为保留三位小数的字符串

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



def from_data_get_feature_str(data_dir, start_time, end_time, lag=8, if_normalized=False):
    """
    SPSPHM_train_json = from_data_get_json(train_data, lag, fault_dict_chinese)

    Args:
        data:
        start_time: 样本开始时间
        end_time: 样本结束时间，没用到
        lag:
        if_normalized: 是否是归一化的数据

    Returns:
        result_str:###指令:...###形容:...###数据:{} ###回答格式：...###回答：

    """
    # 读取数据
    df = pd.read_csv(data_dir, sep=',', index_col=False)
    df['Time'] = pd.to_datetime(df['Time'])

    # 开始时间转为datetime对象
    start_time = pd.to_datetime(start_time)

    # 截取数据
    start_index = df[df['Time'] == start_time].index[0]
    end_index = start_index + lag if start_index + lag < len(df) else len(df)
    sample = df.iloc[start_index:end_index, :]

    # Time列不要了
    sample = sample.drop(columns=['Time'])

    # 表头提取出来，是各个传感器的名称：
    sensor_list = sample.columns.tolist()
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

    # 提取特征
    temporal_feature_dict, frequency_feature_dict, observed_feature_dict \
        = feature_exact_and_str(sample, sensor_dict, sensor_list, if_normalized)
    temporal_feature_str = '\n'.join(
        ['各传感器的' + key + temporal_feature_dict[key] for key in temporal_feature_dict.keys()]
    )
    frequency_feature_str = '\n'.join(
        ['各传感器的' + key + frequency_feature_dict[key] for key in frequency_feature_dict.keys()]
    )
    observed_feature_str = '\n'.join(
        [key + '传感器监测数据为: ' + observed_feature_dict[key] for key in observed_feature_dict.keys()]
    )
    input_information = "各传感器监测数据如下（数列元素对应各时间步）：" + '\n' \
                        + observed_feature_str + '\n' \
                        + "各传感器提取的时域特征如下（数列元素对应各传感器）：" + '\n' \
                        + temporal_feature_str + '\n' \
                        + "各传感器提取的频域特征如下（数列元素对应各传感器）：" + '\n' \
                        + frequency_feature_str

    question_style = """
###指令:请基于所提供数据，告诉我发生了异常类型候选集中的哪一个异常？异常类型候选集：[太阳能电池板部分元件或支路开路,太阳能电池板部分元件或支路短路,太阳能电池板元件老化,BCR开路,BCR短路,BCR功率损耗增加,电池组老化,电池组开路,母线开路,母线短路,母线绝缘击穿,PDM1开路或短路,PDM2开路或短路,PDM3开路或短路,负载1开路,负载2开路,负载3开路,其他]
###形容:该段数据采集于航天器电源系统，其由太阳能电池板、3组蓄电池组、3路负载、充电控制器BCR、母线和功率分配模块PDM等子系统和部件组成。
###数据:
{}
###回答格式：直接输出判断结果，从异常类型候选集中选择且只能选择一种进行输出。
###回答：
"""
    result_str = question_style.format(input_information)

    # print(result_str)
    return result_str





def response_fault_localization(User_prompt, User_question, RAG_materials, data,
                                temperature, top_p, 
                                stream_mode, 
                                model, tokenizer,
                                fault_dict_chinese,
                                second_question_style,
                                second_question_if_use_fintuned = True,
                                max_new_tokens = 64,
                                second_question_max_new_tokens = 2048,
                                ):
    """
    获取微调模型生成的答案，并进行二次提问，返回reponse，包括判断的异常类型，以及判断标准
    如果效果不好，可以考虑进行二次提问是使用更大参数量的模型。

    :param User_prompt: 用户的提示
    :param User_question: 用户提问的问题
    :param RAG_materials: RAG材料
    :param data: 历史对话记录，用来提取异常检测的文件路径
    :param temperature: 模型生成的温度
    :param top_p: 模型生成的top_p
    :param stream_mode: 是否使用流式输出，已弃用、不生效
    :param model: 加载好的模型
    :param tokenizer: 模型对应的分词器
    :param fault_dict_chinese: 故障类型字典，包含中文故障类型
    :param second_question_style: 二次提问的格式化字符串
    :param second_question_if_use_fintuned: 是否使用微调模型进行二次提问,若为True，则使用微调模型，否则使用网上的大模型进行二次提问的响应
    :param max_new_tokens: 第一次生成的最大新token数量，就是个故障类型，有64足够
    :param second_question_max_new_tokens: 第二次提问生成的最大新token数量，这个要多一点，2048

    :return: 返回生成的响应
    """
    # 用户的提问有两种可能：
    # 1. User_question是一个完整的带有数据的：###指令:请基于所提供数据，。。。###数据: 提取特征和原始数据如下：
    # 2. User_question是一个不带数据的：“那么你所检测出的<2024/10/18  18:42:01 - 2024/10/18  18:43:01>时间段内，发生了什么类型的故障？ ”
    # 对第一种，prompt直接用User_question
    # 对第二种，需要从User_prompt中提取出路径、起始时间，并从本地读取
    if "###数据" in User_question or "### 数据" in User_question or "### Instruction" in User_question or "###Instruction" in User_question:
        prompt = User_question.split("## My question is:")[1].strip()
    else:
        # 提取路径
        csv_path = get_ADfile_path_from_history(data)
        # 提取起始时间和结束时间
        # 定义正则表达式，匹配横杠前的时间字符串，假定用户提问形式是<2024/10/18  18:42:01 - 2024/10/18  18:43:01>
        start_time, end_time = get_user_time_from_question(User_question)
        prompt = from_data_get_feature_str(data_dir=csv_path, start_time=start_time, end_time=end_time, lag=8, if_normalized=False)


    # 4. 添加更多生成参数和超时控制
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)    
    # outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        # temperature=0.9,
        # top_p=0.9,
        # do_sample=True,  # 启用采样以应用温度和top_p
        do_sample=False,  # 第一个问题回答不需要应用温度和top_p，直接根据事实回答
        pad_token_id=tokenizer.eos_token_id,  # 确保正确的填充
        use_cache=True,
    )
    
    # 5. 处理输出，移除提示部分
    # response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prompt_length = inputs.input_ids.shape[1]
    # generated_tokens = outputs[0][:]
    generated_tokens = outputs[0][prompt_length:]  # 生成的回答总是倾向于复述一次问题，去掉
    response = tokenizer.decode(generated_tokens, skip_special_tokens=False)   # skip_special_tokens=True 移除所有特殊标记包括：[PAD], [CLS], [SEP], [MASK], <s>, </s>, <eos> 等
    # print(type(response))
    # 如果是列表输入而不只是单个
    # response = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
    # 如果使用以上代码，则返回时要返回response[0]而不是response
    # # response 有时后面会跟一些有的没的，提取出来有用的
    for FD_type in fault_dict_chinese.keys():
        if FD_type in response:
            response = FD_type
    print('response_firstFD\n', response, '\n response_firstFD_end')
    # 检查生成的文本是否为空
    if not response.strip():
        print("警告: 生成的响应为空!")
        response = "警告: 生成的响应为空!"
    FD_type_result = response.strip()
    
    # 6. 进行提问
    # 先提取出 '###数据：' 和  其后面出现的第一个'###'之间的内容
    if "###数据：" in prompt:
        sample_data_text = prompt.split("###数据：", 1)[1].split("###")[0].strip()
    elif "###数据:" in prompt:
        sample_data_text = prompt.split("###数据:", 1)[1].split("###")[0].strip()
    elif "### 数据：" in prompt:
        sample_data_text = prompt.split("### 数据：", 1)[1].split("###")[0].strip()
    elif "### 数据:" in prompt:
        sample_data_text = prompt.split("### 数据:", 1)[1].split("###")[0].strip()
    else:
        sample_data_text = ""
        print("警告: 未找到 '###数据：' 或 '### 数据：'!")
    # sample_data_text = prompt.split("###数据：")[1].split("###回答格式")[0].strip()
    message2 = [
        {"role": "user", "content": second_question_style.format(response, sample_data_text, response)}
        ]
    prompt2 = tokenizer.apply_chat_template(message2,   # 交流消息列表
                                            tokenize = False,   # 返回格式化后的文本字符串，而不进行 ID 的标记（tokens）
                                            add_generation_prompt = True,   # 是否在模板末尾添加生成提示符
                                            enable_thinking = True,   # 让其思考
                                            )
    print('prompt2:\n', prompt2)

    if second_question_if_use_fintuned:
        input2 = tokenizer(prompt2, return_tensors="pt").to(model.device)  # 将输入转换为张量
        outputs2 = model.generate(
            input_ids=input2.input_ids,
            attention_mask=input2.attention_mask,
            max_new_tokens=second_question_max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,  # 启用采样以应用温度和top_p
            pad_token_id=tokenizer.eos_token_id,  # 确保正确的填充
            use_cache=True,
        )
        response2 = tokenizer.decode(outputs2[0], skip_special_tokens=True)   # skip_special_tokens=True 移除所有特殊标记包括：[PAD], [CLS], [SEP], [MASK], <s>, </s>, <eos> 等
        response2 = response2.strip()
        # print(response2)
    else:
        return sample_data_text, FD_type_result

    # 7. 拼接并更新reponse
    # 首先把response2中的“assistant”前面的问题复述去掉
    if "assistant" in response2:
        response2 = response2.split("assistant", 1)[-1].strip()
    # 然后如果检测到<think>和</think>，则将其放在前面
    if "<think>" in response2 and "</think>" in response2:
        response_think = "<think>" + response2.split("<think>", 1)[-1].split("</think>", 1)[0].strip() + "</think>"
        response_answer = response2.split("</think>", 1)[-1].strip()
        response = response_think + '\n' + response_answer
    else:
        response = response2
        # response = "经过分析，异常/故障类型为{}，分析依据为{}。".format(response, response2)
    print('response_secondFD\n', response2, '\n response_secondFD_end')

    # 8. 返回最终的响应
    return response, FD_type_result








