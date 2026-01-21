# %% [markdown]
"""# ## 设置参数"""

# %%
import sys
from pathlib import Path
# get the parent directory of the current file's parent directory (project root's parent)
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "7"
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
# os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"

os.environ["HF_HUB_BASE_URL"] = "https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 设置Hugging Face 镜像站 URL
os.environ["HF_HUB_CACHE"] = r"/data/DiYi/MyWorks_Results/DY-LLM_4SPS-PHM_Project/LLM"  # 缓存路径

# # vLLM 可能识别的环境变量，没用。。。
# os.environ["VLLM_USE_V1"] = '0'
# os.environ["VLLM_HF_HUB_BASE_URL"] = "https://hf-mirror.com"
# os.environ["VLLM_DISABLE_ONLINE_CHECKS"] = "1"  # 尝试禁用在线检查
# os.environ['CURL_CA_BUNDLE'] = ''
# 上面这句是为了告诉底层的 cURL 库不使用任何 CA 证书进行 SSL/TLS 验证，因为如果启用fast_inference = True, vllm仍然会尝试连接hugging face的官网而并非镜像站，会报错MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443)。因此加这一句使其不验证SSL证书，避免了这个问题。https://github.com/huggingface/transformers/issues/17611

"""确保 os.environ['CUDA_VISIBLE_DEVICES'] = "2" 在任何 CUDA 初始化代码（如 import torch 或 from transformers import ...）之前设置，
否则没用，某些库（如 torch 或 transformers）可能已经先初始化了 CUDA 环境，导致 CUDA_VISIBLE_DEVICES 设置无效"""
"""Make sure os.environ['CUDA_VISIBLE_DEVICES'] = "2" is set before any CUDA initialization code (like import torch or from transformers import ...),
otherwise it won't work, as some libraries (like torch or transformers) may have already initialized the CUDA environment, making the CUDA_VISIBLE_DEVICES setting ineffective.
"""

import requests
response = requests.get("https://hf-mirror.com")   # 测试访问 Hugging Face # Test access to Hugging Face
print(response.status_code)  # 如果返回 200，说明代理生效 # if return 200, it means the proxy is effective

import sys, os
sys.path.append(os.path.abspath('..'))  # 加入上级目录 # Add parent directory

import json
import time

# %%
# if_load_LoRA_or_merged = "merged" # 加载合并后的微调模型 # Load merged fine-tuned model
if_load_LoRA_or_merged = "lora" # 加载训练权重到基础模型 # Load training weights to base model
# if_load_LoRA_or_merged = "from_checkpoint"  # 从检查点加载模型 # Load model from checkpoint


if if_load_LoRA_or_merged == "lora":
    # model_name = r'/data/DiYi/MyWorks_Results/DY-LLM_4SPS-PHM_Project/LLM/ModelScope_unsloth_DeepSeek-R1-Distill-Qwen-14B-bnb-4bit'  # 指定要加载的模型名称 # Specify the model name to load
    model_name = r"/data/DiYi/MyWorks_Results/DY-LLM_4SPS-PHM_Project/LLM/ModelScope_Qwen3-14B-unsloth-bnb-4bit"  # 指定要加载的模型名称 # Specify the model name to load
    adapter_path = r"/data/DiYi/MyWorks_Results/DY-LLM_4SPS-PHM_Project/finetuned_adapter_LLMmodel_HF"
    checkpoint_path = None
    checkpoint_max_step = None
elif if_load_LoRA_or_merged == "merged":
    model_name = r"/data/DiYi/MyWorks_Results/DY-LLM_4SPS-PHM_Project/finetuned_LLMmodel_HF"
    adapter_path = None
    checkpoint_path = None
    checkpoint_max_step = None
elif if_load_LoRA_or_merged == "from_checkpoint":
    model_name = r"/data/DiYi/MyWorks_Results/DY-LLM_4SPS-PHM_Project/LLM/ModelScope_Qwen3-14B-unsloth-bnb-4bit"
    adapter_path = None
    checkpoint_path = r"/data/DiYi/MyWorks_Results/DY-LLM_4SPS-PHM_Project/checkpoint_2025.5.18"
    checkpoint_max_step = max([int(d.split('-')[1]) for d in os.listdir(checkpoint_path) if d.startswith('checkpoint-') and d.split('-')[1].isdigit()])
else:
    raise ValueError("if_load_LoRA_or_merged must be 'lora' or 'merged'")
# model_name = "unsloth/Qwen3-14B",
# model_name="unsloth/DeepSeek-R1-Distill-Llama-8B",


cache_dir = r"/data/DiYi/MyWorks_Results/DY-LLM_4SPS-PHM_Project/LLM"
FD_first_question_max_seq_length = 3584  # 设置模型处理文本的最大长度，输入多长，相当于给模型设置一个“最大容量” # Set the maximum length of text the model can process, equivalent to setting a "maximum capacity" for the model
# max_seq_length = 2048  # 设置模型处理文本的最大长度，输入多长，相当于给模型设置一个“最大容量” # Set the maximum length of text the model can process, equivalent to setting a "maximum capacity" for the model
# max_seq_length = 3072  # 设置模型处理文本的最大长度，输入多长，相当于给模型设置一个“最大容量” # Set the maximum length of text the model can process, equivalent to setting a "maximum capacity" for the model
# max_seq_length = 4096  # 设置模型处理文本的最大长度，输入多长，相当于给模型设置一个“最大容量” # Set the maximum length of text the model can process, equivalent to setting a "maximum capacity" for the model
FD_first_question_max_new_tokens = 64  # 设置模型生成文本的最大长度，输出多长 # Set the maximum length of text the model can generate, equivalent to setting a "maximum output length"
FD_second_question_max_new_tokens = 2048
FD_second_question_if_use_fintuned = False

temperature_0 = 0.7  # 默认的模型生成温度，模型生成文本的随机程度。值越大，回复内容越赋有多样性、创造性、随机性；设为0根据事实回答。日常聊天建议设置为0.7 # Default model generation temperature, the randomness of the model's generated text. The larger the value, the more diverse, creative, and random the response content; set to 0 for factual answers. For daily chat, it is recommended to set it to 0.7
top_p_0 =  0.9  # 默认的模型生成文本top_p，默认值为1，值越小，A生成的内容越单调，也越容易理解：值越大，A回复的词汇围越大，越多样化 # Default model generation text top_p, default value is 1. The smaller the value, the more monotonous the generated content A is, and the easier it is to understand: the larger the value, the larger the vocabulary range of A's reply, and the more diverse it is


dtype = None  # 设置数据类型，让模型自动选择最适合的精度 # Set data type to let the model automatically choose the most suitable precision
load_in_4bit = True  # 使用4位量化来节省内存，就像把大箱子压缩成小箱子 # Use 4-bit quantization to save memory


# 云模型调用
yun_model_Scope = 'aliyun'
yun_model_api_id = "https://dashscope.aliyuncs.com/compatible-mode/v1/"
# yun_model_api_key = os.getenv("DASHSCOPE_API_KEY")  # 替换为你的API密钥 # Replace with your API key
with open('/home/dy29/MyWorks_Codes/15_LLM_4_SPS_PHM/API_server/api_keys.json', 'r') as file:
    yun_model_api_key = json.load(file)[yun_model_Scope]
# # yun_model_Scope = 'ModelScope'
# yun_model_api_id = "https://api-inference.modelscope.cn/v1/"
# # yun_model_api_key = os.getenv("DASHSCOPE_API_KEY")  # 替换为你的API密钥 # Replace with your API key
# with open('api_keys.json', 'r') as file:
#     yun_model_api_key = json.load(file)[yun_model_Scope]
# # yun_model_Scope = 'SiliconFlow'
# yun_model_api_id = "https://api.siliconflow.cn/v1/chat/completions"
# # yun_model_api_key = os.getenv("DASHSCOPE_API_KEY")  # 替换为你的API密钥 # Replace with your API key
# with open('api_keys.json', 'r') as file:
#     yun_model_api_key = json.load(file)[yun_model_Scope]

# # yun_model_if_stream = True  # 是否使用流式输出 这个不在这里设置，是在客户端设置好后，传输过来取用传输值 # Whether to use streaming output This is not set here, it is set on the client side and then passed over for use
yun_max_seq_length = 4058
yun_max_new_tokens = 2048

yun_model_name = "qwen3-235b-a22b"
# yun_model_name = "qwen-plus-latest"
# yun_model_name = "deepseek-r1"


# 是否手动总结健康管理报告，若为'manual'，则手动总结报告；若为'yun'，则使用云模型总结报告 # Whether to manually summarize the health management report, if 'manual', then manually summarize the report; if 'yun', then use the cloud model to summarize the report
export_report_by_manual_or_yun = 'yun'

# 为了保证总结时信息完整，客户端的设置里面一定要设置上下文数目拉满！！！ # To ensure the completeness of information during summarization, the number of contexts in the client settings must be set to the maximum!!!
# 但是在除了总结的其他任务，不需要用到那么多记忆，因此，设置 记忆对话轮数： # But for other tasks besides summarization, so much memory is not needed, so set the number of memory conversation turns:
memory_conv_turns = 1  # 记忆对话轮数，默认只保留1轮对话的记忆，节省token # Number of memory conversation turns, by default only 1 turn of conversation memory is retained to save tokens
# memory_conv_turns = 3  # 记忆对话轮数，默认只保留3轮对话的记忆，节省token # Number of memory conversation turns, by default only 3 turns of conversation memory is retained to save tokens
# 是以轮数为单位的，即一次user和assistant的对话算一轮 # It is measured in turns, that is, one user and assistant conversation counts as one turn


# %%
fault_dict_chinese = {
    # '正常': 0,
    'SA_部分元件或支路开路': 1,
    'SA_部分元件或支路短路': 2,
    'SA_元件老化': 3,
    'BCR开路': 4,
    'BCR短路': 5,
    'BCR功率损耗增加': 6,
    '电池组老化': 7,
    '电池组开路': 8,
    '母线开路': 9,
    '母线短路': 10,
    '母线绝缘击穿': 11,
    'PDM1开路或短路': 12,
    'PDM2开路或短路': 13,
    'PDM3开路或短路': 14,
    '负载1开路': 15,
    '负载2开路': 16,
    '负载3开路': 17
    # 0: '正常',
    # 1: '太阳能电池板部分元件或支路开路',
    # 2: '太阳能电池板部分元件或支路短路',
    # 3: '太阳能电池板元件老化',
    # 4: 'BCR开路',
    # 5: 'BCR短路',
    # 6: 'BCR功率损耗增加',
    # 7: '电池组老化',
    # 8: '电池组开路',
    # 9: '母线开路',
    # 10: '母线短路',
    # 11: '母线绝缘击穿',
    # 12: 'PDM1开路或短路',
    # 13: 'PDM2开路或短路',
    # 14: 'PDM3开路或短路',
    # 15: '负载1开路',
    # 16: '负载2开路',
    # 17: '负载3开路'
}
# fault label information：
# 0: 正常: normal
# 1: SA部分部件或支路断路: SA_partial component or branch open circuit
# 2: SA部分部件或支路短路: SA_partial component or branch short circuit
# 3: SA部件老化: SA_component deterioration
# 4: BCR断路: BCR_open circuit
# 5: BCR短路: BCR_short circuit
# 6: BCR损耗增大: BCR_increased power loss
# 7: BAT退化: BAT_degradation
# 8: BAT断路: BAT_open circuit
# 9: 母线断路: Bus_open circuit
# 10: 母线短路: Bus_short circuit
# 11: 母线绝缘破损: Bus_insulation breakdown
# 12: PDM1断路或短路: PDM1_open circuit or short circuit
# 13: PDM2断路或短路: PDM2_open circuit or short circuit
# 14: PDM3断路或短路: PDM3_open circuit or short circuit
# 15: Load1断路: Load1_open circuit
# 16: Load2断路: Load2_open circuit
# 17: Load3断路: Load3_open circuit



if FD_second_question_if_use_fintuned == False:
    second_question_style = """
###指令:下面这段数据及其特征被你判断为发生了[{}]故障，请你分析这段数据，说明你为什么做出这个判断？请给出你的分析依据。
###形容:该段数据采集于航天器电源系统，其由太阳能电池板SA、3组蓄电池组BAT、3路负载Load、充电控制器BCR、母线和功率分配模块PDM等子系统和部件组成。
###数据:
{}
###基础设计信息：该型航天器相关设计信息可供你参考：
1.太阳能电池板正常工作时，若位于日照区，其电压应大于10V，电流应为正数，表示正在放电，在0到2.7A之间，放电功率在0到70W之间；若位于地影区，其电压、电流、功率应接近0。
2.BCR正常工作时，其电压应在16到17V之间，电流为正数表示正在给电池充电，或者电流接近0表示无电可充，总体电流范围在0到3.5A之间，功率应在0到58W之间。
3.各组电池组正常工作时，其电压应在16到17V之间，电流应在-0.7到+0.5A之间，注意：电池组电流为负数表示电池组正在充电，正数表示正在放电，此处符号容易混淆。
4.母线正常工作时，其电压应稳定在16到17V之间，电流为正数表示正在给负载供能，或者电流接近0表示负载不需供能，总体电流范围在0到2A之间，功率应在0到32W之间。
5.各路负载正常工作时，其电压、 电流、温度和功率均应在设计范围内，正数电流表示正在工作，接近0表示没在工作。具体来讲：负载1的电压应在11到13V之间，电流应在0到3A之间，温度应在-20到+250摄氏度之间，功率应在0到30W之间；负载2的电压应在4到6V之间，电流应在0到3A之间，温度应在-20到+250摄氏度之间，功率应在0到16W之间；负载3的电压应在4到6V之间，电流应在0到3A之间，温度应在-20到+250摄氏度之间，功率应在0到16W之间。
###检查反驳项：一般来讲，最频发的故障类型包括：[太阳能电池板部分元件或支路开路,太阳能电池板部分元件或支路短路,太阳能电池板元件老化,BCR开路,BCR短路,BCR功率损耗增加,电池组老化,电池组开路,母线开路,母线短路,母线绝缘击穿,PDM1开路或短路,PDM2开路或短路,PDM3开路或短路,负载1开路,负载2开路,负载3开路,其他]，我们告诉你的故障定位结果类型可能是其中一种也可能不是，但如果你在提供详细的分析依据的时候，顺便检查一下，说明为什么不是上述其他故障类型，涵盖一些反驳依据在分析依据中，会让你的分析更有说服力。
###回答格式：请严格遵循以下格式进行回答：“该段数据发生的故障类型为{}，分析依据为<>。”。其中<>为分析依据，自行思考进行填写。
""" 
else: 
    second_question_style = """
###指令:下面这段数据及其特征被你判断为发生了[{}]故障，请你分析这段数据，说明你为什么做出这个判断？请给出你的分析依据。
###形容:该段数据采集于航天器电源系统，其由太阳能电池板、3组蓄电池组、3路负载、充电控制器BCR、母线和功率分配模块PDM等子系统和部件组成。
###数据:
{}
###参考信息：1.太阳能电池板电流应为正数表示正在生电。2.BCR电流为正数表示正在给电池充电。3.注意：电池组电流为负数表示电池组正在充电，正数表示正在放电。4.母线电流为正数表示正在给负载供能。5.各路负载电流为正数表示正在工作。
###回答格式：请严格遵循以下格式进行回答：“该段数据发生的故障类型为{}，分析依据为<>。”。其中<>为分析依据，自行思考进行填写。
"""



# %%
MD_prompt_style = """
## 任务：你是一个航天器电源系统运行维护专家，下面内容是执行工况识别任务的分析步骤。如果用户正在进行健康管理并关心某段数据的工况信息，用户可能向你输入一段数据，请你按照下面的分析方法和回答格式对用户提供的传感器监测数据进行工况识别。

## 相关知识：已知该段数据采集于一个航天器电源系统，其由太阳能电池板、3组蓄电池组、3路负载、充电控制器BCR 、放电调节器BDR、分流调节器SR 、母线和功率分配模块PDM等子系统或部件组成。

## 分析方法：请根据以下步骤进行分析：
***步骤1：确认太阳能电池板电压是否大于1V？若是，记录为日照区，转入步骤2；若否，记录为地影区，转入步骤6；
***步骤2：确认各路负载电流是否大于0.5A？若是，第<N>路负载<N>电流大于0.5A则记录为正在执行任务<N>，转入步骤3；若否，记录为无任务，转入步骤4；
***步骤3：确认负载<N>功率是否大于太阳能电池板功率？若是，输出工作状态为【联合供电、日照区、任务<N> 】，转入步骤7；若否，输出工作状态为【分流、日照区、任务<N> 】，转入步骤7；
***步骤4：确认各组电池电流值是否小于-0.1A？若是，转入步骤5；若否，记录为涓流充电，输出工作状态为【涓流充电、日照区、无任务】，转入步骤7；
***步骤5：确认电池组2的电压 的几个值之间 前后增幅是否大于0.05V？若是，记录为CC充电，输出工作状态为【CC充电、日照区、无任务】，转入步骤7；若否，记录为CV充电，输出工作状态为【CV充电、日照区、无任务】，转入步骤7；
***步骤6：确认各路负载电流是否大于0.5A？若是，第<N>路负载<N>电流大于0.5A则记录为正在执行任务<N>，输出工作状态为【放电、地影区、任务<N>】，转入步骤7；若否，记录为无任务，输出工作状态为【空闲、地影区、无任务】，转入步骤7；
***步骤7：根据所输出工作状态检查各传感器观测值，# 太阳能电池板电压是否在0到35V之间？太阳能电池板电流是否在0到2.7A之间？太阳能电池板功率是否在0到70W之间？# 负载总电压是否在0到13V之间？负载总电流是否在0到3A之间？负载总功率是否在0到27W之间？# BCR电压是否在16到17V之间？BCR电流是否在0到3.5A之间？BCR功率是否在0到58W之间？# 各组电池组电压是否在16到17V之间？各组电池组电流是否在-0.7到+0.5A之间？各组电池组温度是否在20到30摄氏度之间？# 母线电压是否在16到17V之间？母线电流是否在0到2A之间？母线功率是否在0到32W之间？# 负载1电压是否在11到13V之间？负载1电流是否在0到3A之间？负载1温度是否在-20到+250摄氏度之间？负载1功率是否在0到30W之间？# 负载2电压是否在4到6V之间？负载2电流是否在0到3A之间？负载2温度是否在-20到+250摄氏度之间？负载2功率是否在0到16W之间？# 负载3电压是否在4到6V之间？负载3电流是否在0到3A之间？负载3温度是否在-20到+250摄氏度之间？负载3功率是否在0到16W之间？

## 回答格式：你回答问题的格式必须遵照如下模板，替换填写[]和<>中的内容：
“步骤1：确认[确认内容]，[确认过程推理] ，答案为[是/否]，记录为 [日照区/地影区]，转入步骤<X>；步骤2：确认 [确认内容]，[确认过程推理] ，答案为[是/否]，记录为[正在执行任务<N>/无任务]，转入步骤<X>；步骤3：确认 [确认内容]，[确认过程推理] ，答案为[是/否]，输出工作状态为[…、…、…]，转入步骤7；步骤4：确认 [确认内容]，[确认过程推理] ，答案为[是/否]，[记录为涓流充电/输出工作状态为...]，转入步骤<X>；步骤5：确认 [确认内容]，[确认过程推理] ，答案为[是/否]，记录为[CC充电/CV充电]，输出工作状态为[…、…、…]，转入步骤7；步骤6：确认 [确认内容]，[确认过程推理] ，答案为[是/否]，记录为[正在执行任务<N>/无任务]，输出工作状态为[…、…、…]，转入步骤7；步骤7：根据所输出工作状态检查各传感器观测值：[检查内容]，[检查过程推理] ，记录为[检查正常/检查不通过，结果不可信]； 综上，该航天器目前的工作状态为：[输出工作状态]。”

## 注意：1.当你对用户提供的步骤进行分析时，若在数据段中同时出现“大于”和“小于”的情况，则说明该时段可能发生了工况切换。请你将该段划分为“切换前”和“切换后”两个阶段（若发生多次切换，则依次划分为多个阶段），分别在每个阶段依据上述给出的步骤进行分析，并为每个阶段输出独立的回答。 2.虽然用户往往关心的只是正常数据段的工况，大部分情况下输入的都是正常数据，但若你发现不符合正常工作模式的特征，仍需报警。3.若步骤7太长且没什么异常情况可以直接说明“检查正常”，不需要每一个传感器都写一遍，若有异常情况则需要详细说明。
"""

# *****下面可以提供几个经典样本示例的，但是我们不需要few-shot，zero-shot压力测试更能体现性能：*****
# ## 回答示例：同时，为你提供一些正确回答的示例可供你参考：
# “<经典样本示例1>”、“<经典样本示例2>”、“<经典样本示例3>”……

# 这个style是用于执行MR任务时的system prompt
# 用户提问时不需要每次都输入这么长一段，这一段添加入system prompt，已集成到系统后台
# 而User prompt可以仅仅是“请执行工况识别任务。需要分析的各传感器数据如下：”
# 也可能是“既然你检测到<T1-T2>这段时间发生了异常，那么异常发生之前航天器正处于什么工况？请执行工况识别任务。”
# 也可能是“XXX时间航天器正处于什么工况？”

# %%
RCA_prompt_style = """
## 任务：用户正在进行异常或故障的根因分析及维护决策任务，他将发送给你航天器电源系统发生的故障类型和具体细节，由你来查询用户的知识库文件夹，为用户完成根因分析、风险评判、严重程度定级、辅助决策和检修策略推荐等任务。

## 背景：已知运维对象为航天器电源系统，其由太阳能电池板（负责将光能转为电能）、3组蓄电池组（负责储存电能）、3路负载（负责消耗电能和执行任务）、充电调节器BCR（负责调节电池充电、放电、分流）、放电调节器BDR、分流调节器SR、母线（负责传输电能）和功率分配模块PDM（负责分配电能到各路负载）等子系统或部件组成。

## 回答格式：你回答问题的格式最好能遵照如下模板，替换填写补充 [] 和 <> 和...中的内容，可以进行适当的改写和拓展，若时间或国家等信息位置可以不写：
“步骤1—根因分析：查询往年归零文档、相关论文、技术报告与设计文档等，总结可能导致[当前异常或故障类型]发生的原因有：（1）[高能粒子撞击/帆板卡死/...]，即[相关详细的解释、严谨地分析等]、 （2）[可能的原因]，即[相关详细的解释、严谨地分析等]、 （3）...、 （4）...、 （5）...、 （6）...、 （7）...、 （8）...；
步骤2—风险评判：历史案例显示，该故障可能导致以下后果：（1）[xxx后果]：[相关详细的解释、严谨地分析等]，比如<N>年 [xxx]国[xxx]航天器发生了[xxx]，导致[xxx后果]...、（2）[xxx后果]：[相关详细的解释、严谨地分析等]，比如<N>年 [xxx]国[xxx]航天器发生了[xxx]，导致[xxx后果]...、 （3）...、 （4）...、 （5）...、 （6）...、 （7）...、 （8）...。
总之，结合相关论文、归零报告、技术文件，可总结该类型故障可能导致的后果有：[后果一xxx、后果二xxx、后果三xxx、后果四xxx、后果五xxx、后果六xxx、...]，
因此，建议将该故障定档为[灾难级/严重/一般/轻微]；
步骤3—检修策略：查询检修策略文件夹、专家咨询留档、相关论文和技术报告等，总结该故障可能的检修策略包括：(1)[系统重启...]、(2)[供配电策略切换...]、(3)[启动冗余备份单元...]、(4)...、(5)...、(6)...、(7)...、(8)...；
步骤4—提示用户：点击搜索工具执行结果的放大镜图标，或将鼠标悬停于各条信息对应的引用图标数字标识上方，即可预览源参考材料的内容及其访问路径。

## 注意：内容要详细，在填写补充 [] 和 <> 和...中的内容时每一个要多写几段，不要吝惜token消耗，但是在[]和<>和...之外不要更改！也不要补充多余的内容。
"""

# 回答太长了，cherry studio自带参考文献预览功能，不需要列出sourceUrl或是source路径了，太消耗有限的token，token主要用来丰富回答内容
# 步骤4：以上回答参考文献访问路径：< [1]完整的sourceUrl或是source路径, ... , [n]... >。”
# ## 注意：1.参考文献复述列出其完整的sourceUrl或是source路径，不要擅自删减，也不要列出具体指向的内容；2.内容要详细，在填写补充 [] 和 <> 和...中的内容时每一个要多写几段，不要吝惜token消耗。

## 任务：我们已经定位到当前发生的是< 太阳能电池板部分元件或支路开路 / 太阳能电池板部分元件或支路短路 / 太阳能电池板元件老化 / BCR开路 / BCR短路 / BCR功率损耗增加 / 电池组老化 / 电池组开路 / 母线开路 / 母线短路 / 母线绝缘击穿 / 第<N> 路PDM开路或短路 //第<N> 路负载开路 ...>故障，表现为<...>。请你通过查询、分析我的知识库文件夹，为我完成根因分析、风险评判、严重程度定级、辅助决策和检修策略推荐等任务。





# %% [markdown]
"""# ## 导入模型 （预训练好的本地模型）"""

# %%
from API_server.FD_api_utils import load_local_finetuned_model
# from FD_api_utils import load_local_finetuned_model
from unsloth import FastLanguageModel
from peft import PeftModel
FD_model, FD_tokenizer = None, None
"""
使用方法：
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
"""








# %% [markdown]
"""# # 调用模型"""

# %%
from API_server.Other_api_utils import get_response_from_aliyunAPI

"""
使用方法：https://help.aliyun.com/zh/model-studio/use-qwen-by-calling-api?spm=a2c4g.11186623.0.0.bdd41d1cky7RPB#a75fcbc1dchyc

先定义  messages=[{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': '你是谁？'}]
调用get_response_from_aliyunAPI： completion = get_response_from_aliyunAPI(*****)

"qwen3-235b-a22b" 
1.普通：直接通过.model_dump_json()获取json格式的response 打印complection.model_dump_json()
2.流式：for循环获取chunk： for chunk in complection得到chunk, 然后逐个得到chunk.model_dump_json()，可以for循环打印chunk.model_dump_json()

"deepseek-r1"
1.普通：通过reasoning_content字段打印思考过程print(completion.choices[0].message.reasoning_content)；# 通过content字段打印最终答案print(completion.choices[0].message.content)
2.流式：for循环获取chunk： for chunk in complection得到chunk, chunk.choices[0].delta里面包含.reasoning_content和.content，有时当前chunk中content为空，可能正在思考中，就先打印chunk.choices[0].delta.reasoning_content，等到content不为空时，才打印chunk.choices[0].delta.content
"""








# %% [markdown]
"""# # 意图识别"""

# %%
Task_Now = 'Normal_QA'
# 意图：普通问答、异常检测、工况识别、故障定位、根因分析及维护决策
Task_Label_Dict = {
    "请执行普通问答任务": 'Normal_QA',
    "Please execute the normal question and answer task": 'Normal_QA',
    "请执行异常检测任务": 'Anomaly_Detection',
    "Please execute the anomaly detection task": 'Anomaly_Detection',
    "请执行工况识别任务": 'Work_Condition_Recognition',
    "Please execute the work condition recognition task": 'Work_Condition_Recognition',
    "请执行故障定位任务": 'Fault_Localization',
    "Please execute the fault localization task": 'Fault_Localization',
    "请执行根因分析及维护决策任务": 'RCA_and_MDM',
    "Please execute the root cause analysis and maintenance decision-making task": 'RCA_and_MDM',
    "请导出健康管理报告": 'Export_Health_Management_Report',
    "Please export the health management report": 'Export_Health_Management_Report'
}
# Tool_Now = {}  # 当前使用的工具，默认为无工具
# Tool_already_id = []
User_prompt = None
User_question = None
RAG_materials = None





# %% [markdown]
"""# # 启动api服务"""

# %% [markdown]
# ##### 端口监听

# %%
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
import time
import uuid
import json
from threading import Thread
import re
# import multiprocessing
# from multiprocessing import Process, Queue, Manager
# multiprocessing.set_start_method('spawn', force=True)


app = Flask(__name__) # 创建Flask应用实例
CORS(app)  # 启用CORS支持


# 把 /SPS_AD_LLM_Project/<path:filename> 映射到磁盘上的 /data/SPS_AD_LLM_Project/<path:filename>
# 这样的话，在电脑客户端markdown里面访问：
# ![异常检测结果图片](http://<服务器IP>:1999/SPS_AD_LLM_Project/reports/2025.5.18/2025.5.18-1.png)
# 就可以直接访问到服务器上的data/DiYi/MyWorks_Results/SPS_AD_LLM_Project/reports/2025.5.18/2025.5.18-1.png这个文件了。
app.add_url_rule(
    '/SPS_AD_LLM_Project/<path:filename>',
    endpoint='SPS_AD_LLM_Project',                    # 内部端点名
    view_func=lambda filename: send_from_directory(
        '/data/DiYi/MyWorks_Results/SPS_AD_LLM_Project', filename
    )
)


# @app.route('/v1/chat/completions', methods=['POST'])  # 定义一个路由，监听POST请求
# @app.route('/generate', methods=['POST'])  # 定义一个路由，监听POST请求
# def chat_completions():
@app.route('/generate/v1/chat/completions', methods=['POST'])  # 定义一个路由，监听POST请求
def generate():
    
    global Task_Now  # 声明 Task_Now 为全局变量
    global User_prompt, User_question, RAG_materials

    # 1. 添加请求数据验证
    data = request.json
    print(data)
    # 处理不同格式的请求
    prompt = ""
    if 'prompt' in data:
        # 原始格式 - 直接使用提示
        cherry_prompt = []
        history = []
        prompt = data['prompt']
    elif 'messages' in data:
        # Cherry Studio/OpenAI格式 - 处理对话历史
        messages = data.get('messages', [])
        # # 合并所有消息成一个提示文本
        # for message in messages:
        #     role = message.get('role', '')
        #     content = message.get('content', '')
        #     if content:  # 只处理有内容的消息
        #         prompt += f"{role}: {content}\n"
        cherry_prompt = [messages[0]] # Cherry Studio自动生成的提示，好像是system prompt，如果tool调用效果不佳，可以把cherry_prompt也加入messages，因为Cherry Studio自动生成的system提示里面有很多很好的tool调用教程
        history = messages[1:-1]  # 保留之前的消息历史
        prompt = messages[-1].get('content', '') # 只选择最后一条消息作为提示
    else:
        return jsonify({"error": "Unsupported request format"}), 400


    # 2. 提取客户端设定的大模型model生成参数
    # print(data)
    # max_new_tokens = data.get('max_new_tokens', max_new_tokens_0)
    temperature = data.get('temperature', temperature_0)
    top_p = data.get('top_p', top_p_0)
    stream_mode = data.get('stream', False)  # 提问时开启流式，那回应时就必须以流式的格式返回，提问时不开启流式，那回应时必须以非流式的格式直接返回



    # 3.意图识别
    # 先提取用户问题
    """如果您使用的是Cherry Studio v1.2.4或者更早版本，RAG的材料会直接包含在prompt中，采用以下代码进行RAG材料的分割："""
    # if "## Reference Materials:" in prompt:
    #     if "## My question is:" in prompt:
    #         User_question = "## My question is:\n\n" + prompt.split("## My question is:")[1].split("## Reference Materials:")[0].strip()
    #     else:
    #         User_question = "## My question is:\n\n" + prompt.split("## Reference Materials:")[0].strip()
    # else:
    #     if "## My question is:" in prompt:
    #         User_question = "## My question is:\n\n" + prompt.split("## My question is:")[1].strip()
    #     else:
    #         User_question = prompt.strip()
    """但是，如果您使用的是Cherry Studio v1.6.2或者更新版本，RAG的材料得要发去XML请求，才会进行搜索："""
    if ("<tool_use_result>" in prompt) and ("builtin_knowledge_search" in prompt):
        User_question = history[-2].get('content', '').strip()  # history里面倒数第二条的content就是原始提问
    else:
        User_question = prompt.strip()

    messages= [{'role': 'system', 'content': '你是一个能识别用户意图的机器人，从[Normal_QA, Anomaly_Detection, Work_Condition_Recognition, Fault_Localization, RCA_and_MDM, Export_Health_Management_Report]中选择一个输出，不要输出任何多余信息。'},
              {'role': 'user', 'content': f'用户的问题是：{User_question}，请识别用户的意图。若用户希望进行普通问答(比如询问一些普通问题)，则输出"Normal_QA"；若用户希望进行异常检测(比如询问可供选择的算法、询问需要设置的参数、要求设置或修改参数、要求对某数据执行异常检测模型的训练、调用、测试、检测)，则输出"Anomaly_Detection"；若用户希望进行工况识别(比如询问处于什么工况)，则输出"Work_Condition_Recognition"；若用户希望进行故障定位(比如询问发生了什么种类或者类型的故障、询问哪里发生了故障)，则输出"Fault_Localization"；若用户希望进行根因分析及维护决策(比如询问根因分析、可能的成因、风险定级、危害等级、维护决策蹬专业知识)，则输出"RCA_and_MDM"；若用户希望导出健康管理报告，则输出"Export_Health_Management_Report"，不要输出任何多余信息。'}]
    response = get_response_from_aliyunAPI(messages=messages,yun_model_if_stream=False, 
                                               yun_model_api_id=yun_model_api_id,  yun_model_api_key=yun_model_api_key, 
                                               yun_model_name=yun_model_name, stream_options={"include_usage": True},
                                               modalities=["text"], temperature=temperature, top_p=top_p, top_k=None,
                                               presence_penalty=None, response_format=None, 
                                               max_tokens=96, n=1,
                                               enable_thinking=False, thinking_budget=None,
                                               seed=None, stop=None, tools=None, tool_choice=None, parallel_tool_calls=None, # translation_options=None,
                                               enable_search=False, forced_search=None, search_strategy='max',
                                               )
    print('response from intent recognition:\n', response)
    for word in Task_Label_Dict.keys():
        if word in User_question:
            Task_Now = Task_Label_Dict[word]
            print('Task_Now:', Task_Now)
            break  # 跳出循环，找到第一个匹配的意图就行
        else:
            task_ = Task_Label_Dict[word]
            if task_ in response.choices[0].message.content.strip():
                Task_Now = task_
                print('Task_Now:', Task_Now)
    # 开始计时并打印当前时间
    start_time = time.time()
    print(f"Current Task_Now: {Task_Now} | Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")



    # 4. 处理输入、获取RAG材料
    print('prompt——all:\n', prompt)
    """如果您使用的是Cherry Studio v1.2.4或者更早版本，RAG的材料会直接包含在prompt中，采用以下代码进行RAG材料的分割："""
    #     # 一般来讲，user的prompt里面有三个部分：
    #     # 开头到“## My question is:”之间的部分是系统提示，告诉模型它是什么角色、如何使用引用，
    #     # 接下来是“## My question is:”和“## Reference Materials:”之间的部分是用户的问题及提供的数据，即对话框里面用户问的，
    #     # 最后是“## Reference Materials:”之后的部分是参考材料，也就是RAG搜索出来的一堆ID及其原文。
    # if "## Reference Materials:" in prompt:
    #     if "## My question is:" in prompt:
    #         User_prompt = "You are an expert in the operation and maintenance of spacecraft power systems." + prompt.split("## My question is:")[0].strip()
    #         User_question = "## My question is:\n\n" + prompt.split("## My question is:")[1].split("## Reference Materials:")[0].strip()
    #         RAG_materials = "## Reference Materials:\n\n" + prompt.split("## Reference Materials:")[1].strip()
    #     else:
    #         User_prompt = "You are an expert in the operation and maintenance of spacecraft power systems."
    #         User_question = "## My question is:\n\n" + prompt.split("## Reference Materials:")[0].strip()
    #         RAG_materials = "## Reference Materials:\n\n" + prompt.split("## Reference Materials:")[1].strip()
    # else:
    #     if "## My question is:" in prompt:
    #         User_prompt = "You are an expert in the operation and maintenance of spacecraft power systems." + prompt.split("## My question is:")[0].strip()
    #         User_question = "## My question is:\n\n" + prompt.split("## My question is:")[1].strip()
    #         RAG_materials = ""
    #     else:
    #         User_prompt = "You are an expert in the operation and maintenance of spacecraft power systems."
    #         User_question = prompt.strip()
    #         RAG_materials = ""
    """但是，如果您使用的是Cherry Studio v1.6.2或者更新版本，RAG的材料得要发去XML请求，才会进行搜索："""
    if ("<tool_use_result>" in prompt) and ("builtin_knowledge_search" in prompt):
        # 说明这种进行RAG搜索，这是Cherry Studio返回来的,所返回；
        # 1. 第一个是system prompt，前面已经通过cherry_prompt = [messages[0]]保存到cherry_prompt里面了
        # 2. 中间的是之前的对话历史，已经通过history = messages[1:-1]保存到history里面了
        # 3. 后面是两个 RAG查询历史，分别是'role': 'user'的原始提问，和'role': 'assistant'将该原始提问作为Keywords产生的那个XML，
        #    这两role和content也被history = messages[1:-1]保存到history里面了，就是history里面最后两条
        # 4. 最后一条是'role': 'user'的RAG查询结果，也就是RAG_materials作为这一条的content，用<result>和</result>包裹的就是那一推  id及查询结果  列表
        # RAG_materials = prompt.split("<tool_use_result>")[-1].split("</tool_use_result>")[0].strip()
        m = re.search(r"<result>(.*)</result>", prompt, re.S)
        RAG_materials = m.group(1).strip() if m else ""
        # RAG_materials = prompt.split("<result>")[-1].split("</result>")[0].strip()
        User_question = history[-2].get('content', '').strip()  # history里面倒数第二条的content就是原始提问
        User_prompt = """Please answer the question based on the reference materials\n\n## Citation Rules:\n- Please cite the context at the end of sentences when appropriate.\n- Please use the format of citation number [number] to reference the context in corresponding parts of your answer.\n- If a sentence comes from multiple contexts, please list all relevant citation numbers, e.g., [1][2]. Remember not to group citations at the end but list their sourceUrl."""
        # 既然User_question和RAG_materials都已经从history里面提取出来了，那就把history里面的最后两条删掉，避免重复
        history = history[:-2]
        print('RAG_materials:\n', RAG_materials)
        print('history after removing last two RAG-related entries:\n', history)
    else:
        # 那这时就有两种可能，一个是该任务需要RAG，那就要把这个prompt生成XML去RAG系统搜索；
        # 另一种是该任务不需要RAG，那就直接把RAG_materials置为空
        User_prompt = "You are an expert in the operation and maintenance of spacecraft power systems."
        User_question = prompt.strip()
        # if Task_Now in ['RCA_and_MDM', 'Normal_QA']:  # 这两种任务需要RAG
        if Task_Now == 'RCA_and_MDM':
            from API_server.RCA_api_utils import get_RAG_materials_from_XML_result, get_RAG_materials_def_xml
            from API_server.Other_api_utils import non_stream_json_generate, stream_manual_no_think_chunks_generate
            XML_data = get_RAG_materials_def_xml(Keywords=prompt)
            print('XML_data:\n', XML_data)
            # 将XML发送给RAG系统，获取搜索结果
            if stream_mode == False:  # 非流式,若以非流式方式提问，那就以非流式方式回应
                return jsonify(non_stream_json_generate(XML_data, data, model_name))
            else:  # 流式，若以流式方式提问，那就以流式方式回应
                return Response(stream_manual_no_think_chunks_generate(XML_data, data), mimetype='text/event-stream')
        else:
            RAG_materials = ""



    # 5.开始处理用户问题
    if Task_Now == 'Normal_QA':
        # 有记忆：
        # messages = data.get('messages', [])
        # 无记忆：
        messages=[{'role': 'system', 'content': '你是一个航天器电源系统运行维护专家，能够执行包括知识问答、异常检测、工况识别、故障定位、根因分析及维护决策等任务。具体来讲包括：1.知识问答：通过查询来自网络及用户本地的知识，回答用户提出的关于航天器电源系统的问题；2.异常检测：当检测到用户有执行异常检测任务的意图时，你可以为用户提供本地异常检测算法的选择、参数设置、训练和测试等功能；3.工况识别：当用户关注某数据段的工况时，你可以根据用户提供的传感器监测数据，分析该段数据的工况；4.故障定位：当用户关注故障数据段的故障类型时，你能够调用微调模型对故障数据段进行故障类型判别或定位；5.根因分析及维护决策：当用户关注故障发生的根本原因及维护方法时，你可以查询用户的知识库文件夹，为用户完成根因分析、风险评判、严重程度定级、辅助决策和检修策略推荐等任务。注意1：如果用户的意图难以识别，你可以向用户复述下上述功能并提醒用户以更清晰的方式表达问题，例如：【请以以下格式明确告知您需要执行的任务：请执行<普通问答/异常检测/工况识别/故障定位/根因分析及维护决策>任务...】。注意2：用户的健康管理追求在高效快速完成，无法进行多轮深入对话，请勿反复追问、不需多余确认是与否直接执行用户的意图，一次性提供所有可能有助于分析的信息，尽量在单次交互中解决用户的问题。'},
                  # You are an expert in the operation and maintenance of spacecraft power systems, able to perform tasks such as knowledge question answering, anomaly detection, work condition recognition, fault localization, and root cause analysis and maintenance decision-making.
                  {'role': 'user', 'content': User_question}]
        from API_server.Other_api_utils import add_history_memory
        messages = add_history_memory(messages=messages, memory_conv_turns=memory_conv_turns, history_messages=history)
        response = get_response_from_aliyunAPI(messages=messages,yun_model_if_stream=stream_mode, 
                                               yun_model_api_id=yun_model_api_id,  yun_model_api_key=yun_model_api_key, 
                                               yun_model_name=yun_model_name, stream_options={"include_usage": True},
                                               modalities=["text"], temperature=temperature, top_p=top_p, top_k=None,
                                               presence_penalty=None, response_format=None, 
                                               max_tokens=None, n=1,
                                               enable_thinking=True, thinking_budget=None,
                                               seed=None, stop=None, tools=None, tool_choice=None, parallel_tool_calls=None, # translation_options=None,
                                               enable_search=True, forced_search=True, search_strategy='max',
                                               )
        
    elif Task_Now == 'Anomaly_Detection':
        # 得到messages和tools
        from API_server.AD_api_utils import get_messages_and_tools_for_anomaly_detection
        messages, tools = get_messages_and_tools_for_anomaly_detection(User_prompt, User_question, RAG_materials)
        from API_server.Other_api_utils import add_history_memory
        messages = add_history_memory(messages=messages, memory_conv_turns=memory_conv_turns, history_messages=history)
        # ****** 如果tool调用效果不佳，可以把cherry_prompt也加入messages，因为Cherry Studio自动生成的system提示里面有很多很好的tool调用教程 ******
        # messages = cherry_prompt + messages
        # 调用云模型API
        tool_if_stream = False  # LLM在生成工具调用信息时是否使用流式输出，不建议开启，执行函数难写，后续的拼接并二次总结的添加也难写: https://help.aliyun.com/zh/model-studio/qwen-function-calling?spm=a2c4g.11186623.0.0.23b51d1cBV7hr1#dad2dbe656yhp
        tool_if_search = True  # LLM在生成工具调用信息时是否使用搜索功能，要的，因为要介绍算法
        tool_able_thinking = False  # 这个模型在非流式输出时 不支持思考，若需思考，要么换流式，要么换模型
        completion = get_response_from_aliyunAPI(messages=messages, yun_model_if_stream=tool_if_stream, 
                                               yun_model_api_id=yun_model_api_id,  yun_model_api_key=yun_model_api_key, 
                                               yun_model_name=yun_model_name, stream_options={"include_usage": True},
                                               modalities=["text"], temperature=temperature, top_p=top_p, top_k=None,
                                               presence_penalty=None, response_format=None, 
                                               max_tokens=None, n=1,
                                               enable_thinking=tool_able_thinking, thinking_budget=None,
                                               seed=None, stop=None, 
                                               tools=tools, tool_choice=None, parallel_tool_calls=True, 
                                               # translation_options=None,
                                               enable_search=tool_if_search, forced_search=None, search_strategy='max',
                                               )
        print('completion——all:\n', completion.model_dump_json())  # 打印完整的completion信息
        # 定义主进程共享字典，用于存储工具调用的结果
        # manager = Manager()
        # shared_dict = manager.dict()
        shared_dict = {}  # 使用普通字典
        shared_dict['response'] = None
        shared_dict['tool_reasoning_content'] = None
        shared_dict['tool_answer_content'] = None
        shared_dict['Tool_Now'] = {}
        shared_dict['Tool_already_id'] = []
        # 从completion中提取出tool
        for tool_call in completion.choices[0].message.tool_calls:
            if tool_call.function not in shared_dict['Tool_Now'].values():
                # 如果这个函数不在Tool_Now中，则添加它
                shared_dict['Tool_Now'][tool_call.id] = tool_call.function
            else:
                # 如果这个函数已经在Tool_Now中，则先返回正在执行的响应信息给用户，让其耐心等待
                shared_dict['Tool_already_id'].append(tool_call.id)
        # # 定义训练副进程
        def get_reponse_from_tools(shared_dict):
            # 根据completion的内容，执行其中的tool调用
            from API_server.AD_api_utils import execute_tools_for_anomaly_detection
            tool_results, tool_call_ids, tool_reasoning_content, tool_answer_content = execute_tools_for_anomaly_detection(completion, tool_if_stream, shared_dict['Tool_already_id'])
            print('tool_results:', tool_results)
            if tool_results == []:
                # 如果返回为空，说明模型训练正在进行，此次tool调用只是客户端因为较长时间没得到响应，重复询问、调用了同样的tool，因此不需执行、没结果
                response = []
            else:
                # 这些id的tool已经执行完毕，从Tool_Now中删除它们
                for tool_call_id in tool_call_ids:
                    if tool_call_id in shared_dict['Tool_Now']:
                        del shared_dict['Tool_Now'][tool_call_id]    # 删除Tool_Now中已经执行完的tool
                # 二次输入总结：（5. 大模型总结工具函数输出（可选））https://help.aliyun.com/zh/model-studio/qwen-function-calling?spm=a2c4g.11186623.0.0.23b51d1cBV7hr1#4fcae70821lua
                # 首先，添加Assistant Message，通过completion.choices[0].message得到 Assistant Message，将它添加到 messages 数组中
                messages.append(completion.choices[0].message)
                # 然后，将tool执行结果添加到messages中，再次调用云模型API，对结果进行总结
                for result, tool_call_id in zip(tool_results, tool_call_ids):
                    messages.append({'role': 'tool', 'content': result, "tool_call_id": tool_call_id})
                # 再次调用前，再加个system prompt，告诉模型首先要将tools的执行结果完整、准确地返回给用户，其次要对其中需要补充的信息在后面进行附加式的补充
                messages.append({'role': 'system', 'content': '以两个部分进行回复：1.第一个部分是首先将所调用工具的执行结果复述一遍，一个字都不许少、完整、准确、一致地返回给用户（复述工具执行结果，具体的后台工具调用信息不要提到）。尤其工具执行结果内若含有markdown形式插入的图片和代码块等，不准更改、不许减少、原样提供用户。2.第二部分是在其后附上一些需要补充的有用的参考信息在后面。'})
                # Please first return the execution result of the called tool completely, accurately and consistently to the user (only the result is needed, and the specific function call information does not need to be returned). Especially if the execution result contains images and code blocks inserted in markdown format, do not change them, and provide them to the user as they are. Then, the information that needs to be supplemented can be added in the back.
                response = get_response_from_aliyunAPI(messages=messages,yun_model_if_stream=stream_mode, 
                                                    yun_model_api_id=yun_model_api_id,  yun_model_api_key=yun_model_api_key, 
                                                    yun_model_name=yun_model_name, stream_options={"include_usage": True},
                                                    modalities=["text"], temperature=temperature, top_p=top_p, top_k=None,
                                                    presence_penalty=None, response_format=None, 
                                                    max_tokens=None, n=1,
                                                    enable_thinking=False, thinking_budget=None,
                                                    seed=None, stop=None, tools=None, tool_choice=None, parallel_tool_calls=None, # translation_options=None,
                                                    enable_search=True, forced_search=None, search_strategy='max',
                                                    )
            # 将结果存入共享字典
            shared_dict['response'] = response
            shared_dict['tool_reasoning_content'] = tool_reasoning_content
            shared_dict['tool_answer_content'] = tool_answer_content

        # 启动子进程
        # process = Process(target=get_reponse_from_tools, args=(shared_dict,))
        process = Thread(target=get_reponse_from_tools, args=(shared_dict,), daemon=True)
        process.start()
        
    elif Task_Now == 'Work_Condition_Recognition':
        from API_server.MR_api_utils import get_MD_data_from_question_or_history
        MD_data_question = get_MD_data_from_question_or_history(User_question, data)
        messages=[{'role': 'system', 'content': MD_prompt_style}, 
                  {'role': 'user', 'content': MD_data_question}]
        from API_server.Other_api_utils import add_history_memory
        messages = add_history_memory(messages=messages, memory_conv_turns=memory_conv_turns, history_messages=history)
        response = get_response_from_aliyunAPI(messages=messages,yun_model_if_stream=stream_mode, 
                                               yun_model_api_id=yun_model_api_id,  yun_model_api_key=yun_model_api_key, 
                                               yun_model_name=yun_model_name, stream_options={"include_usage": True},
                                               modalities=["text"], temperature=temperature, top_p=top_p, top_k=None,
                                               presence_penalty=None, response_format=None, 
                                               max_tokens=None, n=1,
                                               enable_thinking=True, thinking_budget=None,
                                               seed=None, stop=None, tools=None, tool_choice=None, parallel_tool_calls=None, # translation_options=None,
                                               enable_search=True, forced_search=None, search_strategy='max',
                                               )
        # 用户提问的两种方法：
        # 1.“请执行工况识别任务。需要分析的各传感器数据如下：....”
        # 2.“既然你检测到<2024/10/18  18:42:01 - 2024/10/18  18:43:01>这段时间发生了异常，那么异常发生之前航天器正处于什么工况？请执行工况识别任务。”

    elif Task_Now == 'Fault_Localization':
        global FD_model, FD_tokenizer  # 添加这一行声明全局变量
        if (FD_model is None) or (FD_tokenizer is None):
            # 如果模型和tokenizer还没有加载，则加载它们
            FD_model, FD_tokenizer = load_local_finetuned_model(
                if_load_LoRA_or_merged=if_load_LoRA_or_merged,  # 是否加载LoRA适配器或合并模型
                model_name=model_name,  # 模型名称
                max_seq_length=FD_first_question_max_seq_length,  # 最大序列长度
                dtype=dtype,  # 数据类型
                load_in_4bit=load_in_4bit,  # 是否使用4位量化
                cache_dir=cache_dir,  # 缓存目录
                proxies= None,  # 代理设置
                adapter_path=adapter_path,  # LoRA适配器路径
                checkpoint_path=checkpoint_path,  # 检查点路径
                checkpoint_max_step=checkpoint_max_step,  # 检查点最大步数
            )
            # 准备模型以进行推理
            FastLanguageModel.for_inference(FD_model)
        from API_server.FD_api_utils import response_fault_localization
        response_or_data, FD_type_result = response_fault_localization(User_prompt=User_prompt, User_question=User_question, RAG_materials=RAG_materials, 
                                                                data=data, temperature=temperature, top_p=top_p, stream_mode=stream_mode,
                                                                model=FD_model, tokenizer=FD_tokenizer, fault_dict_chinese=fault_dict_chinese,
                                                                second_question_style= second_question_style, 
                                                                second_question_if_use_fintuned = FD_second_question_if_use_fintuned,
                                                                max_new_tokens = FD_first_question_max_new_tokens, 
                                                                second_question_max_new_tokens = FD_second_question_max_new_tokens)
        # 用户的提问有两种可能：
        # 1. User_question是一个完整的带有数据的(finetune LLM那个jupyter)：“###指令:请基于所提供数据，。。。###数据: 提取特征和原始数据如下：。。。”
        # 2. User_question是一个不带数据的：“那么你所检测出的<2024/10/18  18:42:01 - 2024/10/18  18:43:01>时间段内，发生了什么类型的故障？ ”
        if FD_second_question_if_use_fintuned == False:
            sample_data_text = response_or_data
            messages=[{'role': 'system', 'content': '你是一个航天器电源系统运行维护专家，能够基于传感器监测数据执行故障定位任务。请基于所提供的数据，判断该数据段内发生的故障类型，并给出理由说明。'},
                    {"role": "user", "content": second_question_style.format(FD_type_result, sample_data_text, FD_type_result)},]
            response = get_response_from_aliyunAPI(messages=messages,yun_model_if_stream=stream_mode, 
                                                yun_model_api_id=yun_model_api_id,  yun_model_api_key=yun_model_api_key, 
                                                yun_model_name=yun_model_name, stream_options={"include_usage": True},
                                                modalities=["text"], temperature=temperature, top_p=top_p, top_k=None,
                                                presence_penalty=None, response_format=None, 
                                                max_tokens=None, n=1,
                                                enable_thinking=True, thinking_budget=None,
                                                seed=None, stop=None, tools=None, tool_choice=None, parallel_tool_calls=None, # translation_options=None,
                                                enable_search=True, forced_search=None, search_strategy='max',
                                                )
        else:
            response = response_or_data

    elif Task_Now == 'RCA_and_MDM':
        from API_server.RCA_api_utils import get_all_messages_for_RCA
        messages = get_all_messages_for_RCA(User_prompt, User_question, RAG_materials, RCA_prompt_style)
        from API_server.Other_api_utils import add_history_memory
        messages = add_history_memory(messages=messages, memory_conv_turns=memory_conv_turns, history_messages=history)
        response = get_response_from_aliyunAPI(messages=messages,yun_model_if_stream=stream_mode, 
                                               yun_model_api_id=yun_model_api_id,  yun_model_api_key=yun_model_api_key, 
                                               yun_model_name=yun_model_name, stream_options={"include_usage": True},
                                               modalities=["text"], temperature=temperature, top_p=top_p, top_k=None,
                                               presence_penalty=None, response_format=None, 
                                               max_tokens=None, n=1,
                                               enable_thinking=True, thinking_budget=None,
                                               seed=None, stop=None, tools=None, tool_choice=None, parallel_tool_calls=None, # translation_options=None,
                                               enable_search=True, forced_search=True, search_strategy='max',
                                               )
        # 用户提问的两种方法：
        # 1.“我们检测到航天器电源系统发生了【...】故障，请查询我的知识库，执行根因分析及维护决策任务。”
        # 2.按照RCA_prompt_style提供全部信息：“## 任务：用户正在进行异。。。## 背景：已知运维对。。。## 回答格式：你回答。。。”

    elif Task_Now == 'Export_Health_Management_Report':
        # 健康管理报告导出
        if export_report_by_manual_or_yun == 'manual':
            # 手动导出
            from API_server.Other_api_utils import export_health_management_report_manual, save_report_markdown
            response = export_health_management_report_manual(User_question, data)
            save_report_markdown(response, export_report_by_manual_or_yun)
        else:
            # 调用大模型进行总结
            messages = messages = data.get('messages', [])
            # 替换第一个role': 'system'的为：
            messages[0] = {'role': 'system', 'content': 'You are an expert in the operation and maintenance of spacecraft power systems. Please summarize all conversations as a health management report and return to markdown.'}
            # # 凡是'role': 'user'的全部去掉
            # messages = [message for message in messages if message['role'] != 'user']
            # 最后补一个'role': 'user'
            messages[-1] = {'role': 'user', 'content': User_question + 'Please sort out, summarize all the above conversation content and generate an "Health Management Report" with the following requirements: 1. Format: Use Markdown typesetting and return in the format of a Markdown file. 2. Completeness: Do not miss any key information and be as detailed as possible (Not only some part, all the history task should be summarized), especially the anomaly detection results (abnormal range list and visualization) and their download links. 3. Language: Output the Chinese version first, and then the English version. The content of the two versions must be exactly the same.'}
            print('messages——all:\n', messages)
            response = get_response_from_aliyunAPI(messages=messages,yun_model_if_stream=stream_mode, 
                                                yun_model_api_id=yun_model_api_id,  yun_model_api_key=yun_model_api_key, 
                                                yun_model_name=yun_model_name, stream_options={"include_usage": True},
                                                modalities=["text"], temperature=temperature, top_p=top_p, top_k=None,
                                                presence_penalty=None, response_format=None, 
                                                max_tokens=None, n=1,
                                                enable_thinking=True, thinking_budget=None,
                                                seed=None, stop=None, tools=None, tool_choice=None, parallel_tool_calls=None, # translation_options=None,
                                                enable_search=False, forced_search=None, search_strategy='max',
                                                )
            # from API_server.Other_api_utils import save_report_markdown
            # save_report_markdown(response, export_report_by_manual_or_yun)
    else:
        response = "抱歉，请以以下格式明确告知您需要执行的任务：请执行【普通问答/异常检测/工况识别/故障定位/根因分析及维护决策】任务....  \n Sorry, please tell me which task you want to execute in the following format: Please execute the [normormal question and answer task/ anomaly detection task/ work condition recognition task/ fault localization task/ root cause analysis and maintenance decision-making task]..."

    # 6. 返回结果
    # 结束计时并打印耗时以及当前时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))} | Elapsed time: {elapsed_time:.6f} seconds")
    # 我在cherry studio上选择的API形式的OpenAI格式，所以要return jsonify({OPENAI格式})    
    # 如果OPAI格式来的时候是流式的，Cherry Studio 发送了 "stream": true，那就要流式返回，否则就是普通返回json
    if stream_mode:
        # 返回流式响应，除了AD时需要显示图片，返回的chunk是markdown格式的，其他都是文本格式的
        if Task_Now == 'Fault_Localization':
            if FD_second_question_if_use_fintuned == True:
                # 流式、文本返回
                # 微调模型返回的，文本格式的会通过if "<think>" in response提取思考部分
                from API_server.Other_api_utils import stream_manual_extract_think_chunks_generate
                return Response(stream_manual_extract_think_chunks_generate(response, data), mimetype='text/event-stream')
            else:
                # 流式、文本返回
                # 但是云端Qwen返回的，已经生成好OpenAI的chunk格式了
                from API_server.Other_api_utils import stream_Yun_chunks_generate
                return Response(stream_Yun_chunks_generate(response, data, yun_model_name), mimetype='text/event-stream')
        elif Task_Now == 'Anomaly_Detection':
            # 流式、图片返回，最后一个问答含有markdown格式的里面只有包含了图片的异常检测结果展示
            from API_server.Other_api_utils import stream_Yun_chunks_generate_4_2Process
            return Response(stream_Yun_chunks_generate_4_2Process(process, shared_dict, data, yun_model_name, heartbeat_interval=50), mimetype='text/event-stream')
            # if tool_results == []:
            #     # 如果返回为空，说明模型训练正在进行，此次tool调用只是客户端因为较长时间没得到响应，重复询问、调用了同样的tool，因此不执行、无结果
            #     # 但是要返回一个让客户端继续等待的信号，不要关闭信息接收
            #     # 状态码：1. 202 Accepted（推荐）请求已被接受，但处理尚未完成；2. 503 Service Unavailable服务暂时不可用，通常会自动重试；3. 429 Too Many Requests 请求过于频繁，等待一段时间后重试
            #     # return Response(status=202), {"message": "模型训练正在进行中，请耐心等待"}
            #     return Response(response=json.dumps({"message": "模型训练正在进行中，请耐心等待。The model training is currently underway, please wait...", "status": "processing", "retry_after": 300}), status=202, mimetype='application/json')  # 建议300秒后重试
            # else:
            #     # 流式、图片返回，最后一个问答含有markdown格式的里面只有包含了图片的异常检测结果展示
            #     from API_server.Other_api_utils import stream_Yun_chunks_generate
            #     return Response(stream_Yun_chunks_generate(response, data, yun_model_name), mimetype='text/event-stream')
        elif Task_Now in ['Normal_QA', 'Work_Condition_Recognition', 'RCA_and_MDM']:
            # 流式、文本返回、但是Qwen已经生成好OpenAI的chunk格式了
            # 除了上两个，其他任务都是由 云端大模型生成的response， 格式比较特殊，需要特殊处理
            from API_server.Other_api_utils import stream_Yun_chunks_generate
            return Response(stream_Yun_chunks_generate(response, data, yun_model_name), mimetype='text/event-stream')
        elif Task_Now == 'Export_Health_Management_Report':
            if export_report_by_manual_or_yun == 'manual':
                # 流式、文本返回
                # 这个时候response是一个字符串，里面包含了markdown格式的健康管理结果报告
                from API_server.Other_api_utils import stream_manual_no_think_chunks_generate
                return Response(stream_manual_no_think_chunks_generate(response, data), mimetype='text/event-stream')
            else:
                # 流式、文本返回、但是Qwen已经生成好OpenAI的chunk格式了
                from API_server.Other_api_utils import stream_Yun_chunks_generate
                return Response(stream_Yun_chunks_generate(response, data, yun_model_name), mimetype='text/event-stream')
        else:
            # 返回"抱歉，请以以下格式明确告知您需要执行的任务："
            # 流式、文本返回
            from API_server.Other_api_utils import stream_manual_extract_think_chunks_generate
            return Response(stream_manual_extract_think_chunks_generate(response, data), mimetype='text/event-stream')
    else:
        # 普通返回
        from API_server.Other_api_utils import non_stream_json_generate
        return jsonify(non_stream_json_generate(response, data, model_name))







if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1999)  # 启动Flask应用，监听所有IP地址的1999端口











