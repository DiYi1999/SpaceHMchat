# 本页面介绍了SpaceHMchat的部署方法，也即是其使用说明书。

## 注意事项：

1. 关于客户端和服务器的说明：
   > 本项目的交互主体包括：前端对话软件（客户端安装）、后端服务器（LLM运行）、云端（LLM运行）。  
   > 但这是因为我们的服务器没有足够内存来运行Qwen-235B大模型，所以才采用云端大模型API调用的方式进行交互。在实际部署时，Qwen3-235B-A22B是完全开源的，可以直接下载、部署在您的本地服务器上，无需调用云端API。  
   > 因此，实际部署时，只需要前端对话软件与本地后端服务器两个主体，直接让前端对话软件与本地后端服务器进行交互，代替我们项目里的云端API交互部分，全程数据均在本地流转。
2. 关于为什么基于中文对话及选用Qwen系列模型作为基模型的说明：
   > (1) 该项目受到中国基金项目支持，目前致力于服务中国航天领域的用户群体或研究者。  
   > (2) 我们必须保证使用的模型是开源的，在保证信息安全的同时可以被用户自由下载部署使用，因此我们选择Qwen系列模型（Qwen-14B, Qwen-235B）作为我们的基础模型，Qwen模型是能与ChatGPT等闭源模型相媲美的少数开源大模型之一，且Qwen模型基于大量中文语料进行训练，在中文对话方面表现更加出色。  
   > (3) 我们的知识库主要是中文资料，包含大量中文技术文档、知识总结笔记、论文和报告等，因此在中文对话方面能够更好地利用这些知识库内容。
3. 服务器端下载本项目代码，客户端软件需要对话时，服务器运行[Code\API_server\LLM_4_SPS_PHM_server_api.py](../Code/API_server/LLM_4_SPS_PHM_server_api.py)文件，若对话需求频繁，建议使用tmux等保证该文件于后台长期稳定运行。所需的python环境参考[Deploy\Python Packages Version](../Deploy/Python%20Packages%20Version/)。



## 前期准备：

### 客户端

1. 根据我们在[Deploy\Chat Software\readme.md](../Chat%20Software/readme.md)中提供的说明，安装并配置好前端对话软件，尤其是设置上下文保留数量为最大，temperature和top_p等参数也可以根据需要进行设置。

### 双端通信

1. 通过SSH连接建立客户端与服务器的通信：`ssh -N -L 1999:localhost:1999 your_username@server_ip`,其中`your_username`是服务器的用户名，`server_ip`是服务器的IP地址。此命令会将本地的1999端口映射到服务器的1999端口，从而实现前端对话软件与后端服务器的通信。如果您不使用1999端口，可以根据需要修改为其他端口号，但需要确保[Code\API_server\LLM_4_SPS_PHM_server_api.py](../Code/API_server/LLM_4_SPS_PHM_server_api.py)文件中的端口号设置与之一致。
   > 此步骤使用Xshell等SSH客户端软件的隧道功能实现会更方便。  
   > 此步骤也可以使用其他手段实现，比如内网穿透等，根据您的实际情况进行选择。
2. 软件右下角 -> 设置 -> 模型服务 -> 添加 -> 提供商名称填写“dy@ssh62229” -> 提供商类型选择“OpenAI” -> 确定 -> API地址填写“http://localhost:1999/generate” -> 模型 -> 添加 -> 模型ID、名称等全部填写“SpaceHMchat” -> 添加模型 -> 设置 -> 默认模型 -> 默认助手模型选择“SpaceHmchat”。
   > 提供商名称和模型名称可以根据需要进行修改。  
   > API地址中的端口号需要与第一步中使用的端口号一致。  
   > 确定API地址填写后下方自动补全后显示的详细链接和[Code\API_server\LLM_4_SPS_PHM_server_api.py](../Code/API_server/LLM_4_SPS_PHM_server_api.py)文件中的路由(`@app.route('/generate/v1/chat/completions', methods=['POST'])`)保持一致。

### 云端模拟

1. 将您的阿里云百炼云平台（[CN](https://bailian.console.aliyun.com/) [EN](https://modelstudio.console.alibabacloud.com/)）的API Key（[CN](https://bailian.console.aliyun.com/cn-beijing/?apiKey=1&tab=globalset#/efm/api_key) [EN](https://modelstudio.console.alibabacloud.com/us-east-1?tab=dashboard#/api-key) [SG](https://modelstudio.console.alibabacloud.com/ap-southeast-1/?tab=playground#/api-key)）填写到[Code\API_server\api_keys.json](../Code/API_server/api_keys.json)文件中对应的位置，确保后端服务器可以调用云端Qwen-235B大模型API进行推理。
2. 将您的Hugging Face API Key填写到[Code\API_server\api_keys.json](../Code/API_server/api_keys.json)文件中对应的位置，以便后端服务器可以访问Hugging Face模型。
3. 不同地域的base_url不通用，根据所在地域，修改[Code\API_server\LLM_4_SPS_PHM_server_api.py](../Code/API_server/LLM_4_SPS_PHM_server_api.py)文件中的`yun_model_api_id`变量：
   > - 华北2（北京）: https://dashscope.aliyuncs.com/compatible-mode/v1
   > - 美国（弗吉尼亚）: https://dashscope-us.aliyuncs.com/compatible-mode/v1
   > - 新加坡: https://dashscope-intl.aliyuncs.com/compatible-mode/v1



## 代码适配：

### 工况识别

1. 检查[Code\API_server\LLM_4_SPS_PHM_server_api.py](../Code/API_server/LLM_4_SPS_PHM_server_api.py)以及[Code\API_server\MR_api_utils.py](../Code/API_server/MR_api_utils.py)中的工况识别相关提示词设置是否符合您的需求，若不符合，请根据您的需求进行修改。

### 异常检测

1. 检查[Code\API_server\LLM_4_SPS_PHM_server_api.py](../Code/API_server/LLM_4_SPS_PHM_server_api.py)以及[Code\API_server\AD_api_utils.py](../Code/API_server/AD_api_utils.py)中的异常检测相关提示词设置是否符合您的需求，若不符合，请根据您的需求进行修改。
2. 如有需要，可以定制化修改位于[Code\AD_repository](../Code/AD_repository)文件夹下的异常检测工具库：
   > [Code\AD_repository\data](../Code/AD_repository/data)文件夹下存放的是数据导入脚本。  
   > [Code\AD_repository\graph](../Code/AD_repository/graph)文件夹下存放的是GNN类方法需要的图结构构建脚本。  
   > [Code\AD_repository\utils](../Code/AD_repository/utils)文件夹下存放的是杂项功能脚本。  
   > [Code\AD_repository\model\AD_algorithms_params](../Code/AD_repository/model/AD_algorithms_params)文件夹下存放的是各类异常检测算法的默认参数。  
   > [Code\AD_repository\model\baseline](../Code/AD_repository/model/baseline)文件夹下存放的是各类纯神经网络异常检测算法的实现代码。  
   > [Code\AD_repository\model\ours](../Code/AD_repository/model/ours)文件夹下存放的是PINNs等特殊神经网络特别集成所需要的入口文件。  
   > [Code\AD_repository\model\MyModel.py](../Code/AD_repository/model/MyModel.py)文件是调用baseline和ours文件夹下各类异常检测算法的主接口文件。  
   > [Code\AD_repository\main.py](../Code/AD_repository/main.py)文件是整个异常检测工具库的主入口。  

### 故障定位

1. 检查[Code\API_server\LLM_4_SPS_PHM_server_api.py](../Code/API_server/LLM_4_SPS_PHM_server_api.py)以及[Code\API_server\FD_api_utils.py](../Code/API_server/FD_api_utils.py)中的故障定位相关提示词设置是否符合您的需求，若不符合，请根据您的需求进行修改。
2. 如有需要，可以使用我们公开于[Code\FD_LLM_finetune_juputer](../Code/FD_LLM_finetune_juputer)文件夹下的脚本，对Qwen系列模型进行微调，训练您自己的故障定位专家大模型：
   > [Code\FD_LLM_finetune_juputer\download_model.py](../Code/FD_LLM_finetune_juputer/download_model.py)文件用于下载Qwen系列基座大模型。  
   > [Code\FD_LLM_finetune_juputer\FD_LLM_finetune.ipynb](../Code/FD_LLM_finetune_juputer/FD_LLM_finetune.ipynb)文件是故障定位大模型微调的主脚本。  
   > [Code\FD_LLM_finetune_juputer\QandA_Dataset_4LLM](../Code/FD_LLM_finetune_juputer/QandA_Dataset_4LLM)文件夹下是将时间序列数据集转化为微调所需的问答对数据集的方法。
3. 如果您的航天器电源系统与我们的高度相似，希望直接使用我们微调好的故障定位专家大模型，可以从[https://huggingface.co/DiYi1999/Finetuned-Expert-LLM-of-SpaceHMchat-for-XJTU-SPS-Dataset](https://huggingface.co/DiYi1999/Finetuned-Expert-LLM-of-SpaceHMchat-for-XJTU-SPS-Dataset)下载。
我们的微调模型是基于Qwen-14B大模型进行微调的，需要先使用[Code\FD_LLM_finetune_juputer\download_model.py](../Code/FD_LLM_finetune_juputer/download_model.py)下载Qwen-14B基座大模型，然后将我们微调好的模型权重加载到Qwen-14B大模型上即可使用。
其精度效果通过混淆矩阵展示于[Code\FD_LLM_finetune_juputer\FD_LLM_finetune.ipynb](../Code/FD_LLM_finetune_juputer/FD_LLM_finetune.ipynb)文件的末尾。

### 维护决策

1. 由于我们知识库的许多文件只允许我们研究使用而无权公开分享，所以我们详细讲解[构建知识库的过程](https://docs.cherry-ai.com/docs/en-us/knowledge-base/knowledge-base)：
2. 打开软件 -> 知识库 -> 添加 -> 名称填写“SpaceHMchat知识库” -> 按照需求进行其他设置（请求文档片段数量设置为最大） -> 确定。
3. 点击“文件” -> 点击“添加文件” -> 选择知识库文件 -> 等待文件向量化完成。
4. 点击“笔记” -> 点击“添加笔记” -> 选择知识库笔记 -> 等待笔记向量化完成。
5. 点击“目录” -> 点击“添加目录” -> 选择知识库目录 -> 等待目录内文件向量化完成。
6. 点击“网址” -> 点击“添加网址” -> 输入网址（比如https://en.wikipedia.org/wiki/Failure_mode,_effects,_and_criticality_analysis）-> 确定 -> 等待网址内容向量化完成。
7. 点击“网站” -> 点击“站点地图” -> 输入网站地图XML（比如https://ntrs.nasa.gov/sitemap.xml）-> 确定 -> 等待网站内容向量化完成。
8. 通过以上步骤，即可实现归零文档、运维笔记、维护指导手册、设计文档、全球航天器事故案例研究、论文、技术标准、专家咨询档案等文件的向量化。
9. 打开软件 -> 助手 -> SpaceHMchat右键选择“编辑助手” -> 选择“知识库设置” -> 选择“SpaceHMchat知识库”。

### 主文件检查

[Code\API_server\LLM_4_SPS_PHM_server_api.py](../Code/API_server/LLM_4_SPS_PHM_server_api.py)是连接所有健康管理功能和进行双端通信的主文件，我们已经将主要参数都放在了文件的前几行，准备阶段的最后一步就是检查该文件前面设置的文件路径、启用GPU编号等变量是否与您的实际情况对应。



## Now Enjoy It！

1. 运行[Original Code\15_LLM_4_SPS_PHM\API_server\LLM_4_SPS_PHM_server_api.py](../Code/API_server/LLM_4_SPS_PHM_server_api.py)文件。
2. 打开对话软件，选择默认助手，开始对话！






