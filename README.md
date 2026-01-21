### SpaceHMchat: a Human-AI collaboration framework for all-in-loop health management, it serves across the entire loop of work condition recognition, anomaly detection, fault localization, and maintenance decision making.

> [Deploy](../Deploy/) provides [deployment methods and use instructions](../Deploy/readme.md) for SpaceHMchat.  
> [Code](../Code/) is the [underlying codebase](../Code/API_server/) for SpaceHMchat, such as [anomaly detection tools](../Code/AD_repository/), [LLM finetuned methods](../Code/FD_LLM_finetune_juputer/), etc.  
> [Data](../Data/) is the [open-source dataset](https://diyi1999.github.io/XJTU-SPS/) for spacecraft power system AIL health management.  
> [A Dialogue Example](#spacechat对话示例--dialogue-example-of-spacechat) of SpaceHMchat is provided at the end of this README.

******

#### if it is helpful for your research, you can cite the following works:

```bibtex
@misc{di2026empoweringallinloophealthmanagement,
title={Empowering All-in-Loop Health Management of Spacecraft Power System in the Mega-Constellation Era via Human-AI Collaboration}, 
author={Yi Di and Zhibin Zhao and Fujin Wang and Xue Liu and Jiafeng Tang and Jiaxin Ren and Zhi Zhai and Xuefeng Chen},
year={2026},
eprint={2601.12667},
archivePrefix={arXiv},
primaryClass={cs.AI},
url={https://arxiv.org/abs/2601.12667}, 
}

@article{DI2025113380,
title = {PhyGNN: Physics guided graph neural network for complex industrial power system modeling},
journal = {Mechanical Systems and Signal Processing},
volume = {240},
pages = {113380},
year = {2025},
issn = {0888-3270},
doi = {https://doi.org/10.1016/j.ymssp.2025.113380},
url = {https://www.sciencedirect.com/science/article/pii/S0888327025010817},
author = {Yi Di and Fujin Wang and Zhi Zhai and Zhibin Zhao and Xuefeng Chen},
keywords = {Physics guided graph neural network, Spacecraft power system, Multivariate time series, Complex industrial system},
}
```

******
### SpaceHMchat对话示例  (Dialogue Example of SpaceHMchat):
`从对话软件截图得到，本身交流对话是基于中文，为了便于阅读，我们在每轮中文问题及回答后面使用软件的翻译功能附上了英文翻译。`  
`Captured from Dialogue Software Screenshot, the original conversation is in Chinese. For better readability, we have provided English translations after each Chinese question and answer using the software's translation function.`

******
> 关于为什么基于中文对话及选用Qwen系列模型作为基模型的说明：
>> （1）该项目受到中国基金项目支持，目前致力于服务中国航天领域的用户群体或研究者。  
>> （2）我们必须保证使用的模型是开源的，在保证信息安全的同时可以被用户自由下载部署使用，因此我们选择Qwen系列模型（Qwen-14B, Qwen-235B）作为我们的基础模型，Qwen模型是能与ChatGPT等闭源模型相媲美的少数开源大模型之一，且Qwen模型基于大量中文语料进行训练，在中文对话方面表现更加出色。  
>> （3）我们的知识库主要是中文资料，包含大量中文技术文档、知识总结笔记、论文和报告等，因此在中文对话方面能够更好地利用这些知识库内容。

> About the reason why the current SpaceHMchat mainly uses Chinese for conversations and chooses Qwen series models as the base models:
>> (1) This project is supported by a Chinese fund project and is currently dedicated to serving the user group in the Chinese aerospace field.  
>> (2) We must ensure that the base LLM models used are open source, which can be freely downloaded and deployed by users while ensuring information security. Therefore, we chose the Qwen series models (Qwen-14B, Qwen-235B) as our base models, as Qwen models are among the few open-source large models that can rival closed-source models such as ChatGPT. And Qwen models are trained on a large amount of Chinese corpus, thus performing better in Chinese conversations.  
>> (3) Our knowledge base mainly consists of Chinese materials, including a large number of Chinese technical documents, knowledge summary notes, papers, and reports. Therefore, it can better utilize these knowledge base contents in Chinese conversations.

![SpaceHMchat对话示例1](./Deploy/Pictures/CompleteChatExample.png)










