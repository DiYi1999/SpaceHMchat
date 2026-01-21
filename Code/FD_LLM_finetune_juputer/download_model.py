"""
这个文件用于从ModelScope下载Qwen3系列模型作为微调基座模型（包括量化版本和非量化版本）。
This file is used to download the Qwen3 series models from ModelScope as fine-tuning base models (including quantized and non-quantized versions).
"""



import os
from modelscope import snapshot_download

# path = '/data/DiYi/MyWorks_Results/DY-LLM_4SPS-PHM_Project/LLM/ModelScope_Qwen3-0.6B-unsloth-bnb-4bit'
# os.makedirs(path, exist_ok=True)
# model_dir = snapshot_download('unsloth/Qwen3-0.6B-unsloth-bnb-4bit', local_dir=path, max_workers=8)
# print('Model downloaded successfully!  0')

# path = '/data/DiYi/MyWorks_Results/DY-LLM_4SPS-PHM_Project/LLM/ModelScope_Qwen3-1.7B-unsloth-bnb-4bit'
# os.makedirs(path, exist_ok=True)
# model_dir = snapshot_download('unsloth/Qwen3-1.7B-unsloth-bnb-4bit', local_dir=path, max_workers=8)
# print('Model downloaded successfully!  1')

# path = '/data/DiYi/MyWorks_Results/DY-LLM_4SPS-PHM_Project/LLM/ModelScope_Qwen3-4B-unsloth-bnb-4bit'
# os.makedirs(path, exist_ok=True)
# model_dir = snapshot_download('unsloth/Qwen3-4B-unsloth-bnb-4bit', local_dir=path, max_workers=8)
# print('Model downloaded successfully!  2')

# path = '/data/DiYi/MyWorks_Results/DY-LLM_4SPS-PHM_Project/LLM/ModelScope_Qwen3-8B-unsloth-bnb-4bit'
# os.makedirs(path, exist_ok=True)
# model_dir = snapshot_download('unsloth/Qwen3-8B-unsloth-bnb-4bit', local_dir=path, max_workers=8)
# print('Model downloaded successfully!  3')

path = '/data/DiYi/MyWorks_Results/DY-LLM_4SPS-PHM_Project/LLM/ModelScope_Qwen3-14B-unsloth-bnb-4bit'
os.makedirs(path, exist_ok=True)
model_dir = snapshot_download('unsloth/Qwen3-14B-unsloth-bnb-4bit', local_dir=path, max_workers=8)
print('Model downloaded successfully!  4')

# path = '/data/DiYi/MyWorks_Results/DY-LLM_4SPS-PHM_Project/LLM/ModelScope_Qwen3-32B-unsloth-bnb-4bit'
# os.makedirs(path, exist_ok=True)
# model_dir = snapshot_download('unsloth/Qwen3-32B-unsloth-bnb-4bit', local_dir=path, max_workers=8)
# print('Model downloaded successfully!  5')

# path = '/data/DiYi/MyWorks_Results/DY-LLM_4SPS-PHM_Project/LLM/ModelScope_Qwen3-0.6B'
# os.makedirs(path, exist_ok=True)
# model_dir = snapshot_download('unsloth/Qwen3-0.6B', local_dir=path, max_workers=8)
# print('Model downloaded successfully!  0')

# path = '/data/DiYi/MyWorks_Results/DY-LLM_4SPS-PHM_Project/LLM/ModelScope_Qwen3-1.7B'
# os.makedirs(path, exist_ok=True)
# model_dir = snapshot_download('unsloth/Qwen3-1.7B', local_dir=path, max_workers=8)
# print('Model downloaded successfully!  1')

# path = '/data/DiYi/MyWorks_Results/DY-LLM_4SPS-PHM_Project/LLM/ModelScope_Qwen3-4B'
# os.makedirs(path, exist_ok=True)
# model_dir = snapshot_download('unsloth/Qwen3-4B', local_dir=path, max_workers=8)
# print('Model downloaded successfully!  2')

# path = '/data/DiYi/MyWorks_Results/DY-LLM_4SPS-PHM_Project/LLM/ModelScope_Qwen3-8B'
# os.makedirs(path, exist_ok=True)
# model_dir = snapshot_download('unsloth/Qwen3-8B', local_dir=path, max_workers=8)
# print('Model downloaded successfully!  3')

path = '/data/DiYi/MyWorks_Results/DY-LLM_4SPS-PHM_Project/LLM/ModelScope_Qwen3-14B'
os.makedirs(path, exist_ok=True)
model_dir = snapshot_download('unsloth/Qwen3-14B', local_dir=path, max_workers=8)
print('Model downloaded successfully!  4')

# path = '/data/DiYi/MyWorks_Results/DY-LLM_4SPS-PHM_Project/LLM/ModelScope_Qwen3-32B'
# os.makedirs(path, exist_ok=True)
# model_dir = snapshot_download('unsloth/Qwen3-32B', local_dir=path, max_workers=8)
# print('Model downloaded successfully!  5')




