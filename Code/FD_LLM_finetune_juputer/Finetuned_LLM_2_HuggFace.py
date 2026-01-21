from huggingface_hub import HfApi, login, upload_folder
import json
import os

# 从API_server/api_keys.json读取本地的 Hugging Face 令牌huggingface_token
with open('./API_server/api_keys.json', 'r') as f:
    api_keys = json.load(f)
huggingface_token = api_keys['huggingface']
http_proxy = api_keys['http_proxy']


# 设置代理
os.environ['HTTP_PROXY'] = http_proxy
os.environ['HTTPS_PROXY'] = http_proxy
# (optional) Login with your Hugging Face credentials
login(token=huggingface_token)
# 创建 API 实例
# api = HfApi()
# 登录（或使用 token 参数）
# api = HfApi(token="your_token_here")

# # 或者设置 Hugging Face 镜像
# os.environ["HF_HUB_BASE_URL"] = "https://hf-mirror.com"
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 设置Hugging Face 镜像站 URL
# # 镜像不支持登录功能，因此跳过 login 步骤，在下面的 upload_folder 中直接使用 token 参数

# 上传的文件夹：
folder3 = r"/data/DiYi/MyWorks_Results/DY-LLM_4SPS-PHM_Project/finetuned_adapter_LLMmodel_HF"
folder4 = r"/data/DiYi/MyWorks_Results/DY-LLM_4SPS-PHM_Project/finetuned_LLMmodel_HF"
folder5 = r"/data/DiYi/MyWorks_Results/DY-LLM_4SPS-PHM_Project/checkpoint_2025.5.18"

folders = {
    "finetuned_adapter_LLMmodel_HF": folder3,
    "finetuned_LLMmodel_HF": folder4,
    "checkpoint_2025.5.18": folder5
}

# 一些草稿记录文件不需要上传
ignore_rules = {
    "checkpoint_2025.5.18": ["2rd_ques_Qwen3-14B-20250510.txt"],
    "finetuned_adapter_LLMmodel_HF": ["README.md"],
}

# 上传时保留文件夹结构
for folder_name, folder_path in folders.items():
    print(f"Uploading folder: {folder_path} -> {folder_name}/")
    ignore_patterns = ignore_rules.get(folder_name, [])
    upload_folder(
        folder_path=folder_path,
        path_in_repo=folder_name,  # 在仓库中的路径
        repo_id="DiYi1999/Finetuned-Expert-LLM-of-SpaceHMchat-for-XJTU-SPS-Dataset",
        repo_type="model",
        ignore_patterns=ignore_patterns if ignore_patterns else None,
        token=huggingface_token
    )


