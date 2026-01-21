# This page introduces the deployment methods of SpaceHMchat, which serves as its user manual. (我们也提供了中文版：[Deploy\readme_CN.md](./readme_CN.md))

## Notes:

1. Instructions on the client and server:
   > The interaction subjects of this project include: front-end dialogue software (client installation), back-end server (LLM operation), and cloud (LLM operation).  
   > However, this is because our server does not have enough memory to run the Qwen-235B LLM, so we adopt the cloud LLM API calling method for interaction. In actual deployment, Qwen3-235B-A22B is completely open source and can be directly downloaded and deployed on your local server without calling cloud APIs.  
   > Therefore, in actual deployment, only two subjects are needed: front-end dialogue software and local back-end server. Simply let the front-end dialogue software interact with the local back-end server, replacing the cloud API interaction part in our project, with all data flowing locally.
2. Explanation on why we are based on Chinese dialogue and choose Qwen series models as base models: 
   > (1) This project is supported by Chinese fund projects and is currently committed to serving user groups or researchers in the Chinese aerospace field.  
   > (2) We must ensure that the models used are open source, ensuring information security while allowing users to freely download and deploy them.  Therefore, we choose Qwen series models (Qwen-14B, Qwen-235B) as our base models. Qwen models are among the few open-source LLMs that can rival closed-source models like ChatGPT, and Qwen models are trained on a large amount of Chinese corpus, performing better in Chinese dialogue.  
   > (3) Our knowledge base is mainly Chinese materials, including a large number of Chinese technical documents, knowledge summary notes, papers, and reports, so it can better utilize these knowledge base contents in Chinese dialogue.
3. Download this project code on the server side. When the client software needs to chat, the server runs the [Code\API_server\LLM_4_SPS_PHM_server_api.py](../Code/API_server/LLM_4_SPS_PHM_server_api.py) file. If dialogue demands are frequent, it is recommended to use tmux or similar tools to ensure this file runs stably in the background for a long time. For the required Python environment, refer to [Deploy\Python Packages Version](../Deploy/Python%20Packages%20Version/).



## Preparation:

### Client Side

1. According to the instructions we provide in [Deploy\Chat Software\readme.md](../Chat%20Software/readme.md), install and configure the front-end dialogue software, especially setting the context retention number to maximum. Parameters such as temperature and top_p can also be set according to needs.

### Communication Between Clients and Server

1. Establish communication between client and server through SSH connection: `ssh -N -L 1999:localhost:1999 your_username@server_ip`, where `your_username` is the server username and `server_ip` is the server IP address. This command will map local port 1999 to server port 1999, enabling communication between front-end dialogue software and back-end server. If you don't use port 1999, you can modify it to other port numbers as needed, but ensure that the port number setting in [Code\API_server\LLM_4_SPS_PHM_server_api.py](../Code/API_server/LLM_4_SPS_PHM_server_api.py) file is consistent with it.
   > This step can be more conveniently implemented using the tunnel function of SSH client software such as Xshell.  
   > This step can also be implemented by other means, such as intranet penetration, etc., depending on your actual situation.
2. Software lower right corner -> Settings -> Model Provider -> Add -> Provider Name fill in "dy@ssh62229" -> Provider Type select "OpenAI" -> OK -> API Host fill in "http://localhost:1999/generate" -> Models -> Add -> Model ID, Name, etc. all fill in "SpaceHMchat" -> Add Model -> Software lower right corner -> Settings -> Default Model -> Default Assistant Model select "SpaceHmchat".
   > Provider name and model name can be modified as needed.  
   > The port number in the API address needs to be consistent with the port number used in the first step.  
   > After the API address is filled in, ensure the detailed link automatically completed below is consistent with the route in [Code\API_server\LLM_4_SPS_PHM_server_api.py](../Code/API_server/LLM_4_SPS_PHM_server_api.py) file (`@app.route('/generate/v1/chat/completions', methods=['POST'])`).

### Cloud Simulation

1. Fill in your Alibaba Cloud Model Studio ([CN](https://bailian.console.aliyun.com/) [EN](https://modelstudio.console.alibabacloud.com/)) API Key ([CN](https://bailian.console.aliyun.com/cn-beijing/?apiKey=1&tab=globalset#/efm/api_key) [EN](https://modelstudio.console.alibabacloud.com/us-east-1?tab=dashboard#/api-key) [SG](https://modelstudio.console.alibabacloud.com/ap-southeast-1/?tab=playground#/api-key)) into the corresponding position in [Code\API_server\api_keys.json](../Code/API_server/api_keys.json) file to ensure the back-end server can call the cloud Qwen-235B large model API for inference.
2. Fill in your Hugging Face API Key into the corresponding position in [Code\API_server\api_keys.json](../Code/API_server/api_keys.json) file so that the back-end server can access Hugging Face models.
3. Different regions have different base_urls. According to your region, modify the `yun_model_api_id` variable in the [Code\API_server\LLM_4_SPS_PHM_server_api.py](../Code/API_server/LLM_4_SPS_PHM_server_api.py) file:
   > - North China 2 (Beijing): https://dashscope.aliyuncs.com/compatible-mode/v1
   > - US (Virginia): https://dashscope-us.aliyuncs.com/compatible-mode/v1
   > - Singapore: https://dashscope-intl.aliyuncs.com/compatible-mode/v1



## Code Adaptation:

### Work Mode Recognition

1. Check whether the work mode recognition-related prompt settings in [Code\API_server\LLM_4_SPS_PHM_server_api.py](../Code/API_server/LLM_4_SPS_PHM_server_api.py) and [Code\API_server\MR_api_utils.py](../Code/API_server/MR_api_utils.py) meet your needs. If not, please modify them according to your needs.

### Anomaly Detection

1. Check whether the anomaly detection-related prompt settings in [Code\API_server\LLM_4_SPS_PHM_server_api.py](../Code/API_server/LLM_4_SPS_PHM_server_api.py) and [Code\API_server\AD_api_utils.py](../Code/API_server/AD_api_utils.py) meet your needs. If not, please modify them according to your needs.
2. If needed, you can customize and modify the anomaly detection toolkit located in the [Code\AD_repository](../Code/AD_repository) folder:
   > The [Code\AD_repository\data](../Code/AD_repository/data) folder stores data import scripts.  
   > The [Code\AD_repository\graph](../Code/AD_repository/graph) folder stores graph structure construction scripts needed for GNN-based methods.  
   > The [Code\AD_repository\utils](../Code/AD_repository/utils) folder stores miscellaneous function scripts.  
   > The [Code\AD_repository\model\AD_algorithms_params](../Code/AD_repository/model/AD_algorithms_params) folder stores default parameters for various anomaly detection algorithms.  
   > The [Code\AD_repository\model\baseline](../Code/AD_repository/model/baseline) folder stores implementation code for various pure neural network anomaly detection algorithms.  
   > The [Code\AD_repository\model\ours](../Code/AD_repository/model/ours) folder stores entry files needed for special neural network integration such as PINNs.  
   > The [Code\AD_repository\model\MyModel.py](../Code/AD_repository/model/MyModel.py) file is the main interface file for calling various anomaly detection algorithms in the baseline and ours folders.  
   > The [Code\AD_repository\main.py](../Code/AD_repository/main.py) file is the main entry point for the entire anomaly detection toolkit.

### Fault Localization

1. Check whether the fault localization-related prompt settings in [Code\API_server\LLM_4_SPS_PHM_server_api.py](../Code/API_server/LLM_4_SPS_PHM_server_api.py) and [Code\API_server\FD_api_utils.py](../Code/API_server/FD_api_utils.py) meet your needs. If not, please modify them according to your needs.
2. If needed, you can use the scripts we published in the [Code\FD_LLM_finetune_juputer](../Code/FD_LLM_finetune_juputer) folder to fine-tune Qwen series models and train your own fault localization expert large model:
   > The [Code\FD_LLM_finetune_juputer\download_model.py](../Code/FD_LLM_finetune_juputer/download_model.py) file is used to download Qwen series base large models.  
   > The [Code\FD_LLM_finetune_juputer\FD_LLM_finetune.ipynb](../Code/FD_LLM_finetune_juputer/FD_LLM_finetune.ipynb) file is the main script for fault localization large model fine-tuning.  
   > The [Code\FD_LLM_finetune_juputer\QandA_Dataset_4LLM](../Code/FD_LLM_finetune_juputer/QandA_Dataset_4LLM) folder contains methods for converting time series datasets into question-and-answer pair text datasets required for fine-tuning.
3. If your spacecraft power system is highly similar to ours and you wish to directly use our fine-tuned fault localization expert large model, you can download it from [https://huggingface.co/DiYi1999/Finetuned-Expert-LLM-of-SpaceHMchat-for-XJTU-SPS-Dataset](https://huggingface.co/DiYi1999/Finetuned-Expert-LLM-of-SpaceHMchat-for-XJTU-SPS-Dataset).
Our fine-tuned model is based on the Qwen-14B large model. You need to first use [Code\FD_LLM_finetune_juputer\download_model.py](../Code/FD_LLM_finetune_juputer/download_model.py) to download the Qwen-14B base large model, then load our fine-tuned model weights onto the Qwen-14B large model to use it.
Its precision performance is displayed through confusion matrix at the end of [Code\FD_LLM_finetune_juputer\FD_LLM_finetune.ipynb](../Code/FD_LLM_finetune_juputer/FD_LLM_finetune.ipynb) file.

### Maintenance Decision-making

1. Since many files in our knowledge base only allow us to use them for research and we have no right to share them publicly, we explain in detail [the process of building a knowledge base](https://docs.cherry-ai.com/docs/en-us/knowledge-base/knowledge-base):
2. Open software -> Knowledge Base -> Add -> Name fill in "SpaceHMchat Knowledge Base" -> Set other options according to needs (set "Requested Document Chunks" to maximum) -> OK.
3. Click "Files" -> Click "Add File" -> Select knowledge base files -> Wait for file vectorization to complete.
4. Click "Notes" -> Click "Add Note" -> Select knowledge base notes -> Wait for note vectorization to complete.
5. Click "Directories" -> Click "Add Directory" -> Select knowledge base directory -> Wait for files in directory to be vectorized.
6. Click "URLs" -> Click "Add URL" -> Enter URL (e.g., https://en.wikipedia.org/wiki/Failure_mode,_effects,_and_criticality_analysis) -> OK -> Wait for URL content vectorization to complete.
7. Click "Websites" -> Click "Website Map" -> Enter website sitemap XML (e.g., https://ntrs.nasa.gov/sitemap.xml) -> OK -> Wait for website knowledge vectorization to complete.
8. Through the above steps, you can vectorize historical root cause analysis reports, operation and maintenance notes, design documentation, global spacecraft accident case studies, maintenance technical documentation, relevant papers and technical standards, archived expert consultations, etc.
9. Open software -> Assistants -> Right-click SpaceHMchat and select "Edit Assistant" -> Select "Knowledge Base Settings" -> Select "SpaceHMchat Knowledge Base".

### Main File Check

[Code\API_server\LLM_4_SPS_PHM_server_api.py](../Code/API_server/LLM_4_SPS_PHM_server_api.py) is the main file that connects all-in-loop health management functions and performs client-server communication. We have already placed the main parameters in the first few lines of the file. The last step of the preparation phase is to check whether variables such as file paths and GPU numbers set at the beginning of this file correspond to your actual situation.



## Now Enjoy It!

1. Run [Original Code\15_LLM_4_SPS_PHM\API_server\LLM_4_SPS_PHM_server_api.py](../Code/API_server/LLM_4_SPS_PHM_server_api.py) file in the server.
2. Open the dialogue software, select the default assistant, and start chatting!
