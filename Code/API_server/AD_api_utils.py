"""
# function call: https://help.aliyun.com/zh/model-studio/qwen-function-calling?spm=a2c4g.11186623.0.0.23b51d1cBV7hr1#0548fe3958jh6



# 定义tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_AD_algorithms_candidates",
            "description": "当你想查询本地有哪些可供使用的异常检测算法时非常有用。",
            # "parameters": {
            #     "type": "object",
            #     "properties": {
            #         "AD_dir_path": {
            #             "type": "string",
            #             "description": "本地异常检测算法所在的目录路径，默认为/home/dy29/MyWorks_Codes/15_LLM_4_SPS_PHM/AD_repository/model/AD_algorithms。",
            #         }
            #     },
            #     "required": ["AD_dir_path"]
            # }
        }
    },
    # 这个函数的实现逻辑是：读取/home/dy29/MyWorks_Codes/15_LLM_4_SPS_PHM/AD_repository/model/AD_algorithms_params/all_AD_algorithm_params.yaml这个文件得到methods_params，随后methods_params的keys里面除了"Example_method"和"Nonthing"外的所有key便是可用方法名称，返回该列表以及这些算法所处的路径给LLM。最后也可以加一个字符串，如："关于这些算法的优劣和适用场景，我有一些建议："，让LLM继续输出一些关于这些算法的优劣、适用场景等信息。

    {
        "type": "function",
        "function": {
            "name": "get_AD_algorithm_parameters",
            "description": "当你想查询指定异常检测算法需要设置哪些参数时非常有用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "AD_algorithm_name": {
                        "type": "string",
                        "description": "指定的异常检测算法名称，比如'Informer'、'GDN'等。",
                    }
                },
                "required": ["AD_algorithm_name"]
            }
        }
    },
    # 此函数的实现逻辑是：读取/home/dy29/MyWorks_Codes/15_LLM_4_SPS_PHM/AD_repository/model/AD_algorithms_params文件夹下的 all_AD_algorithm_params.json文件里面"AD_algorithm_name"对应的参数们，返回其中的参数列表及每个参数的变量解释。返回该列表给LLM。最后也可以加一个字符串，如："关于这些参数的设置，我有一些建议："，让LLM继续输出一些关于这些参数的设置建议。

    {
        "type": "function",
        "function": {
            "name": "setting_AD_algorithm_parameters",
            "description": "当你想设置或者修改某指定异常检测算法的参数时非常有用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "AD_algorithm_name": {
                        "type": "string",
                        "description": "指定的异常检测算法名称，比如'Informer'、'GDN'等。",
                    },
                    "params_setting_dict": {
                        "type": "object",
                        "description": '''用户想要修改的参数们，字典格式，包含算法所需的所有参数及其值。格式例如：'{"learning_rate": 0.001, "batch_size": 64}'''',
                        # "type": "string",
                        # "description": "用户想要修改的参数们，JSON格式的参数值字符串，包含算法所需的所有参数及其值。格式例如：'{\"learning_rate\": 0.001, \"batch_size\": 64}'",
                    }
                },
                "required": ["AD_algorithm_name", "params_setting_dict"]
            }
        }
    },
    # 此函数的实现逻辑是：修改/home/dy29/MyWorks_Codes/15_LLM_4_SPS_PHM/AD_repository/model/AD_algorithms_params文件夹下的 all_AD_algorithm_params.json文件，修改该json里面的AD_algorithm_name分支。需要注意的是，有时候可能打错字啥的，所以把字典内key挨个检查，如果json文件里没有这个key，就跳过这个key。如果有，检查value的类型，如果value的类型不对，就跳过这个key。只有在key存在且value类型正确的情况下，才修改该key的value。

    {
        "type": "function",
        "function": {
            "name": "get_AD_algorithm_runner_train",
            "description": "当你想利用某路径的数据对某个异常检测算法进行训练时非常有用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "AD_algorithm_name": {
                        "type": "string",
                        "description": "指定的异常检测算法名称，比如'Informer'、'GDN'等。",
                    },
                    "train_data_file_path": {
                        "type": "string",
                        "description": "用于训练的训练数据文件路径。",
                    },
                    # "train_weight_file_save_path": {
                    #     "type": "string",
                    #     "description": "训练完成后保存的权重文件路径。",
                    # }
                },
                "required": ["AD_algorithm_name", "train_data_file_path"]
                # "required": ["AD_algorithm_name", "train_data_file_path", "train_weight_file_save_path"]
            }
        }
    },
    # 此函数的实现逻辑是：调用main.py文件，main调用例如Informer.py文件，Informer开头读取{AD_algorithm_name}_params.json文件，利用这些参数完成训练，训练完成以后将权重文件保存到{AD_algorithm_name}_weights.pt文件中。返回训练完成以后保存的权重文件路径等相关信息给LLM。比如“训练已经完成，权重文件保存在{AD_algorithm_name}_weights.pt文件中，您可以读取该模型并进行后续使用”

    {
        "type": "function",
        "function": {
            "name": "get_AD_algorithm_runner_test",
            "description": "当你想利用某个训练好的异常检测算法对某指定路径数据进行异常检测时非常有用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "AD_algorithm_name": {
                        "type": "string",
                        "description": "指定的异常检测算法名称，比如'Informer'、'GDN'等。",
                    },
                    "test_data_file_path": {
                        "type": "string",
                        "description": "用户想要进行异常检测的数据文件的路径。",
                    }
                },
                "required": ["AD_algorithm_name", "test_data_file_path"]
            }
        }
    }
    # 此函数的实现逻辑是：调用main.py文件，main调用例如Informer.py文件或者直接读取训练完后保存的{AD_algorithm_name}_weights.pt权重文件，加载模型后对test_data_file_path的数据进行异常检测，并保存异常检测报告。最后将异常检测的异常段落、异常分数、异常检测报告文件路径等信息返回给LLM。例如[[10,30], 45, [68,90]]或者[['2025-01-01 00:00:00', '2025-01-01 01:00:00'], '2025-01-01 01:35:05', ['2025-01-01 02:00:00', '2025-01-01 03:00:00']]。返回的字符串可以是“异常检测已经完成，异常段落为[[10,30], 45, [68,90]]，异常分数为45，异常检测报告文件保存在report_save_path文件中，您可以查看该报告以获取更多信息。”
]



# messages补充:
messages = [
    {
        "role": "system",
        "content": "你是一个能够为用户提供异常检测全流程技术支持的助手。如果用户希望知道有哪些本地异常检测算法可供选择，你可以调用'get_AD_algorithms_candidates'函数来获取可供选择的异常检测算法列表；如果用户希望知道某个异常检测算法需要设置哪些参数，你可以调用'get_AD_algorithm_parameters'函数来获取指定异常检测算法的所需参数列表；如果用户希望设置某个指定算法的参数或者说修改某算法的默认参数，你可以调用'setting_AD_algorithm_parameters'函数来设置或者修改指定异常检测算法的参数；如果用户希望利用某路径的数据对某个异常检测算法进行训练，你可以调用'get_AD_algorithm_runner_train'函数来运行指定的异常检测算法进行训练；如果用户希望利用某个训练好的异常检测算法对某指定路径数据进行异常检测，你可以调用'get_AD_algorithm_runner_test'函数来运行指定的异常检测算法进行异常检测；再这个过程中你也可以主动为用户提供一些关于各算法优劣、算法选择建议、算法参数设置提示等信息，以帮助用户更好的使用本系统。",
    },
    {
        "role": "user",
        "content": "使用本地的XXXX算法、数据。。。"
    }
]

"""

import os
import sys
import torch
import yaml
import glob
import warnings
from argparse import ArgumentParser



def find_matching_ckpt(ckpt_root, current_args):
    """
    在 ckpt_root/lightning_logs/version_* 文件夹里，从最新修改的文件夹开始遍历：
      1. 读取 version_X/hparams.yaml
      2. 和 current_args 里的属性一一对比
      3. 如果所有指定参数都相等，就在 version_X/checkpoints 下取最新的 .ckpt
    返回第一个匹配到的 ckpt 路径，找不到则返回 None。
    """
    # 拿到所有 version_x 目录
    ckpt_root = os.path.join(ckpt_root, "lightning_logs")
    version_dirs = [
        os.path.join(ckpt_root, d)
        for d in os.listdir(ckpt_root)
        if d.startswith("version_") and os.path.isdir(os.path.join(ckpt_root, d))
    ]
    # 按文件夹的修改时间从新到旧排序
    version_dirs.sort(key=lambda d: os.path.getmtime(d), reverse=True)

    for vdir in version_dirs:
        hparam_path = os.path.join(vdir, "hparams.yaml")
        if not os.path.isfile(hparam_path):
            continue

        # 1. 读取超参
        with open(hparam_path, "r") as fp:
            # saved_hp = yaml.safe_load(fp) or {}   # args: !!python/object:argparse.Namespace这种python对象使用safe_load会报错
            saved_hp = yaml.unsafe_load(fp) or {}
            args_namespace = saved_hp.get("args", {})
            # 安全地转换为字典
            if hasattr(args_namespace, '__dict__'):
                args_dict = vars(args_namespace)
            else:
                args_dict = args_namespace if isinstance(args_namespace, dict) else {}


        # 2. 比对：这里只对比 current_args 中出现的字段；你可以定制 keys 列表
        match = True
        for key, val in vars(current_args).items():
            # 如果 hparams.yaml 里没有这个 key 或者值不一致，就跳过这个 version
            # 但是不需要匹配‘AD_threshold’这个参数
            if key in ["AD_threshold", "missvalue"]:
                continue
            if key in args_dict:
                if args_dict[key] != val:
                    match = False
                    print(f"Skipping version {vdir} due to mismatch on key '{key}': {args_dict[key]} != {val}")
                    break
        if not match:
            continue

        # 3. 找到最新的 ckpt 文件
        ckpt_files = glob.glob(os.path.join(vdir, "checkpoints", "*.ckpt"))
        if not ckpt_files:
            continue
        # 按文件修改时间取最新的那个
        best_ckpt = max(ckpt_files, key=os.path.getmtime)
        return best_ckpt

    raise FileNotFoundError(f"No matching ckpt found in {ckpt_root}")
    return None






def get_AD_algorithms_candidates(arguments):
    """
    获取本地可供使用的异常检测算法列表。
    # 这个函数的实现逻辑是：读取/home/dy29/MyWorks_Codes/15_LLM_4_SPS_PHM/AD_repository/model/AD_algorithms_params/all_AD_algorithm_params.yaml这个文件得到methods_params，随后methods_params的keys里面除了"Example_method"和"Nonthing"外的所有key便是可用方法名称，返回该列表以及这些算法所处的路径给LLM。最后也可以加一个字符串，如："关于这些算法的优劣和适用场景，我有一些建议："，让LLM继续输出一些关于这些算法的优劣、适用场景等信息。
    """
    import os
    dir_path = '/home/dy29/MyWorks_Codes/15_LLM_4_SPS_PHM/AD_repository/model/AD_algorithms_params/all_AD_algorithm_params.yaml'
    
    import yaml
    with open(dir_path, 'r') as file:
        methods_params = yaml.safe_load(file)
    # 获取所有算法名称，排除"Example_method"和"Nonthing"
    algorithms_names = [name for name in methods_params.keys() if name not in ["Common_configs", "Example_method", "Nonthing"]]

    result_str = "检查了本地的算法库，可供用于进行异常检测的算法列表为：" + str(algorithms_names) + "。同时，我也可以为你提供一些关于这些算法的介绍、优劣势、适用场景、选用建议如下："

    return result_str



def get_AD_algorithm_parameters(arguments):
    """
    获取指定异常检测算法所需的参数列表。
    # 此函数的实现逻辑是：读取/home/dy29/MyWorks_Codes/15_LLM_4_SPS_PHM/AD_repository/model/AD_algorithms_params文件夹下的 all_AD_algorithm_params.yaml文件里面"AD_algorithm_name"对应的参数们，返回其中的参数列表及每个参数的变量解释。返回该列表给LLM。最后也可以加一个字符串，如："关于这些参数的设置，我有一些建议："，让LLM继续输出一些关于这些参数的设置建议。
    """
    import os
    import json
    import yaml
    print("get_AD_algorithm_parameters函数以及被调用")

    AD_algorithm_name = arguments["AD_algorithm_name"].strip()
    AD_params_dir_path = '/home/dy29/MyWorks_Codes/15_LLM_4_SPS_PHM/AD_repository/model/AD_algorithms_params/all_AD_algorithm_params.yaml'

    # 打开yaml文件
    with open(AD_params_dir_path, 'r') as file:
        methods_params = yaml.safe_load(file)

    Common_config = methods_params["Common_configs"]
    Method_config = methods_params[AD_algorithm_name]
    All_config = {**Common_config, **Method_config}

    result_str = "指定的异常检测算法 "+ AD_algorithm_name + " 需要设置的参数有："
    for num, (key, value) in enumerate(All_config.items()):
        result_str += f"\n{num+1}.{key}: {value['description_CN']} (默认值: {value['value']})。"
        
    result_str += "。同时，我也可以为你提供一些关于这些参数的设置建议如下："

    return result_str



def setting_AD_algorithm_parameters(arguments):
    """
    设置或修改指定异常检测算法的参数。
    # 此函数的实现逻辑是：修改/home/dy29/MyWorks_Codes/15_LLM_4_SPS_PHM/AD_repository/model/AD_algorithms_params文件夹下的 all_AD_algorithm_params.json文件，修改该json里面的AD_algorithm_name分支。需要注意的是，有时候可能打错字啥的，所以把字典内key挨个检查，如果json文件里没有这个key，就跳过这个key。如果有，检查value的类型，如果value的类型不对，就跳过这个key。只有在key存在且value类型正确的情况下，才修改该key的value。
    """
    import os
    import json
    print("setting_AD_algorithm_parameters函数被调用")

    AD_algorithm_name = arguments["AD_algorithm_name"].strip()
    params_setting_dict = arguments["params_setting_dict"]
    params_file_path = '/home/dy29/MyWorks_Codes/15_LLM_4_SPS_PHM/AD_repository/model/AD_algorithms_params/all_AD_algorithm_params.yaml'

    # 读取现有参数
    with open(params_file_path, 'r') as file:
        # existing_params = json.load(file)
        existing_params = yaml.safe_load(file)


    # 先把用户要求使用的算法写入yaml文件
    if AD_algorithm_name in ["TCN", "GRU", "Transformer", "Informer", "Autoformer", "PatchTST", "DLinear"]:
        existing_params["Common_configs"]["temporal_block"]["value"] = AD_algorithm_name
        existing_params["Common_configs"]["spatial_block"]["value"] = "Nothing"
    elif AD_algorithm_name in ["GCN", "GAT", "GIN", "SGC", "MTGNN", "FourierGNN", "StemGNN", "GraphWaveNet"]:
        existing_params["Common_configs"]["temporal_block"]["value"] = "Nothing"
        existing_params["Common_configs"]["spatial_block"]["value"] = AD_algorithm_name
    else:
        raise ValueError(f"指定的异常检测算法 '{AD_algorithm_name}' 不在已知的算法列表中。请检查算法名称是否正确。")
    # with open(params_file_path, "w") as f:
    #     yaml.dump(existing_params, f, default_flow_style=False, allow_unicode=True)


    # 如果params_setting_dict是字符串，则尝试将其转换为字典，如果是object，则直接使用
    if isinstance(params_setting_dict, str):

        # 开头和结尾可能有多余的空格和引号，先去除，开始和结束都要是{}才行
        params_setting_dict = params_setting_dict.strip().strip('"').strip("'")
        if not (params_setting_dict.startswith('{') and params_setting_dict.endswith('}')):
            raise ValueError('params_setting_dict is not a valid dict string: ', params_setting_dict)

        # 尝试将字符串转换为字典
        try:
            params_setting_dict = json.loads(params_setting_dict)
        except:
            # 如果转换失败，需要在特殊字符前加上转义符，使其变为可供json解析的字符串
            try:
                # 方法1: 尝试修复常见的引号问题
                # 这处理未正确转义的引号和使用单引号而非双引号的情况
                import re
                # 先将单引号替换为双引号(如果外层已经是双引号则跳过)
                if not (params_setting_dict.startswith('"') and params_setting_dict.endswith('"')):
                    params_setting_dict = params_setting_dict.replace("'", '"')
                # 处理嵌套的未转义双引号
                # 查找形如 {"key": "value"} 中value内部未转义的引号
                pattern = r'(?<=":\s*")(?:[^"\\]|\\.)*?(?=")'
                for match in re.finditer(pattern, params_setting_dict):
                    value = match.group(0)
                    escaped_value = value.replace('"', '\\"')
                    params_setting_dict = params_setting_dict.replace(f'"{value}"', f'"{escaped_value}"')
                
                params_setting_dict = json.loads(params_setting_dict)
            except:
                try:
                    # 方法2: 使用eval谨慎处理字符串
                    # 注意: eval有安全风险，这里仅用于处理简单的字典字符串
                    import ast
                    params_dict = ast.literal_eval(params_setting_dict)
                    params_setting_dict = params_dict
                except:
                    raise ValueError('params_setting_dict is not a valid dict string: ', params_setting_dict)
    print("params_setting_dict:", params_setting_dict)
    print("type of params_setting_dict:", type(params_setting_dict))

    # 更新参数
    params_type_not_match = []
    not_in_AD_algorithm_name = []
    not_in_Common_config = []
    for key, value in params_setting_dict.items():
        if key in existing_params[AD_algorithm_name].keys():
            if isinstance(existing_params[AD_algorithm_name][key]["value"], type(value)):
                existing_params[AD_algorithm_name][key]["value"] = value
            else:
                params_type_not_match.append(key)
                print(f"参数 '{key}' 的类型不匹配，跳过该参数。预期类型为 {type(existing_params[key])}，实际类型为 {type(value)}。")
        else:
            not_in_AD_algorithm_name.append(key)

        if key in existing_params["Common_configs"].keys():
            if isinstance(existing_params["Common_configs"][key]["value"], type(value)):
                existing_params["Common_configs"][key]["value"] = value
            else:
                params_type_not_match.append(key)
                print(f"参数 '{key}' 的类型不匹配，跳过该参数。预期类型为 {type(existing_params['Common_configs'][key])}，实际类型为 {type(value)}。")
        else:
            not_in_Common_config.append(key)

    not_in_all_params = list(set(not_in_AD_algorithm_name).intersection(set(not_in_Common_config)))

    # 保存更新后的参数
    # with open(params_file_path, 'w') as file:
    #     json.dump(existing_params, file, indent=4)
    with open(params_file_path, "w") as f:
        yaml.dump(existing_params, f, default_flow_style=False, allow_unicode=True)

    result_str = "指定的异常检测算法 '" + AD_algorithm_name + "' 的参数已经成功设置或修改。"

    if params_type_not_match != [] or not_in_all_params != []:
        result_str += "但是以下参数由于格式不匹配或参数不存在的设置失败：" + str(params_type_not_match+not_in_all_params) + "。如果这些参数不重要，建议忽略这些参数，否则需要确认格式输入。"
    
    return result_str



def get_AD_algorithm_runner_train(arguments):
    """
    运行指定异常检测算法进行训练。
    "required": ["AD_algorithm_name", "train_data_file_path"]
    # 此函数的实现逻辑是：调用main.py文件，main调用例如Informer.py文件，Informer开头读取{AD_algorithm_name}_params.json文件，利用这些参数完成训练，训练完成以后将权重文件保存到{AD_algorithm_name}_weights.pt文件中。返回训练完成以后保存的权重文件路径等相关信息给LLM。比如“训练已经完成，权重文件保存在{AD_algorithm_name}_weights.pt文件中，您可以读取该模型并进行后续使用”
    """
    from AD_repository.main import set_args, main_4_LLM_calling
    # import main

    AD_algorithm_name = arguments["AD_algorithm_name"].strip()
    train_data_file_path = arguments["train_data_file_path"].strip()
    # train_data_file_path = arguments["train_data_file_path"].strip().strip('"').strip("'").strip()


    # 获取原始参数
    args = set_args()
    from AD_repository.main_sub import update_args_from_yaml
    args = update_args_from_yaml(yaml_path='/home/dy29/MyWorks_Codes/15_LLM_4_SPS_PHM/AD_repository/model/AD_algorithms_params/all_AD_algorithm_params.yaml', args=args)


    # 修改参数
    if AD_algorithm_name not in [args.spatial_block, args.temporal_block]:
        raise ValueError(f"指定的异常检测算法与先前设定的算法不一致。或者先前并未进行参数设置。请先设置或检查参数。AD_algorithm_name: {AD_algorithm_name}, spatial_block: {args.spatial_block}, temporal_block: {args.temporal_block}")


    # 识别root_path, data_path, data_name并设置
    root_path, remaining_path = os.path.splitdrive(train_data_file_path)
    # if not root_path:  # 如果没有驱动器部分，直接使用 '/'
    #     root_path = '/'
    # # 分解剩余路径
    parts = remaining_path.split(os.sep)
    root_path = os.path.join(root_path, *parts[:3])  # 前3部分作为 root_path
    data_path = os.path.join(*parts[3:-1])  # 中间部分作为 data_path
    file_name = parts[-1]  # 最后一部分作为 file_name
    if file_name.endswith('.csv'):
        file_name = file_name[:-4]
    if file_name.endswith('_Test') or file_name.endswith('_test'):
        file_name = file_name[:-5]
    if file_name.endswith('_Train') or file_name.endswith('_train'):
        file_name = file_name[:-6]
    args.root_path = root_path
    args.data_path = data_path
    args.data_name = file_name


    # 设置设备并运行main函数
    devices = args.devices
    # devices = [3]  # 使用哪些GPU
    main_4_LLM_calling(devices=devices, args=args, train_or_test="train")

    result_str = f"训练已经完成，权重文件保存在{args.ckpt_save_path}，您可以读取该模型用于异常检测"

    return result_str



def get_AD_algorithm_runner_test(arguments):
    """
    读取训练好的异常检测算法模型，对指定数据进行异常检测。
    ["AD_algorithm_name", "test_data_file_path"]
    此函数的实现逻辑是：调用main.py文件，main调用例如Informer.py文件或者直接读取训练完后保存的{AD_algorithm_name}_weights.pt权重文件，加载模型后对test_data_file_path的数据进行异常检测，并保存异常检测报告。最后将异常检测的异常段落、异常分数、异常检测报告文件路径等信息返回给LLM。例如[[10,30], 45, [68,90]]或者[['2025-01-01 00:00:00', '2025-01-01 01:00:00'], '2025-01-01 01:35:05', ['2025-01-01 02:00:00', '2025-01-01 03:00:00']]。返回的字符串可以是“异常检测已经完成，异常段落为[[10,30], 45, [68,90]]，异常分数为45，异常检测报告文件保存在report_save_path文件中，您可以查看该报告以获取更多信息。”

    :param arguments: 函数参数
    :return result_markdown: 异常检测结果的Markdown格式字符串。注意！如果使用第二个结果返回客户端，需要指定md不是str：return jsonify(format="markdown", payload={"text": markdown_str})

    """
    # :return result_str: 异常检测结果字符串


    from AD_repository.main import set_args, main_4_LLM_calling

    AD_algorithm_name = arguments["AD_algorithm_name"].strip()
    test_data_file_path = arguments["test_data_file_path"].strip()
    # test_data_file_path = arguments["test_data_file_path"].strip().strip('"').strip("'").strip()


    # 获取原始参数
    args = set_args()
    from AD_repository.main_sub import update_args_from_yaml
    args = update_args_from_yaml(yaml_path='/home/dy29/MyWorks_Codes/15_LLM_4_SPS_PHM/AD_repository/model/AD_algorithms_params/all_AD_algorithm_params.yaml', args=args)
    report_save_path = args.report_save_path


    # 修改参数
    if AD_algorithm_name not in [args.spatial_block, args.temporal_block]:
        raise ValueError(f"指定的异常检测算法与先前设定的算法不一致。或者先前并未进行参数设置。请先设置或检查参数。")


    # 识别root_path, data_path, data_name并设置
    root_path, remaining_path = os.path.splitdrive(test_data_file_path)
    # if not root_path:  # 如果没有驱动器部分，直接使用 '/'
    #     root_path = '/'
    # # 分解剩余路径
    parts = remaining_path.split(os.sep)
    root_path = os.path.join(root_path, *parts[:3])  # 前3部分作为 root_path
    data_path = os.path.join(*parts[3:-1])  # 中间部分作为 data_path
    file_name = parts[-1]  # 最后一部分作为 file_name
    if file_name.endswith('.csv'):
        file_name = file_name[:-4]
    if file_name.endswith('_Test') or file_name.endswith('_test'):
        file_name = file_name[:-5]
    if file_name.endswith('_Train') or file_name.endswith('_train'):
        file_name = file_name[:-6]
    args.root_path = root_path
    args.data_path = data_path
    args.data_name = file_name


    # 设置设备并运行main函数
    devices = args.devices
    # devices = [3]  # 使用哪些GPU
    main_4_LLM_calling(devices=devices, 
                       args=args, 
                       train_or_test="test", 
                       ckpt_path_4_test=args.ckpt_save_path
                       )
    
    # 读取异常检测结果文件
    AD_result_path = os.path.join(args.table_save_path, "AD_result.json")
    import json
    with open(AD_result_path, 'r') as file:
        anomaly_result = json.load(file)
        anomaly_timestamp_list = anomaly_result["anomaly_timestamp_list"]
        anomaly_ratio = anomaly_result["anomaly_ratio"]
        threshold = anomaly_result["threshold"]
        recommend_threshold = anomaly_result["recommend_threshold"]
        
    result_str = ("异常检测已经完成，检测出的异常段落为：" + str(anomaly_timestamp_list) 
    + "。\n异常段落占比为：" + str(anomaly_ratio) 
    + "。\n所使用的异常检测算法为：" + AD_algorithm_name
    + "。\n异常检测的阈值为：" + str(threshold)
    + "。\n异常检测报告文件保存在" + str(report_save_path) + "文件中，您可以查看该报告以获取更多信息。")

    AD_result_png_path = args.plot_save_path + '/AD_result_figure.png'
    AD_result_png_url = AD_result_png_path.replace("/data/DiYi/MyWorks_Results/SPS_AD_LLM_Project", "http://localhost:1999/SPS_AD_LLM_Project")
    print("AD_result_png_url:", AD_result_png_url)
    # 路径里有空格时要把它们替换，否则 Markdown 解析器不一定能识别。
    AD_result_png_url = AD_result_png_url.replace(" ", "%20")

    # 同理，将anomaly_timestamp_list保存为TXT文件
    anomaly_timestamp_list_str = "\n".join([str(item) for item in anomaly_timestamp_list])
    anomaly_timestamp_list_path = os.path.join(args.table_save_path, "anomaly_timestamp_list.txt")
    with open(anomaly_timestamp_list_path, 'w') as file:
        file.write(anomaly_timestamp_list_str)
    anomaly_timestamp_list_url = anomaly_timestamp_list_path.replace("/data/DiYi/MyWorks_Results/SPS_AD_LLM_Project", "http://localhost:1999/SPS_AD_LLM_Project")
    anomaly_timestamp_list_url = anomaly_timestamp_list_url.replace(" ", "%20")
    print("anomaly_timestamp_list_url:", anomaly_timestamp_list_url)

    result_markdown = f"""
**所检测出异常段落：**
```txt
{anomaly_timestamp_list[0]},
{anomaly_timestamp_list[1]}
{anomaly_timestamp_list[2]}
{anomaly_timestamp_list[3]}
{anomaly_timestamp_list[4]}
{anomaly_timestamp_list[5]}
{anomaly_timestamp_list[6]}
{anomaly_timestamp_list[7]}
{anomaly_timestamp_list[8]}
{anomaly_timestamp_list[9]}
...（预览已折叠，完整结果请下载TXT文件）
```
完整结果下载：[📥 单元素列表表示点异常、双元素列表表示段落异常（TXT）]({anomaly_timestamp_list_url})

**异常检测信息概览：**
所使用异常检测算法为：{AD_algorithm_name}；
所使用异常检测的阈值为：{threshold}。

**异常检测结果可视化：**

![异常检测结果图](<{AD_result_png_url}>)

**补充说明：**
完整的异常检测报告文件保存在{report_save_path}文件中，您可以查看该报告以获取更多信息。
如果您觉得异常检测结果不理想，这里有一些建议（注意若参数得到更改可能需要重新训练）：
"""
# 异常段落占比为：{anomaly_ratio}；

    # if recommend_threshold != None or recommend_threshold != "None":
    #     result_markdown += f"""1. 您可以尝试调整异常检测算法的阈值，当前使用的阈值为 {threshold}，推荐的阈值为 {recommend_threshold}。"""

    # return result_str, result_markdown
    return result_markdown






def get_messages_and_tools_for_anomaly_detection(User_prompt, User_question, RAG_materials):
    """
    获取异常检测任务的消息和工具列表。
    """
    messages = [
    {
        "role": "system",
        "content": """
        你是一个能够为用户提供异常检测全流程技术支持的助手。如果用户希望知道有哪些本地异常检测算法可供选择，你可以调用'get_AD_algorithms_candidates'函数来获取可供选择的异常检测算法列表；如果用户希望知道某个异常检测算法需要设置哪些参数，你可以调用'get_AD_algorithm_parameters'函数来获取指定异常检测算法的所需参数列表；如果用户希望设置某个指定算法的参数或者说修改某算法的默认参数，你可以调用'setting_AD_algorithm_parameters'函数来设置或者修改指定异常检测算法的参数；如果用户希望利用某路径的数据对某个异常检测算法进行训练，你可以调用'get_AD_algorithm_runner_train'函数来运行指定的异常检测算法进行训练；如果用户希望利用某个训练好的异常检测算法对某指定路径数据进行异常检测，你可以调用'get_AD_algorithm_runner_test'函数来运行指定的训练好的异常检测算法进行异常检测。
        注意1：在所有过程中你也可以主动为用户提供一些关于各算法优劣、算法选择建议、算法参数设置提示等信息，以帮助用户更好的使用本系统。
        注意2：这些函数名称都是供你后台调用的，不需出现在你和用户的对话中。
        注意3：直接调用最相关最能完成用户目的那个函数，一次响应直接调用一个最好。除非实在没办法的情况下才可以在一次响应中同时调用多个函数：在返回的 tool_calls 列表中包含多个函数调用的 JSON 对象，系统将按照列表中的顺序依次执行这些函数。
        注意4：优先调用最相关且最能满足用户需求的单个函数——在一次响应中只调用一个函数通常是最佳实践。只有在确实无法通过单个函数完成任务时，才在 `tool_calls` 列表中按序添加多个函数调用json对象，系统将依序执行。例如：
        "tool_calls": [
        {
            "function": {
                "name": "get_AD_algorithms_candidates",
                "arguments": "{}"
            },
            "index": 0,
            "id": "call_***************",
            "type": "function"
        },
        {
            "function": {
                "name": "get_AD_algorithm_parameters",
                "arguments": "{\"AD_algorithm_name\": \"MTGNN\"}"
            },
            "index": 1,
            "id": "call_***************",
            "type": "function"
        },
        {
            "function": {
                "name": "setting_AD_algorithm_parameters",
                "arguments": "{\"AD_algorithm_name\": \"MTGNN\", \"params_setting_dict\": \"{\"learning_rate\": 0.001, \"batch_size\": 64}\"}"
            },
            "index": 2,
            "id": "call_***************",
            "type": "function"
        }
        """,
    },
    {
        "role": "user",
        'content': User_question
    }
    ]

    tools = [
    {
        "type": "function",
        "function": {
            "name": "get_AD_algorithms_candidates",
            "description": "当你想查询本地有哪些可供使用的异常检测算法时非常有用。",
            # "parameters": {
            #     "type": "object",
            #     "properties": {
            #         "AD_dir_path": {
            #             "type": "string",
            #             "description": "本地异常检测算法所在的目录路径，默认为/home/dy29/MyWorks_Codes/15_LLM_4_SPS_PHM/AD_repository/model/AD_algorithms。",
            #         }
            #     },
            #     "required": ["AD_dir_path"]
            # }
        }
    },
    # 这个函数的实现逻辑是：读取/home/dy29/MyWorks_Codes/15_LLM_4_SPS_PHM/AD_repository/model/AD_algorithms_params/all_AD_algorithm_params.yaml这个文件得到methods_params，随后methods_params的keys里面除了"Example_method"和"Nonthing"外的所有key便是可用方法名称，返回该列表以及这些算法所处的路径给LLM。最后也可以加一个字符串，如："关于这些算法的优劣和适用场景，我有一些建议："，让LLM继续输出一些关于这些算法的优劣、适用场景等信息。

    {
        "type": "function",
        "function": {
            "name": "get_AD_algorithm_parameters",
            "description": "当你想查询指定异常检测算法需要设置哪些参数时非常有用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "AD_algorithm_name": {
                        "type": "string",
                        "description": "指定的异常检测算法名称，比如'Informer'、'GDN'等。",
                    }
                },
                "required": ["AD_algorithm_name"]
            }
        }
    },
    # 此函数的实现逻辑是：读取/home/dy29/MyWorks_Codes/15_LLM_4_SPS_PHM/AD_repository/model/AD_algorithms_params文件夹下的 all_AD_algorithm_params.json文件里面"AD_algorithm_name"对应的参数们，返回其中的参数列表及每个参数的变量解释。返回该列表给LLM。最后也可以加一个字符串，如："关于这些参数的设置，我有一些建议："，让LLM继续输出一些关于这些参数的设置建议。

    {
        "type": "function",
        "function": {
            "name": "setting_AD_algorithm_parameters",
            "description": "当你想设置或者修改某指定异常检测算法的参数时非常有用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "AD_algorithm_name": {
                        "type": "string",
                        "description": "指定的异常检测算法名称，比如'Informer'、'MTGNN'等。",
                    },
                    "params_setting_dict": {
                        "type": "object",
                        "description": """用户想要修改的参数们，字典格式，包含算法所需的所有参数及其值。格式例如：'{"learning_rate": 0.001, "batch_size": 64}'""",
                        # "type": "string",
                        # "description": "用户想要修改的参数们，JSON格式的参数值字符串，包含算法所需的所有参数及其值。格式例如：'{\"learning_rate\": 0.001, \"batch_size\": 64}'",
                    }
                },
                "required": ["AD_algorithm_name", "params_setting_dict"]
            }
        }
    },
    # 此函数的实现逻辑是：修改/home/dy29/MyWorks_Codes/15_LLM_4_SPS_PHM/AD_repository/model/AD_algorithms_params文件夹下的 all_AD_algorithm_params.json文件，修改该json里面的AD_algorithm_name分支。需要注意的是，有时候可能打错字啥的，所以把字典内key挨个检查，如果json文件里没有这个key，就跳过这个key。如果有，检查value的类型，如果value的类型不对，就跳过这个key。只有在key存在且value类型正确的情况下，才修改该key的value。

    {
        "type": "function",
        "function": {
            "name": "get_AD_algorithm_runner_train",
            "description": "当你想利用某路径的数据对某个异常检测算法进行训练时非常有用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "AD_algorithm_name": {
                        "type": "string",
                        "description": "指定的异常检测算法名称，比如'Informer'、'GDN'等。",
                    },
                    "train_data_file_path": {
                        "type": "string",
                        "description": "用于训练的训练数据文件路径。",
                    },
                    # "train_weight_file_save_path": {
                    #     "type": "string",
                    #     "description": "训练完成后保存的权重文件路径。",
                    # }
                },
                "required": ["AD_algorithm_name", "train_data_file_path"]
                # "required": ["AD_algorithm_name", "train_data_file_path", "train_weight_file_save_path"]
            }
        }
    },
    # 此函数的实现逻辑是：调用main.py文件，main调用例如Informer.py文件，Informer开头读取{AD_algorithm_name}_params.json文件，利用这些参数完成训练，训练完成以后将权重文件保存到{AD_algorithm_name}_weights.pt文件中。返回训练完成以后保存的权重文件路径等相关信息给LLM。比如“训练已经完成，权重文件保存在{AD_algorithm_name}_weights.pt文件中，您可以读取该模型并进行后续使用”

    {
        "type": "function",
        "function": {
            "name": "get_AD_algorithm_runner_test",
            "description": "当你想利用某个训练好的异常检测算法对某指定路径数据进行异常检测时非常有用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "AD_algorithm_name": {
                        "type": "string",
                        "description": "指定的异常检测算法名称，比如'Informer'、'GDN'等。",
                    },
                    "test_data_file_path": {
                        "type": "string",
                        "description": "用户想要进行异常检测的数据文件的路径。",
                    }
                },
                "required": ["AD_algorithm_name", "test_data_file_path"]
            }
        }
    }
    # 此函数的实现逻辑是：调用main.py文件，main调用例如Informer.py文件或者直接读取训练完后保存的{AD_algorithm_name}_weights.pt权重文件，加载模型后对test_data_file_path的数据进行异常检测，并保存异常检测报告。最后将异常检测的异常段落、异常分数、异常检测报告文件路径等信息返回给LLM。例如[[10,30], 45, [68,90]]或者[['2025-01-01 00:00:00', '2025-01-01 01:00:00'], '2025-01-01 01:35:05', ['2025-01-01 02:00:00', '2025-01-01 03:00:00']]。返回的字符串可以是“异常检测已经完成，异常段落为[[10,30], 45, [68,90]]，异常分数为45，异常检测报告文件保存在report_save_path文件中，您可以查看该报告以获取更多信息。”
    ]
    return messages, tools





def execute_tools_for_anomaly_detection(completion, tool_if_stream, Tool_already_id):
    """
    执行工具调用并处理结果。

    :param completion: LLM的响应对象，包含工具调用信息
    :param tool_if_stream: 输入的completion是否为流式响应
    :param Tool_already_id: 已经正在执行的工具调用ID列表，有时cherry等客户端在模型训练时长时间没得到回应会再次发送问题请求响应，不要重复执行

    :return results: list, 包含每个工具调用的执行结果
    :return tool_call_ids: list, 工具调用的ID列表，用于后续函数调用
    :return reasoning_content: LLM在调用工具时的思考过程
    :return answer_content: LLM在调用工具时的中间回复，并非最终的tool执行结果，tool执行结果是上面的results
    """
    reasoning_content = ""  # 定义完整思考过程
    answer_content = ""     # 定义完整回复
    tool_info = []          # 存储工具调用信息

    if not tool_if_stream:
        answer_content = completion.choices[0].message.content  # 获取回复内容
        if "<think>" in answer_content and "</think>" in answer_content:
            # 提取思考部分
            reasoning_content = answer_content.split("<think>", 1)[1].split("</think>")[0].strip()
            # 提取回答部分
            # answer_content = answer_content.split("</think>", 1)[1].strip()

        for tool_call in completion.choices[0].message.tool_calls:
            index = tool_call.index  # 工具调用索引，用于并行调用
            # 动态扩展工具信息存储列表
            while len(tool_info) <= index:
                tool_info.append({})
            # 收集工具调用ID（用于后续函数调用）
            if tool_call.id:
                tool_info[index]['id'] = tool_call.id
            # 收集函数名称（用于后续路由到具体函数）
            if tool_call.function and tool_call.function.name:
                tool_info[index]['name'] = tool_call.function.name
            # 收集函数参数（JSON字符串格式，需要后续解析）
            if tool_call.function and tool_call.function.arguments:
                tool_info[index]['arguments'] = tool_call.function.arguments

    else:
        warnings.warn("生成tool的LLM不建议开启流式响应，可能导致性能降低，因为添加入messages的操作较繁琐我还没编写，你可以参考https://help.aliyun.com/zh/model-studio/qwen-function-calling?spm=a2c4g.11186623.0.0.23b51d1cBV7hr1#dad2dbe656yhp进行改进")
        for chunk in completion:
            if not chunk.choices:
                # 处理用量统计信息
                print("\n"+"="*20+"Usage"+"="*20)
                print(chunk.usage)
            else:
                delta = chunk.choices[0].delta

                # 处理AI的思考过程（链式推理）
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                    reasoning_content += delta.reasoning_content
                    # print(delta.reasoning_content,end="",flush=True)  # 实时输出思考过程

                # 处理最终回复内容
                else:
                    answer_content += delta.content
                    # print(delta.content,end="",flush=True)  # 流式输出回复内容
                    
                    # 处理工具调用信息（支持并行工具调用）
                    if delta.tool_calls is not None:
                        for tool_call in delta.tool_calls:
                            index = tool_call.index  # 工具调用索引，用于并行调用
                            
                            # 动态扩展工具信息存储列表
                            while len(tool_info) <= index:
                                tool_info.append({})
                            
                            # 收集工具调用ID（用于后续函数调用） 因为是流式，同一个id的tool调用要叠加、拼接起来
                            if tool_call.id:
                                tool_info[index]['id'] = tool_info[index].get('id', '') + tool_call.id
                            
                            # 收集函数名称（用于后续路由到具体函数）因为是流式，同一个id的tool调用要叠加、拼接起来
                            if tool_call.function and tool_call.function.name:
                                tool_info[index]['name'] = tool_info[index].get('name', '') + tool_call.function.name
                            
                            # 收集函数参数（JSON字符串格式，需要后续解析）因为是流式，同一个id的tool调用要叠加、拼接起来
                            if tool_call.function and tool_call.function.arguments:
                                tool_info[index]['arguments'] = tool_info[index].get('arguments', '') + tool_call.function.arguments
                
    # 开始执行工具调用
    results = []
    tool_call_ids =[]
    for tool in tool_info:
        if tool['id'] in Tool_already_id:
            print(f"工具调用ID {tool['id']} 已经在执行中，跳过重复执行。")
            continue
        # 调用对应的函数
        tool_call_ids.append(tool['id'])
        func_name = tool['name']
        func_args = tool['arguments'] if 'arguments' in tool else {}
        if isinstance(func_args, str):
            # 解析函数参数为字典格式
            import json
            try:
                func_args = json.loads(func_args)
            except json.JSONDecodeError as e:
                print(f"参数解析错误: {e}")
                continue
        
        # 这里假设函数名和参数都是正确的，实际使用中需要添加错误处理
        if func_name == "get_AD_algorithms_candidates":
            result = get_AD_algorithms_candidates(func_args)
        elif func_name == "get_AD_algorithm_parameters":
            result = get_AD_algorithm_parameters(func_args)
        elif func_name == "setting_AD_algorithm_parameters":
            result = setting_AD_algorithm_parameters(func_args)
        elif func_name == "get_AD_algorithm_runner_train":
            result = get_AD_algorithm_runner_train(func_args)
        elif func_name == "get_AD_algorithm_runner_test":
            result = get_AD_algorithm_runner_test(func_args)
        else:
            result = f"未知函数调用：{func_name}"
        
        # results.append(func_name + " 执行完毕，结果: " + result + '。')
        results.append(result)

    # 返回最终的思考过程、回复内容和工具调用结果
    return results, tool_call_ids, reasoning_content, answer_content










