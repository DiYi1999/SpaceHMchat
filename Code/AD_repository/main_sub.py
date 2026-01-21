

def set_node_num(data_name, Decompose=None, Wavelet_level=None, if_timestamp=False, if_add_work_condition=False, reco_form=False):
    """

    Args:
        data_name:
        Decompose:
        Wavelet_level:
        if_timestamp:
        reco_form:

    Returns:
        sensor_num:
        node_num:
        timestamp_dim:

    """
    if data_name == 'BIRDS':
        sensor_num = 18
        timestamp_dim = 1
    elif data_name == 'BIRDS_10sensor':
        sensor_num = 10
        timestamp_dim = 1
    elif data_name == 'MSL':
        sensor_num = 55
        timestamp_dim = 6
    elif data_name == 'SMAP':
        sensor_num = 25
        timestamp_dim = 6
    elif data_name == 'SWaT':
        sensor_num = 51
        timestamp_dim = 1
    elif data_name == 'SMD':
        sensor_num = 38
        timestamp_dim = 1
    elif data_name == 'PSM':
        sensor_num = 25
        timestamp_dim = 1
    elif data_name == 'ETTh1' or data_name == 'ETTh2':
        sensor_num = 7
        timestamp_dim = 4
    elif data_name == 'ETTm1' or data_name == 'ETTm2':
        sensor_num = 7
        timestamp_dim = 5
    elif data_name == 'weather':
        sensor_num = 21
        timestamp_dim = 5
    elif data_name == 'Electricity':
        sensor_num = 321
        timestamp_dim = 1
    elif data_name == 'exchange_rate':
        sensor_num = 8
        timestamp_dim = 1
    elif data_name == 'traffic':
        sensor_num = 862
        timestamp_dim = 4
    elif data_name == 'solar_energy':
        sensor_num = 137
        timestamp_dim = 1
    # elif data_name == 'XJTU_SPS_for_Modeling_and_PINN_1Hz':
    #     sensor_num = 33
    #     timestamp_dim = 5
    #     work_condition_dim = 4
    # elif data_name == 'XJTU_SPS_for_Modeling_and_PINN_05Hz':
    #     sensor_num = 33
    #     timestamp_dim = 5
    #     work_condition_dim = 4
    # elif data_name == 'XJTU_SPS_for_Modeling_and_PINN_01Hz':
    #     sensor_num = 33
    #     timestamp_dim = 5
    #     work_condition_dim = 4
    elif 'XJTU-SPS for' in data_name:
        sensor_num = 33
        timestamp_dim = 5
        work_condition_dim = 4

    else:
        raise ValueError(f'node_num is not defined， 请在main.py中定义node_num, data_name={data_name}')

    if Decompose is not None:
        if Decompose == 'STL':
            node_num = sensor_num * 3
        elif Decompose == 'Wavelet':
            node_num = sensor_num * (Wavelet_level + 1)
        elif Decompose == 'WaveletPacket':
            node_num = sensor_num * (2**Wavelet_level)
        else:
            node_num = sensor_num

    if if_timestamp:
        node_num = node_num + timestamp_dim

    if if_add_work_condition:
        node_num = node_num + work_condition_dim

    return sensor_num, node_num, timestamp_dim



def get_plot_pram(args):
    """
    通过args得到画图的参数

    Args:
        args:

    Returns:
        plot_pram: 一个字典

    """
    data_name = args.data_name
    if data_name == 'XJTU_SPS_for_Modeling_and_PINN_1Hz':
        plot_pram = {'figsize': (4, 1.5), 'dpi': 1200, 'fontsize': 12}
    elif data_name == 'XJTU_SPS_for_Modeling_and_PINN_01Hz':
        plot_pram = {'figsize': (6, 1.5), 'dpi': 1200, 'fontsize': None}
    elif data_name == 'XJTU_SPS_for_Modeling_and_PINN_05Hz':
        plot_pram = {'figsize': (4, 1.5), 'dpi': 1200, 'fontsize': 12}
    elif data_name == 'XJTU-SPS for AD':
        plot_pram = {'figsize': (6, 1.5), 'dpi': 1200, 'fontsize': None}
    elif data_name == 'SixD_Hyperchaotic3':
        plot_pram = {'figsize': (4, 2), 'dpi': 1200, 'fontsize': 12}
    elif data_name == 'Double_2D_Spring4':
        plot_pram = {'figsize': (4, 2), 'dpi': 1200, 'fontsize': 12}
    elif data_name == 'Cart_Pendulum3':
        plot_pram = {'figsize': (4, 2), 'dpi': 1200, 'fontsize': 12}
    else:
        if args.TASK == 'reconstruct':
            plot_pram = None
        elif args.TASK == 'forecast':
            plot_pram = {'figsize': (6, 1.5), 'dpi': 1200, 'fontsize': None}
        elif args.TASK == 'anomaly_detection':
            plot_pram = {'figsize': (6, 2), 'dpi': 1200, 'fontsize': None}
        else:
            plot_pram = {'figsize': (6, 1.5), 'dpi': 1200, 'fontsize': None}

    return plot_pram



def update_args(Sample_config, args):
    """
    更新args

    Args:
        Sample_config: 这个不是那个ray库里的search space，而是Tune已经采样好的一组，是字典
        args:

    Returns:
        args: 更新后的args

    """

    exp_name = (Sample_config['Version'] + '_' + Sample_config['Method'] + '_' + Sample_config['data_name']
                + '_' + Sample_config['Decompose'] + '_' + Sample_config['TASK'])
    setattr(args, 'exp_name', exp_name)
    if 'data_path' not in Sample_config.keys():
        data_path = Sample_config['TASK'] + '/' + Sample_config['data_name'] + '/' + Sample_config['data_name']
    else:
        data_path = Sample_config['data_path']
    setattr(args, 'data_path', data_path)
    save_path = (args.result_root_path + '/' + Sample_config['Method']
                 + '/' + Sample_config['data_name']
                 + '/' + exp_name)
    setattr(args, 'save_path', save_path)
    setattr(args, 'ckpt_save_path', save_path + '/ckpt')
    setattr(args, 'table_save_path', save_path + '/table')
    setattr(args, 'plot_save_path', save_path + '/plot')

    for key, value in Sample_config.items():
        setattr(args, key, value)

    return args


def args_update_ray(Search_config, args):
    """
    这个主要是在ray_tune_run里面使用的一些args参数，比如patience、TASK、exp_name、ckpt_save_path、grid_num_samples，在Search_space里设置后，有需要更新

    Args:
        Search_config:
        args:

    Returns:

    """

    variables = {'TASK': 'reconstruct', 'data_name': 'MIC_simulate', 'Decompose': 'None', 'Version': 'Vtest',
                 'Method': 'MadjGCN_Project', 'patience': 20, 'grid_num_samples': 1000}
    for key in variables.keys():
        if type(Search_config[key]) == str:
            variables[key] = Search_config[key]
        elif type(Search_config[key]) == int:
            variables[key] = Search_config[key]
        else:
            variables[key] = Search_config[key].categories[0]
    TASK, data_name, Decompose, Version, Method, patience, grid_num_samples = variables['TASK'], \
        variables['data_name'], variables['Decompose'], variables['Version'], variables['Method'], \
        variables['patience'], variables['grid_num_samples']

    exp_name = (Version + '_' + Method + '_' + data_name + '_' + Decompose + '_' + TASK)
    setattr(args, 'exp_name', exp_name)
    save_path = (args.result_root_path + '/' + Method + '/' + data_name + '/' + exp_name)
    setattr(args, 'save_path', save_path)
    setattr(args, 'ckpt_save_path', save_path + '/ckpt')

    setattr(args, 'TASK', TASK)
    setattr(args, 'patience', patience)
    setattr(args, 'grid_num_samples', grid_num_samples)

    return args



def update_args_from_yaml(yaml_path, args):
    """
    更新args, 从yaml文件中读取配置，若yaml文件中有的参数在args中没有，则添加到args中，若yaml文件中有的参数在args中有，则覆盖args中的参数

    Args:
        yaml_path: yaml文件的路径
        args:

    Returns:
        args: 更新后的args

    """

    "# 读取yaml文件"
    import yaml
    with open(yaml_path, 'r') as file:
        Yaml_params = yaml.safe_load(file)
    Common_config = Yaml_params["Common_configs"]
    temporal_block = Common_config['temporal_block']['value']
    spatial_block = Common_config['spatial_block']['value']
    Sample_config = Yaml_params[temporal_block]
    Sample_config.update(Yaml_params[spatial_block])


    "# Sample_config的优先级更高，若是共同的参数，则用Sample_config的参数覆盖Common_config的参数"
    for key, value in Sample_config.items():
        if key in Common_config:
            Common_config[key]["value"] = value["value"]

    "提取出来value"
    for key, value in Common_config.items():
        Common_config[key] = value["value"]
    for key, value in Sample_config.items():
        Sample_config[key] = value["value"]

    "# node_num相关参数需要重新计算并更新"
    # "# node_num相关参数可能在Search_config中没有定义，需要重新计算并更新"
    sensor_num, node_num, timestamp_dim = set_node_num(Common_config['data_name'],
                                                       Common_config['Decompose'],
                                                       Common_config['Wavelet_level'],
                                                       Common_config['if_timestamp'],
                                                       Common_config['if_add_work_condition'],
                                                       Common_config['reco_form'])
    setattr(args, 'sensor_num', sensor_num)
    setattr(args, 'node_num', node_num)
    setattr(args, 'timestamp_dim', timestamp_dim)
    setattr(args, 'Dataset', Common_config['data_name'] + '_Dataset')


    "# 更新exp_name和data_path和save_path"
    exp_name = (Common_config['Version'] + '_' + Common_config['Method'] + '_' + Common_config['data_name']
                + '_' + Common_config['Decompose'] + '_' + Common_config['TASK'])
    setattr(args, 'exp_name', exp_name)
    # 如果Common_config没有'data_path'，那就重新定义，否则就用Common_config里的
    if 'data_path' not in Common_config.keys():
        data_path = Common_config['TASK'] + '/' + Common_config['data_name'] + '/' + Common_config['data_name']
    else:
        data_path = Common_config['data_path']
    setattr(args, 'data_path', data_path)
    save_path = (args.result_root_path + '/' + Common_config['Method']
                 + '/' + Common_config['data_name']
                 + '/' + exp_name)
    setattr(args, 'save_path', save_path)
    setattr(args, 'ckpt_save_path', save_path + '/ckpt')
    setattr(args, 'table_save_path', save_path + '/table')
    setattr(args, 'report_save_path', save_path + '/report')
    setattr(args, 'plot_save_path', save_path + '/plot')

    "# 进行覆盖"
    for key, value in Common_config.items():
        setattr(args, key, value)
    for key, value in Sample_config.items():
        setattr(args, key, value)

    return args



def get_config_sensor_num(config):
    """
    通过传入的config，得到sensor_num

    Args:
        config: 一个字典，Search Space

    Returns:
        sensor_num: int

    """
    sensor_num, node_num, timestamp_dim = set_node_num(config['train_loop_config']['data_name'],
                                                       config['train_loop_config']['Decompose'],
                                                       config['train_loop_config']['Wavelet_level'],
                                                       config['train_loop_config']['if_timestamp'],
                                                       config['train_loop_config']['if_add_work_condition'],
                                                       config['train_loop_config']['reco_form'])
    # sensor_num, node_num, timestamp_dim = set_node_num(config['train_loop_config'].data_name,
    #                                                    config['train_loop_config'].Decompose,
    #                                                    config['train_loop_config'].Wavelet_level,
    #                                                    config['train_loop_config'].if_timestamp,
    #                                                    config['train_loop_config'].if_add_work_condition,
    #                                                    config['train_loop_config'].reco_form)
    return sensor_num


def get_config_node_num(config):
    """
    通过传入的config，得到node_num

    Args:
        config: 一个字典，Search Space

    Returns:
        node_num: int

    """
    sensor_num, node_num, timestamp_dim = set_node_num(config['train_loop_config']['data_name'],
                                                       config['train_loop_config']['Decompose'],
                                                       config['train_loop_config']['Wavelet_level'],
                                                       config['train_loop_config']['if_timestamp'],
                                                       config['train_loop_config']['if_add_work_condition'],
                                                       config['train_loop_config']['reco_form'])
    # sensor_num, node_num, timestamp_dim = set_node_num(config['train_loop_config'].data_name,
    #                                                    config['train_loop_config'].Decompose,
    #                                                    config['train_loop_config'].Wavelet_level,
    #                                                    config['train_loop_config'].if_timestamp,
    #                                                    config['train_loop_config'].if_add_work_condition,
    #                                                    config['train_loop_config'].reco_form)
    return node_num


def get_config_timestamp_dim(config):
    """
    通过传入的config，得到timestamp_dim

    Args:
        config: 一个字典，Search Space

    Returns:
        timestamp_dim: int

    """
    sensor_num, node_num, timestamp_dim = set_node_num(config['train_loop_config']['data_name'],
                                                       config['train_loop_config']['Decompose'],
                                                       config['train_loop_config']['Wavelet_level'],
                                                       config['train_loop_config']['if_timestamp'],
                                                       config['train_loop_config']['if_add_work_condition'],
                                                       config['train_loop_config']['reco_form'])
    # sensor_num, node_num, timestamp_dim = set_node_num(config['train_loop_config'].data_name,
    #                                                    config['train_loop_config'].Decompose,
    #                                                    config['train_loop_config'].Wavelet_level,
    #                                                    config['train_loop_config'].if_timestamp,
    #                                                    config['train_loop_config'].if_add_work_condition,
    #                                                    config['train_loop_config'].reco_form)
    return timestamp_dim
