import csv

def save_single_element_params_to_csv(model):
    single_element_params_dict = {}

    def extract_params(module, prefix=''):
        for name, param in module.named_parameters(recurse=False):
            if param.numel() == 1:
                single_element_params_dict[prefix + name] = param.item()
        for child_name, child_module in module.named_children():
            extract_params(child_module, prefix + child_name + '.')

    extract_params(model)

    # with open(file_path, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['Parameter Name', 'Value'])
    #     for name, value in single_element_params.items():
    #         writer.writerow([name, value])

    return single_element_params_dict


