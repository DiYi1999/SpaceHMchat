import torch
import torch.nn as nn

class SPS_Model_Phy_wo_BCR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, input_data):
        pass
