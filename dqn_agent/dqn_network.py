import torch
import torch.nn as nn
import torch.nn.functional as F
import hexagdly
from config.hex_setting import HEX_INPUT_CHANNELS, HEX_OUTPUT_CHANNELS, HEX_KERNEL_SIZE, HEX_STRIDE, NUM_REACHABLE_HEX, LEVEL_OF_SOC
class DQN_network(nn.Module):
    '''
    Full connected layer with Relu
    input state(dim): time feature(4), SOC(10), Hex_id(1)
    process: one-hot encoding hex_id, then concat with the vector of 1 x (10+4)
    '''
    def __init__(self,input_dim,output_dim):
        super(DQN_network,self).__init__()
        self.fc = nn.Linear(input_dim,256)
        self.fc_adv = nn.Linear(256,64)
        self.fc_v = nn.Linear(256, 64)
        self.output_adv = nn.Linear(64,output_dim)
        self.output_v = nn.Linear(64,1)

    def forward(self,input_state):
        one_hot_hex = F.one_hot(input_state[:,0].to(dtype=torch.int64), num_classes = NUM_REACHABLE_HEX)
        one_hot_soc = F.one_hot(input_state[:,1].to(dtype=torch.int64), num_classes = LEVEL_OF_SOC)
        fc_out = F.relu(self.fc(torch.cat([input_state[:,2:],one_hot_soc, one_hot_hex], dim = 1)))

        adv_out1 = F.relu(self.fc_adv(fc_out))
        v_out1 = F.relu(self.fc_v(fc_out))

        adv_values = self.output_adv(adv_out1)
        v_value = self.output_v(v_out1)

        q_values = v_value + adv_values - torch.mean(adv_values, dim=1, keepdim=True)
        return q_values

class DQN_target_network(nn.Module):
    '''
    Full connected layer with Relu
    input state(dim): time feature(4), SOC(10), Hex_id(1)
    process: one-hot encoding hex_id, then concat with the vector of 1 x (10+4)
    '''
    def __init__(self,input_dim,output_dim):
        super(DQN_target_network,self).__init__()
        self.fc = nn.Linear(input_dim,256)
        self.fc_adv = nn.Linear(256,64)
        self.fc_v = nn.Linear(256, 64)
        self.output_adv = nn.Linear(64,output_dim)
        self.output_v = nn.Linear(64,1)

    def forward(self,input_state):
        one_hot_hex = F.one_hot(input_state[:,0].to(dtype=torch.int64), num_classes = NUM_REACHABLE_HEX)
        one_hot_soc = F.one_hot(input_state[:,1].to(dtype=torch.int64), num_classes = LEVEL_OF_SOC)
        fc_out = F.relu(self.fc(torch.cat([input_state[:,2:],one_hot_soc, one_hot_hex], dim = 1)))

        adv_out1 = F.relu(self.fc_adv(fc_out))
        v_out1 = F.relu(self.fc_v(fc_out))

        adv_values = self.output_adv(adv_out1)
        v_value = self.output_v(v_out1)

        q_values = v_value + adv_values - torch.mean(adv_values, dim=1, keepdim=True)
        return q_values
