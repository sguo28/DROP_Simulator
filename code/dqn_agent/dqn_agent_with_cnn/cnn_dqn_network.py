import torch
import torch.nn as nn
import torch.nn.functional as F
import hexagdly
from config.hex_setting import NUM_REACHABLE_HEX, LEVEL_OF_SOC
class DQN_network(nn.Module):
    '''
    Full connected layer with Relu
    input state(dim): time feature(4), SOC(10), Hex_id(1)
    process: one-hot encoding hex_id, then concat with the vector of 1 x (10+4)
    '''
    def __init__(self,input_dim,output_dim):
        super(DQN_network,self).__init__()
        ## global state

        self.hexconv_1 = hexagdly.Conv2d(in_channels=4, out_channels=16, kernel_size=5, stride=3)
        # self.hexpool = hexagdly.MaxPool2d(kernel_size=1, stride=2)
        self.hexconv_2 = hexagdly.Conv2d(16, 64, 3, 3)  # 1, 16, 15, 18
        self.global_fc = nn.Linear(64 * 6 * 6, 256)  # ,nn.Dropout(0.5)
        # nn.init.xavier_normal_(self.global_fc.weight)
        ## local state
        self.local_fc = nn.Linear(input_dim, 256)
        # nn.init.xavier_normal_(self.local_fc.weight)
        self.fc_adv = nn.Linear(256, 64)
        # nn.init.xavier_normal_(self.fc_adv.weight)
        self.fc_v = nn.Linear(256, 64)
        # nn.init.xavier_normal_(self.fc_v.weight)
        self.output_adv = nn.Linear(256, output_dim)
        # nn.init.xavier_normal_(self.output_adv.weight)
        self.output_v = nn.Linear(64, 1)
        # nn.init.xavier_normal_(self.output_v.weight)

        ## concat_fc
        self.cat_fc = nn.Linear(64*6*6+1347+1,256)
        self.dense1=nn.Linear(54*46*3,512)
        self.new_cat_fc=nn.Linear(512+1347+1,256)
        self.new_cat_fc2=nn.Linear(512,256)
        # nn.init.xavier_normal_(self.cat_fc.weight)

    def forward(self,local_state, global_state):
        # ## global state
        # conv1_out = F.relu(self.hexconv_1(global_state))  # 1, 16, 15, 18
        # conv2_out = F.relu(self.hexconv_2(conv1_out))  # 1, 64, 5, 6
        # flattened = torch.flatten(conv2_out, start_dim=1) # 1, 64*5*6
        # # global_fc_out = self.global_fc(flattened)
        #
        #
        # ## local state
        # # one_hot_hex = F.one_hot(local_state[:,0].to(dtype=torch.int64), num_classes = NUM_REACHABLE_HEX)
        # # one_hot_soc = F.one_hot(local_state[:,1].to(dtype=torch.int64), num_classes = LEVEL_OF_SOC)
        # # local_fc_out = F.relu(self.local_fc()) # we take [1,2,3] which are SOC and time repreentation
        #
        # concat_fc = torch.cat((flattened,local_state[:,1:]),dim=1)
        # fc_out = F.relu(self.cat_fc(concat_fc))
        global_state = global_state[:, [0,1,3], :, :]
        g_feature=torch.flatten(global_state,start_dim=1)
        d1=F.relu(self.dense1(g_feature))
        new_fc=torch.cat((d1,local_state[:,1:]),dim=1)
        fc_out=F.relu(self.new_cat_fc(new_fc))
        q_values=self.output_adv(fc_out)

        return q_values

class DQN_target_network(nn.Module):
    '''
    Full connected layer with Relu
    input state(dim): time feature(4), SOC(10), Hex_id(1)
    process: one-hot encoding hex_id, then concat with the vector of 1 x (10+4)
    '''
    def __init__(self,input_dim,output_dim):
        super(DQN_target_network,self).__init__()
        ## global state

        self.hexconv_1 = hexagdly.Conv2d(in_channels=4, out_channels=16, kernel_size=5, stride=3)
        # self.hexpool = hexagdly.MaxPool2d(kernel_size=1, stride=2)
        self.hexconv_2 = hexagdly.Conv2d(16, 64, 3, 3)  # 1, 16, 15, 18
        self.global_fc = nn.Linear(64 * 6 * 6, 256)  # ,nn.Dropout(0.5)
        # nn.init.xavier_normal_(self.global_fc.weight)
        ## local state
        self.local_fc = nn.Linear(input_dim, 256)
        # nn.init.xavier_normal_(self.local_fc.weight)
        self.fc_adv = nn.Linear(256, 64)
        # nn.init.xavier_normal_(self.fc_adv.weight)
        self.fc_v = nn.Linear(256, 64)
        # nn.init.xavier_normal_(self.fc_v.weight)
        self.output_adv = nn.Linear(256, output_dim)
        # nn.init.xavier_normal_(self.output_adv.weight)
        self.output_v = nn.Linear(64, 1)
        # nn.init.xavier_normal_(self.output_v.weight)

        ## concat_fc
        self.cat_fc = nn.Linear(64*6*6+1347+1,256)
        self.dense1=nn.Linear(54*46*3,512)
        self.new_cat_fc=nn.Linear(512+1347+1,256)
        self.new_cat_fc2=nn.Linear(512,256)
        # nn.init.xavier_normal_(self.cat_fc.weight)


    def forward(self, local_state, global_state):
        # ## global state
        # conv1_out = F.relu(self.hexconv_1(global_state))  # 1, 16, 15, 18
        # conv2_out = F.relu(self.hexconv_2(conv1_out))  # 1, 64, 5, 6
        # flattened = torch.flatten(conv2_out, start_dim=1)  # 1, 64*5*6
        # # global_fc_out = self.global_fc(flattened)
        #
        # ## local state
        # # one_hot_hex = F.one_hot(local_state[:,0].to(dtype=torch.int64), num_classes = NUM_REACHABLE_HEX)
        # # one_hot_soc = F.one_hot(local_state[:,1].to(dtype=torch.int64), num_classes = LEVEL_OF_SOC)
        # # local_fc_out = F.relu(self.local_fc()) # we take [1,2,3] which are SOC and time repreentation
        #
        # concat_fc = torch.cat((flattened, local_state[:, 1:]), dim=1)
        # fc_out = F.relu(self.cat_fc(concat_fc))
        global_state = global_state[:, [0,1,3], :, :]
        g_feature=torch.flatten(global_state,start_dim=1)
        d1=F.relu(self.dense1(g_feature))
        new_fc=torch.cat((d1,local_state[:,1:]),dim=1)
        fc_out=F.relu(self.new_cat_fc(new_fc))
        q_values=self.output_adv(fc_out)



        # adv_out1 = F.relu(self.fc_adv(fc_out))
        # v_out1 = F.relu(self.fc_v(fc_out))



        #
        # adv_values = self.output_adv(adv_out1)
        # v_value = self.output_v(v_out1)
        #
        # q_values = v_value + adv_values - torch.mean(adv_values, dim=1, keepdim=True)
        return q_values

