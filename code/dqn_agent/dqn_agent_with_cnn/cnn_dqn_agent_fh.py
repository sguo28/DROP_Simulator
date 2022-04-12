import os
import random
from collections import defaultdict
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.hex_setting import LEARNING_RATE, GAMMA, REPLAY_BUFFER_SIZE, BATCH_SIZE, RELOCATION_DIM, CHARGING_DIM, \
    INPUT_DIM, DQN_OUTPUT_DIM, FINAL_EPSILON, HIGH_SOC_THRESHOLD, LOW_SOC_THRESHOLD, CLIPPING_VALUE, START_EPSILON, \
    EPSILON_DECAY_STEPS, CNN_SAVE_PATH, SAVING_CYCLE, CNN_RESUME, STORE_TRANSITION_CYCLE, H_AGENT_SAVE_PATH, TERMINAL_STATE_SAVE_PATH
from dqn_agent.dqn_feature_constructor import FeatureConstructor
from dqn_agent.replay_buffer import ReplayMemory
from .cnn_dqn_network import DQN_network, DQN_target_network
from torch.optim.lr_scheduler import StepLR
from dqn_option_agent.option_network import OptionNetwork
import time

class DeepQNetworkAgent:
    def __init__(self,hex_diffusion, h_network, option_num=0, isoption=False,islocal=True,ischarging=True):
        self.learning_rate = LEARNING_RATE
        self.gamma = GAMMA
        self.start_epsilon = START_EPSILON
        self.final_epsilon = FINAL_EPSILON
        self.epsilon_steps = EPSILON_DECAY_STEPS
        self.f_memory = ReplayMemory(REPLAY_BUFFER_SIZE) #used for generaitng data for f_functions, this will store option pairs
        self.memory=ReplayMemory(REPLAY_BUFFER_SIZE) #this is for small replay buffer, this will store primitive action pairs
        self.batch_size = BATCH_SIZE
        self.clipping_value = CLIPPING_VALUE
        self.input_dim = INPUT_DIM
        self.relocation_dim = RELOCATION_DIM
        self.charging_dim = CHARGING_DIM
        self.output_dim = DQN_OUTPUT_DIM
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.path = CNN_SAVE_PATH
        self.state_feature_constructor = FeatureConstructor()
        self.train_step = 0
        self.decayed_epsilon = self.start_epsilon
        self.record_list = []
        self.global_state_dict = OrderedDict()
        self.time_interval = int(0)
        self.global_state_capacity = 5*1440 # we store 5 days' global states to fit replay buffer size.
        self.with_option = isoption
        self.with_charging = ischarging
        self.local_matching = islocal
        self.hex_diffusion = hex_diffusion
        self.num_option=option_num
        self.option_dim=option_num
        self.output_dim = self.num_option + self.relocation_dim + self.charging_dim
        self.h_network_list = h_network

        if option_num>0:
            self.middle_terminal = self.init_terminal_states()
            self.load_option_networks(option_num)

    def load_option_networks(self,option_num):
        self.h_network_list=[]
        for option_net_id in range(option_num):
            h_network = OptionNetwork(self.input_dim,1+6+5)
            checkpoint = torch.load('saved_h/ht_network_option_1000_%d.pkl'%(option_net_id))  # lets try the saved networks after the 14th day.
            h_network.load_state_dict(checkpoint['net'])  # , False
            self.h_network_list.append(h_network.to(self.device))
            print('Successfully load H network {}, total option network num is {}'.format(option_net_id,len(self.h_network_list)))

    def init_terminal_states(self):
        """
        we initial a dict to check the sets of terminal hex ids by hour by option id
        :param oid: ID for option network
        :return:
        """
        middle_terminal = dict()
        for oid in range(self.num_option):
            with open('saved_f/term_states_%d.csv' % oid, 'r') as ts:
                next(ts)
                for lines in ts:
                    line = lines.strip().split(',')
                    hr, hid = line  # option_network_id, hour, hex_ids in terminal state
                    if (oid, int(hr)) in middle_terminal.keys():
                        middle_terminal[(oid, int(hr))].append(int(hid))
                    else:
                        middle_terminal[(oid, int(hr))] = [int(hid)]

        return middle_terminal

    def is_terminal(self,states,oid):
        """
        :param states:
        :return: a list of bool
        """

        return [True if state[1] in self.middle_terminal[(oid,int(state[0] // (60 * 60) % 24))] else False for state in states]




    def get_action_mask(self, batch_state, batch_valid_action):
        """
        the action space: the first 3 is for h_network slots, then 7 relocation actions,and 5 nearest charging stations.
        :param batch_state: state
        :param batch_valid_action: info that limites to relocate to reachable neighboring hexes
        :return:
        """
        mask = np.zeros((len(batch_state), self.output_dim),dtype=bool)  # (num_state, 15)
        for i, state in enumerate(batch_state):
            mask[i][(self.option_dim+ batch_valid_action[i]):(self.option_dim+self.relocation_dim)] = 1  # limited to relocate to reachable neighboring hexes
            if state[-1] > HIGH_SOC_THRESHOLD:
                mask[i][(self.option_dim+self.relocation_dim):] = 1  # no charging, must relocate
            elif state[-1] < LOW_SOC_THRESHOLD:
                mask[i][:(self.option_dim+self.relocation_dim)] = 1  # no relocation, must charge
        return mask

    def get_option_mask(self,states):
        """
        self.is_terminal is to judge if the state is terminal state with the info of hour and hex_id
        :param states:
        :return:
        """
        terminate_option_mask = np.zeros((len(states),self.output_dim),dtype=bool)
        t1=time.time()
        for option in range(self.num_option):
            terminate_option_mask[:,option] = self.is_terminal(states,option)  # set as 0 if not in terminal set
        # for oid in range(self.num_option,self.option_dim):
        #     terminate_option_mask[:,oid] = 1 # mask out empty options
        return terminate_option_mask



    def get_actions(self, states, num_valid_relos, global_state,assigned_option_ids):
        """
        :param global_states:
        :param states: tuple of (tick, hex_id, SOC) and SOC is 0 - 100%
        :param num_valid_relos: only relocation to ADJACENT hexes / charging station is valid
        :states:
        :return:
        """
        with torch.no_grad():
            self.decayed_epsilon = max(self.final_epsilon, (self.start_epsilon - self.train_step * (
                    self.start_epsilon - self.final_epsilon) / self.epsilon_steps))
            state_reps =np.array([self.state_feature_constructor.construct_state_features(state) for state in states])
            hex_diffusions = np.array([np.tile(self.hex_diffusion[state[1]], (1, 1, 1)) for state in
                              states] )# state[1] is hex_id
            mask = self.get_action_mask(states, num_valid_relos)  # mask for unreachable primitive actions
            option_mask = self.get_option_mask(states) # if the state is considered as terminal, we dont use it..

            #We only use random actions
            if True:
                full_action_values = np.random.random(
                    (len(states), self.output_dim))  # generate a matrix with values from 0 to 1
                assigned_option_ids=np.array(assigned_option_ids,dtype=int)
                update_idx=assigned_option_ids>-1
                full_action_values[np.arange(full_action_values.shape[0])[update_idx],assigned_option_ids[update_idx]]=9e10 #must choose
                full_action_values[option_mask] = -9e10 #if the chosen policy is in a terminal state, mask it out
                full_action_values[mask] = -9e10 #choose other actions
                # full_action_values[:,self.num_option]=-9e10 #disable stay aciton for learning f and h
                action_indexes = np.argmax(full_action_values, 1)

        selected_actions=list(action_indexes)  #this is the set of actions selected by DQN, which will be used for training
        converted_action_indexes, new_assigned_opts = self.convert_option_to_primitive_action_id(np.array(action_indexes), state_reps,
                                                                                             global_state,
                                                                                             hex_diffusions, mask)
        action_to_execute=converted_action_indexes-self.option_dim
        #start with false
        contd_options = np.zeros(len(action_indexes), dtype=bool)
        for idx,opts in enumerate(zip(new_assigned_opts,assigned_option_ids)):
            new_opt,old_opt=opts
            if new_opt==old_opt:
                contd_options[idx]=True
        #returning the identified actions, assigned options, and if it is a continuing option of a new option
        return selected_actions,action_to_execute,new_assigned_opts, contd_options


    def convert_option_to_primitive_action_id(self, action_indexes, state_reps, global_state, hex_diffusions, mask):
        """
        we convert the option ids, e.g., 0,1,2 for each h network, to the generated primitive action ids
        :param action_indexes:
        :param state_reps:
        :param global_state:
        :param hex_diffusions:
        :param mask:
        :return:
        """
        with torch.no_grad():
            assigned_options = -np.ones(len(action_indexes))
            ids_require_option = defaultdict(list)
            for id, action_id in enumerate(action_indexes):
                if action_id < self.option_dim:
                    ids_require_option[action_id].append(id)

            converted_states=torch.from_numpy(np.array(state_reps)).to(dtype=torch.float32,device=self.device)
            global_states=torch.from_numpy(np.concatenate([np.tile(global_state,(len(state_reps),1,1,1)),np.array(hex_diffusions)],axis=1)).to(dtype=torch.float32, device=self.device)
            all_mask=torch.from_numpy(mask).to(dtype=torch.bool,device=self.device)

            option_generated=[]
            for option_id in range(self.option_dim):
                if ids_require_option[option_id]:
                    full_option_values = self.h_network_list[option_id].forward(converted_states,global_states)
                    full_option_values=full_option_values[ids_require_option[option_id]]
                    # here mask is of batch x 15 dimension, we omit the first 3 columns, which should be options.
                    primitive_action_mask = all_mask[ids_require_option[option_id],
                                            self.option_dim:]  # only primitive actions in option generator
                    full_option_values[primitive_action_mask] = -9e10
                    full_option_values[:,0]=-9e10 #no self relocation
                    # option_generated.append(torch.argmax(full_option_values, dim=1))
                    #lets try a softmax implementation
                    log_softmax=torch.softmax(full_option_values,dim=1)
                    actions=torch.flatten(torch.multinomial(log_softmax,1)) #choose one action
                    option_generated.append(actions)
                else:
                    option_generated.append(None)

            for option_id in range(self.option_dim):
                if ids_require_option[option_id]:
                    option_generated_premitive_action_ids = option_generated[option_id].cpu().numpy()  # let option network select primitive action
                    action_indexes[ids_require_option[option_id]] = option_generated_premitive_action_ids + self.option_dim  # 12 to 15
                    assigned_options[ids_require_option[option_id]] = option_id
                    # cover the option id with the generated primitive action id
            # for a,o in zip(action_indexes,assigned_options):
            #     if o>-1 and a==self.option_dim:
            #         print('wrong results in cnn_dqn_Agent line 258',o,a)
            #     if a<self.option_dim:
            #         print('wrong results in cnndqnagent line 258',o,a)
        return action_indexes, assigned_options

    def add_global_state_dict(self, global_state_list):
        for tick in global_state_list.keys():
            if tick not in self.global_state_dict.keys():
                self.global_state_dict[tick] = global_state_list[tick]
        if len(self.global_state_dict.keys()) > self.global_state_capacity: #capacity limit for global states
            for _ in range(len(self.global_state_dict.keys())-self.global_state_capacity):
                self.global_state_dict.popitem(last=False)


    def add_transition(self, state, action, next_state, reward, terminate_flag, time_steps, valid_action):
        self.memory.push(state, action, next_state, reward, terminate_flag, time_steps, valid_action)

    def batch_sample(self):
        samples = self.memory.sample(self.batch_size)  # random.sample(self.memory, self.batch_size)
        return samples
        # state, action, next_state, reward = zip(*samples)
        # return state, action, next_state, reward

    # def get_action_mask(self, batch_state, batch_valid_action):
    #     mask = np.zeros([len(batch_state), self.output_dim])
    #     for i, state in enumerate(batch_state):
    #         mask[i][batch_valid_action[i]:self.relocation_dim] = 1
    #         # here the SOC in state is still continuous. the categorized one is in state reps.
    #         if state[-1] > HIGH_SOC_THRESHOLD:
    #             mask[i][self.relocation_dim:] = 1  # no charging, must relocate
    #         elif state[-1] < LOW_SOC_THRESHOLD:
    #             mask[i][:self.relocation_dim] = 1  # no relocation, must charge
    #
    #     mask = torch.from_numpy(mask).to(dtype=torch.bool, device=self.device)
    #     return mask
