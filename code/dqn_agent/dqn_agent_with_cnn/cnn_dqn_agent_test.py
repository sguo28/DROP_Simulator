import os
import random
from collections import defaultdict
from collections import OrderedDict, deque
from scipy.spatial import distance
import numpy as np
import torch
from scipy.special import softmax
import torch_optimizer as optim
import itertools
import torch.nn as nn
import torch.nn.functional as F
from config.hex_setting import LEARNING_RATE, GAMMA, REPLAY_BUFFER_SIZE, BATCH_SIZE, RELOCATION_DIM, CHARGING_DIM, \
    INPUT_DIM, DQN_OUTPUT_DIM, FINAL_EPSILON, HIGH_SOC_THRESHOLD, LOW_SOC_THRESHOLD, CLIPPING_VALUE, START_EPSILON, \
    EPSILON_DECAY_STEPS, CNN_SAVE_PATH, SAVING_CYCLE, CNN_RESUME, STORE_TRANSITION_CYCLE, H_AGENT_SAVE_PATH, TERMINAL_STATE_SAVE_PATH,CUDA,NUM_REACHABLE_HEX, MAX_OPTION,\
    MAP_WIDTH,MAP_HEIGHT
from dqn_agent.dqn_feature_constructor import FeatureConstructor
from dqn_agent.replay_buffer import ReplayMemory,Prime_ReplayMemory,F_ReplayMemory,Trajectory_ReplayMemory
from .cnn_dqn_network import DQN_network, DQN_target_network
from torch.optim.lr_scheduler import StepLR
from dqn_option_agent.option_network import OptionNetwork,TargetOptionNetwork
from dqn_option_agent.f_approx_network import F_Network_all

import time

class DeepQNetworkAgent:
    def __init__(self,hex_diffusion, option_num=0, isoption=False,islocal=True,ischarging=True,writer=None):
        self.learning_rate = LEARNING_RATE
        self.gamma = GAMMA
        self.start_epsilon = START_EPSILON
        self.final_epsilon = FINAL_EPSILON
        self.epsilon_steps = EPSILON_DECAY_STEPS
        # self.beta=(self.final_epsilon/self.start_epsilon)**(1/self.epsilon_steps) #epsilon

        self.beta=(self.start_epsilon-self.final_epsilon)/self.epsilon_steps
        self.decayed_epsilon=START_EPSILON
        self.memory = ReplayMemory(REPLAY_BUFFER_SIZE)
        self.batch_size = BATCH_SIZE
        self.clipping_value = CLIPPING_VALUE
        self.input_dim = INPUT_DIM
        self.relocation_dim = RELOCATION_DIM
        self.charging_dim = CHARGING_DIM
        self.output_dim = DQN_OUTPUT_DIM
        self.n_options=0 #number of options generated
        # self.terminal_threshold=0.3

        self.log_softmax=torch.nn.LogSoftmax(dim=1)
        self.device =torch.device("cuda:{}".format(CUDA) if torch.cuda.is_available() else "cpu")
        self.path = CNN_SAVE_PATH

        self.state_feature_constructor = FeatureConstructor()
        self.q_network = DQN_network(self.input_dim, self.output_dim)
        self.target_q_network = DQN_target_network(self.input_dim, self.output_dim)

        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        #     self.q_network=nn.DataParallel(self.q_network)
        #     self.target_q_network=nn.DataParallel(self.target_q_network)


        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate,eps=1e-4)
        self.lr_scheduler = StepLR(optimizer=self.optimizer,step_size=1000, gamma=0.99) # 1.79 e-6 at 0.5 million step.
        self.train_step = 0
        self.h_train_step=0
        self.f_train_step=0
        self.n_f_nets=0  #number of f networks generated so far
        self.fo_train_step=0
        self.q_network.to(self.device)
        self.target_q_network.to(self.device)
        self.decayed_epsilon = self.start_epsilon
        self.record_list = []
        self.global_state_dict = OrderedDict()
        self.time_interval = int(0)
        self.global_state_capacity = 50*1440 # we store 5 days' global states to fit replay buffer size.
        self.with_option = isoption
        self.with_charging = ischarging
        self.local_matching = islocal
        self.hex_diffusion = hex_diffusion
        self.option_dim=option_num
        self.writer=writer
        if option_num>0:
            self.option_dim=option_num
            self.h_network_list = []
            self.h_target_network_list=[]

        self.h_network_list=[]
        self.h_target_network_list=[]
        self.h_optimizer=[]


    def load_option_networks(self,option_num):
        self.h_network_list=[]
        self.h_target_network_list=[]
        for option_net_id in range(MAX_OPTION):
            h_network = OptionNetwork(self.input_dim,1+6)
            h_target_network=TargetOptionNetwork(self.input_dim, 1+6)
            # checkpoint = torch.load('saved_h/ht_network_option_1000_%d.pkl'%(option_net_id))  # lets try the saved networks after the 14th day.
            # h_network.load_state_dict(checkpoint['net'])  # , False
            # h_target_network.load_state_dict(checkpoint['net'])
            self.h_network_list.append(h_network.to(self.device))
            self.h_target_network_list.append(h_target_network.to(self.device))
            # self.h_optimizer = [torch.optim.Adam(self.h_network_list[i].parameters(), lr=1e-3) for i in range(len(self.h_network_list))]



    def get_local_f_by_f(self,global_state,fid,local_state):
        with torch.no_grad():
            #use state[0] for hex id if after feature construction, otherwise use state[1]
            f_vals=self.f_network[fid].forward(global_state,local_state).cpu().numpy()
            return f_vals


    def is_terminal(self,local_state,tick):
        with torch.no_grad():
            # terminal_flag=[1 if veh.option_cruise>2 else 0 for veh in nidle]
            v_idx=[];terminal=[];opt_idx=[]
            # h=int(tick//3600%24)
            # for i in range(len(local_state)):
            #     pos=local_state[i][1]
                # if self.h_train_step>0:
                #     distance=self.current_dist[pos]
                #     pd= np.mean(self.dist_samples[h] < distance)  #move away
                #     print('my distance {} and percentage:{}'.format(distance,pd))
                #     termina_far=pd>0.75 #move away from 0
                #     terminal_near=pd<0.25 #move close to 0
                #     terms=[termina_far,terminal_near]
                #     opts=[j+self.relocation_dim for j in range(1,self.option_dim) if terms[j-1]]
                #     vs=[i for j in opts]
                #     terminal.append(opts)
                #     v_idx+=vs; opt_idx+=opts


            return terminal,v_idx,opt_idx

    def is_local_terminal(self,local_state,global_state):
        # state_reps = np.array([self.state_feature_constructor.construct_f_features(state) for state in local_state])
        # state_reps=self.state_feature_constructor.construct_f_features_batch(local_state)
        # hex_diffusions = np.array([np.tile(self.hex_diffusion[state[1]], (1, 1, 1)) for state in
        #                            local_state])  # state[1] is hex_id
        # terminal,fvals=self.is_terminal(torch.from_numpy(np.array(state_reps)).to(dtype=torch.float32, device=self.device),
        #             torch.from_numpy(np.concatenate([np.tile(global_state,(len(state_reps),1,1,1)),np.array(hex_diffusions)],axis=1)).to(dtype=torch.float32, device=self.device),local_state) # if the state is considered as terminal, we dont use it..
        terminal,fvals=self.is_terminal(local_state,local_state,local_state)
        # print('N f values ={}, mean ={}, 75 percentile={}, 25 percentile={}'.format(fvals.shape[0],np.mean(fvals),np.percentile(fvals,75),np.percentile(fvals,25)))

        return terminal



    def get_action_mask(self, batch_state, batch_valid_action):
        """
        the action space: the first 3 is for h_network slots, then 7 relocation actions,and 5 nearest charging stations.
        :param batch_state: state
        :param batch_valid_action: info that limites to relocate to reachable neighboring hexes
        :return:
        """
        mask = np.zeros((len(batch_state), self.output_dim),dtype=bool)  # (num_state, 15)
        mask[:,self.relocation_dim+self.n_f_nets*self.option_dim:]=1 #not allowed use those f nets when they are not availabel yet.
        return mask


    def get_actions(self, states, num_valid_relos, global_state,assigned_option_ids, v_idx, opt_idx,tick):
        """
        :param global_states:
        :param states: tuple of (tick, hex_id, SOC) and SOC is 0 - 100%
        :param num_valid_relos: only relocation to ADJACENT hexes / charging station is valid
        :states:
        :return:
        """
        with torch.no_grad():
            self.decayed_epsilon=self.final_epsilon
            # state_reps = np.array([self.state_feature_constructor.construct_state_features(state) for state in states])
            state_reps = self.state_feature_constructor.construct_f_features_batch(states)
            hex_diffusions = np.array([np.tile(self.hex_diffusion[state[1]], (1, 1, 1)) for state in
                              states])  # state[1] is hex_id
            mask = self.get_action_mask(states, num_valid_relos)  # mask for unreachable primitive actions
            take_rands=np.random.binomial(1,self.decayed_epsilon,len(states))

            action_indexes=np.zeros(len(states)).astype(np.int32)

            print(self.n_f_nets,self.option_dim,self.decayed_epsilon)
            status_code = [0 for _ in range(len(states))]
            if 1:  # epsilon = 0.1
                if self.decayed_epsilon<1:
                    full_action_values = self.q_network.forward(
                        torch.from_numpy(np.array(state_reps)).to(dtype=torch.float32, device=self.device),
                        torch.from_numpy(np.concatenate([np.tile(global_state,(len(states),1,1,1)),np.array(hex_diffusions)],axis=1)).to(dtype=torch.float32, device=self.device))
                    assigned_option_ids=np.array(assigned_option_ids,dtype=int)

                    update_idx=assigned_option_ids>-1 #those states who are under options
                    full_action_values[np.arange(full_action_values.shape[0])[update_idx], assigned_option_ids[update_idx]] = 9e10 #must take option

                    #     terminate_option_mask] = -9e10  # if the chosen policy is in a terminal state, mask it out
                    full_action_values[torch.from_numpy(mask).to(dtype=torch.bool,device=self.device)] = -9e10

                    q_action_indexes = torch.argmax(full_action_values, dim=1).cpu().numpy()
                    assigned_option_ids=np.array(assigned_option_ids,dtype=int)
                    update_idx=assigned_option_ids>-1

                    action_indexes[take_rands == 0] = q_action_indexes[take_rands == 0]
                #exploration based on f values, choose the large ones.
                if 1:
                    full_action_values = -9e10*np.ones(
                        (len(states), self.output_dim))  # generate a matrix with values from 0 to 1
                    full_action_values[:, :self.relocation_dim+self.n_f_nets*self.option_dim]=np.random.random((len(states), self.relocation_dim+self.n_f_nets*self.option_dim))  # generate a matrix with values from 0 to 1
                    # if self.option_dim>0:
                    #     full_action_values[:,0]=-9e10
                    if self.option_dim>0:
                        full_action_values[np.arange(full_action_values.shape[0])[update_idx], assigned_option_ids[update_idx]] = 9e10  # must choose


                    full_action_values[mask] = -9e10  # choose other actions
                    full_action_values[v_idx, opt_idx] = -9e10
                    # if random.random()>self.decayed_epsilon:
                    #     full_action_values[:,self.relocation_dim+1:]=-9e10

                    r_action_indexes = np.argmax(full_action_values, 1)
                    # log_softmax = torch.softmax(full_action_values, dim=1)
                    # action_indexes = torch.flatten(torch.multinomial(log_softmax, 1)).cpu().numpy() # choose one action
            action_indexes[take_rands==1]=r_action_indexes[take_rands==1]


        selected_actions=list(action_indexes)  #this is the set of actions selected by DQN, which will be used for training
        converted_action_indexes, new_assigned_opts = self.convert_option_to_primitive_action_id(np.array(action_indexes), state_reps,
                                                                                             global_state,
                                                                                             hex_diffusions, mask)

        action_to_execute=converted_action_indexes
        contd_options = np.zeros(len(action_indexes), dtype=bool)
        for idx,opts in enumerate(zip(new_assigned_opts,assigned_option_ids)):
            new_opt,old_opt=opts
            if new_opt==old_opt and new_opt>=0:
                contd_options[idx]=True
        #returning the identified actions, assigned options, and if it is a continuing option of a new option
        return selected_actions,action_to_execute,new_assigned_opts, contd_options,status_code

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
            # reached_terminal=np.ones(len(action_indexes)) #check if an action reaches the destination
            ids_require_option = defaultdict(list)
            for id, action_id in enumerate(action_indexes):
                if action_id >self.relocation_dim-1: #using options
                    ids_require_option[action_id].append(id)

            converted_states=torch.from_numpy(np.array(state_reps)).to(dtype=torch.float32,device=self.device)
            global_states=torch.from_numpy(np.concatenate([np.tile(global_state,(len(state_reps),1,1,1)),np.array(hex_diffusions)],axis=1)).to(dtype=torch.float32, device=self.device)


            option_generated=[]
            for option_id in range(self.n_f_nets*self.option_dim):
                if ids_require_option[option_id+self.relocation_dim]:
                    full_option_values = self.h_network_list[option_id].forward(converted_states,global_states)
                    full_option_values=full_option_values[ids_require_option[option_id+self.relocation_dim]]
                    #all negative values for H at the location
                    acts=torch.argmax(full_option_values, dim=1)
                    option_generated.append(acts.cpu().numpy())
                    #lets try a softmax implementation
                    # log_softmax=torch.softmax(full_option_values,dim=1)
                    # actions=torch.flatten(torch.multinomial(log_softmax,1)) #choose one action
                    # option_generated.append(actions)
                else:
                    option_generated.append(None)

            for option_id in range(self.n_f_nets*self.option_dim):
                if ids_require_option[option_id+self.relocation_dim]:
                    option_generated_premitive_action_ids = option_generated[option_id]  # let option network select primitive action
                    action_indexes[ids_require_option[option_id+self.relocation_dim]] = option_generated_premitive_action_ids # 12 to 15
                    assigned_options[ids_require_option[option_id+self.relocation_dim]] = option_id+self.relocation_dim
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

    def add_trajectories(self, t):
        self.trajectory_memory.push(t)

    def batch_sample(self):
        samples = self.memory.sample(self.batch_size)  # random.sample(self.memory, self.batch_size)
        return samples
        # state, action, next_state, reward = zip(*samples)
        # return state, action, next_state, reward

    def add_H_transition(self, state, action, next_state, trip_flag, time_steps, valid_action):
        self.h_memory.push(state, action, next_state, trip_flag, time_steps, valid_action)

    def H_batch_sample(self,batch_size):
        samples = self.h_memory.sample(batch_size)  # random.sample(self.memory, self.batch_size)
        return samples

    def f_batch_sample(self,hrs):
        samples = self.f_memory[hrs].sample(self.batch_size)  # random.sample(self.memory, self.batch_size)
        return samples

    def add_f_transition(self,data,hrs):
        #data is of format [state, state_]
        self.f_memory[hrs].push(data[0],data[1],data[2])

    def add_fo_transition(self,data,hrs):
        #data is of format [state, state_]
        self.fo_memory[hrs].push(data[0],data[1])

    def get_f_value(self,local_state,global_state,hrs):
        return self.f_network[-1].forward(global_state,local_state)

    def get_main_Q(self, local_state, global_state):
        return self.q_network.forward(local_state, global_state)

    def save_parameter(self,trial):
        # torch.save(self.q_network.state_dict(), self.dqn_path)
        if self.option_dim>0:
            h_net=[i.state_dict() for i in self.h_network_list[:self.option_dim]]
        else:
            h_net=[]
        if 1:
            checkpoint = {
                "net_dqn": self.q_network.state_dict(),
                "net_f":[],
                "net_h":h_net,
                # 'optimizer': self.optimizer.state_dict(),
                "step": self.train_step,
            }

        if not os.path.isdir(self.path):
                os.mkdir(self.path)
            # print('the path is {}'.format('logs/dqn_model/duel_dqn_%s.pkl'%(str(self.train_step))))
        torch.save(checkpoint, 'logs/test/cnn_dqn_model/dqn_fh_{}_{}_{}_{}.pkl'.format(self.learning_rate,self.option_dim,str(self.train_step),trial))
            # record training process (stacked before)

    def load_parameter(self,model_file):
        checkpoint = torch.load(model_file)

        params=model_file.split('_')
        params[-1]=params[-1][:-4]

        trial=int(params[-1])
        train_step=int(params[-2])
        self.option_dim=int(params[-3])
        learning_rate=params[-4]


        self.q_network.load_state_dict(checkpoint['net_dqn'])
        self.h_network_list=[]
        for _ in range(len(checkpoint['net_h'])):
            h_network = OptionNetwork(self.input_dim, 1 + 6)
            self.h_network_list.append(h_network.to(self.device))
        [h.load_state_dict(i) for h,i in zip(self.h_network_list,checkpoint['net_h'])] #load h


        print('Load model', model_file)
        print('Number of networks:',len(checkpoint['net_h']))
        if self.option_dim>0:
            self.n_f_nets = int(len(self.h_network_list)/self.option_dim)
        else:
            self.n_f_nets=0

        return trial,train_step,self.option_dim,learning_rate


    def save_final_parameter(self,trial):
        # torch.save(self.q_network.state_dict(), self.dqn_path)
        if 1:
            checkpoint = {
                "net_dqn": self.q_network.state_dict(),
                "net_f":[],
                "net_h":[],
                # 'optimizer': self.optimizer.state_dict(),
                "step": self.train_step,
                "lr_scheduler": self.lr_scheduler.state_dict()
            }

        if not os.path.isdir(self.path):
                os.mkdir(self.path)
            # print('the path is {}'.format('logs/dqn_model/duel_dqn_%s.pkl'%(str(self.train_step))))
        torch.save(checkpoint, 'logs/test/cnn_dqn_model/final_models/dqn_fh_{}_{}_{}_{}.pkl'.format(self.learning_rate,self.option_dim,str(self.train_step),trial))
            # record training process (stacked before)
