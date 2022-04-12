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
    MAP_WIDTH,MAP_HEIGHT, USE_RANDOM, LIMIT_ONE_NET
from dqn_agent.dqn_feature_constructor import FeatureConstructor
from dqn_agent.replay_buffer import ReplayMemory,Prime_ReplayMemory,F_ReplayMemory,Trajectory_ReplayMemory, Step_ReplayMemory, Step_FReplayMemory,Step_PrimeReplayMemory
from .cnn_dqn_network import DQN_network, DQN_target_network
from torch.optim.lr_scheduler import StepLR
from dqn_option_agent.option_network import OptionNetwork,TargetOptionNetwork
from dqn_option_agent.f_approx_network import F_Network_all

import time

def neg_loss(x,c=2,reg=0):
    # z=torch.ones((x.shape[0],2)) #augment each vector with ones as f1
    # z[:,0]=x[:,0]
    z=x
    n = z.shape[0]
    d = z.shape[1]
    inprods=z@z.T
    norms = inprods[torch.arange(n), torch.arange(n)]
    part1 = inprods.pow(2).sum()- norms.pow(2).sum()
    part1 = part1 / ((n - 1) * n)
    part2 = - 2 * c * norms.mean() / d
    part3 = c * c / d
    #
    # part4= sum([(c**2- x[:,i].pow(2).mean(0)).pow(2) for i in range(d)])
    # regularization
    if reg > 0.0:
        reg_part1 = norms.pow(2).mean()
        reg_part2 = - 2 * c * norms.mean()
        reg_part3 = c * c
        reg_part = (reg_part1 + reg_part2 + reg_part3) / n
    else:
        reg_part = 0.0
    return part1 + part2 + part3  + reg * reg_part

def neg_loss_smooth(x1):
    dist=torch.cdist(x1,x1,p=2)
    n=x1.shape[0]
    loss=torch.exp(-dist)
    loss=loss-torch.diag(torch.diag(loss))
    return loss.sum()/(n*(n-1))

def neg_loss_smooth_trajectory(x1):
    dist=torch.cdist(x1,x1,p=2)
    n=x1.shape[0]
    loss=torch.exp(-dist)
    loss=loss-torch.diag(torch.diag(loss))
    return loss.sum(),n*(n-1)

def pos_loss(x1,x2):
    return (x1-x2).pow(2).sum(-1).mean()

def pos_loss_trajectory(x1,x2):
    n=x1.shape[0]
    return (x1-x2).pow(2).sum(),n


def neg_loss_manual(x1,x2,c=.1):
    n=x1.shape[0]
    inprods=x1@x2.T
    norm1=x1.pow(2).sum(-1).mean()
    norm2=x2.pow(2).sum(-1).mean()
    part1=inprods.mean()
    part2=norm1+norm2
    part3=c*c/n

    loss=part1-part2+part3

    return loss



# def neg_loss_dimension(x1):
#     loss=0
#     D=x1.shape[1]
#     for d in range(x1.shape[1]):
#         loss+=neg_loss(x1[:,:d+1])
#     return loss
#
# def pos_loss_dim(x1,x2):
#     X=(x1 - x2).pow(2)
#     loss=0
#     D=x1.shape[1]
#     for d in range(x1.shape[1]):
#         loss+=X[:,:d+1].sum(-1)
#
#     return loss.mean()



class DeepQNetworkAgent:
    def __init__(self,hex_diffusion, option_num=0, isoption=False,islocal=True,ischarging=True,writer=None):
        self.learning_rate = LEARNING_RATE
        self.gamma = GAMMA
        self.start_epsilon = START_EPSILON
        self.final_epsilon = FINAL_EPSILON
        self.epsilon_steps = EPSILON_DECAY_STEPS
        self.alpha=0
        self.beta=(self.final_epsilon/self.start_epsilon)**(1/self.epsilon_steps) #epsilon

        # self.beta=(self.start_epsilon-self.final_epsilon)/self.epsilon_steps
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
        self.option_train_step=[]
        self.load_network()
        self.q_network.to(self.device)
        self.target_q_network.to(self.device)
        self.f_memory=[F_ReplayMemory(int(200000)) for _ in range(24)]
        self.fo_memory = [F_ReplayMemory(int(2e4)) for _ in range(24)]
        self.h_memory=Prime_ReplayMemory(int(200000))
        self.trajectory_memory=Trajectory_ReplayMemory(int(30000))
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
            self.load_option_networks(self.option_dim)

            self.f_network =[] # [F_Network_all(INPUT_DIM,self.option_dim//2) for _ in range(MAX_OPTION)] #24 networks
            self.f_target_network=[] #[F_Network_all(INPUT_DIM,self.option_dim//2) for _ in range(MAX_OPTION)]
            [i.to(self.device) for i in self.f_target_network]
            # if torch.cuda.device_count() > 1:
            #     print("Let's use", torch.cuda.device_count(), "GPUs!")
            #     self.f_network=[nn.DataParallel(i) for i in self.f_network]
            # self.load_f_params()
            [i.to(self.device) for i in self.f_network]
            self.trained=[0 for _ in range(24)]

            self.fo_network=[F_Network_all(INPUT_DIM,1) for _ in range(24)] #24 networks
            [i.to(self.device) for i in self.fo_network]

            self.f_lower=np.array([-.1 for _ in range(24)])
            self.f_upper = np.array([.1 for _ in range(24)])
            self.f_median=np.array([0.0 for _ in range(24)])
            self.f_max=np.array([1 for _ in range(24)])
            self.f_median_episode = np.array([0.0 for _ in range(24)])
            self.f_max_episode = np.array([1 for _ in range(24)])

            self.fo_lower=np.array([-.1 for _ in range(24)])
            self.fo_upper = np.array([.1 for _ in range(24)])
            self.fo_median=np.array([0.0 for _ in range(24)])


            self.f_optimizer = [torch.optim.Adam(i.parameters(), lr=1e-3) for i in self.f_network]
            self.fo_optimizer=[torch.optim.Adam(i.parameters(), lr=1e-3) for i in self.fo_network]
            self.h_optimizer = [torch.optim.Adam(self.h_network_list[i].parameters(), lr=1e-3) for i in range(len(self.h_network_list))]

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


            # print('Successfully load H network {}, total option network num is {}'.format(option_net_id,len(self.h_network_list)))

    # def init_terminal_states(self):
    #     """
    #     we initial a dict to check the sets of terminal hex ids by hour by option id
    #     :param oid: ID for option network
    #     :return:
    #     """
    #     middle_terminal = dict()
    #     for oid in range(self.option_dim):
    #         with open( 'saved_f/term_states_%d.csv' % oid, 'r') as ts:
    #             next(ts)
    #             for lines in ts:
    #                 line = lines.strip().split(',')
    #                 hr, hid = line  # option_network_id, hour, hex_ids in terminal state
    #                 if (oid, int(hr)) in middle_terminal.keys():
    #                     middle_terminal[(oid, int(hr))].append(int(hid))
    #                 else:
    #                     middle_terminal[(oid, int(hr))] = [int(hid)]
    #     return middle_terminal

    def load_f_params(self):
        checkpoint = torch.load('saved_f/f_network_option_1000_%d.pkl' % (0))  # lets try the saved networks after the 14th day.
        self.f_network.load_state_dict(checkpoint['net'])  # , False
        # print('Successfully load f network {}, total option network num is {}'.format(0,1))


    def reset_f_storage(self):
        self.f_median_all=[[] for _ in range(24)]
        self.fo_median_all = [[] for _ in range(24)]
        self.f_all=np.zeros((24*60,NUM_REACHABLE_HEX))
        self.fo_all = np.zeros((24 * 60, NUM_REACHABLE_HEX))
        self.f_sliding=deque(maxlen=NUM_REACHABLE_HEX*15) #track 15 ticks


    def summarize_median(self):
        self.f_median_episode=np.array([np.median(val) for val in self.f_median_all])
        minf=np.array([np.min(val) for val in self.f_median_all])
        maxf=np.array([np.max(val) for val in self.f_median_all])
        self.f_max_episode=np.array([np.maximum(abs(minv-medv),abs(maxv-medv)) for minv,maxv,medv in zip(minf,maxf,self.f_median_episode)])


    def get_local_f_by_f(self,global_state,fid,local_state):
        with torch.no_grad():
            #use state[0] for hex id if after feature construction, otherwise use state[1]
            f_vals=self.f_network[fid].forward(global_state,local_state).cpu().numpy()
            return f_vals

    def get_local_f(self,global_state,tick):
        with torch.no_grad():
            # local_state=[]
            # for i in range(NUM_REACHABLE_HEX):
            #     state=[tick,i,1]
            #     local_state.append(self.state_feature_constructor.construct_state_features(state))

            # local_state=[self.state_feature_constructor.construct_f_features_batch([tick,i,1]) for i in range(NUM_REACHABLE_HEX)]
            local_states=np.ones((NUM_REACHABLE_HEX,3))
            local_states[:,0]=tick; local_states[:,1]=np.arange(NUM_REACHABLE_HEX)
            local_state=self.state_feature_constructor.construct_f_features_batch(local_states)

            #use state[0] for hex id if after feature construction, otherwise use state[1]
            hex_diffusions = np.array([np.tile(self.hex_diffusion[i], (1, 1, 1)) for i in
                                       range(NUM_REACHABLE_HEX)])  # state[1] is hex_id
            local_state=torch.from_numpy(np.array(local_state)).to(dtype=torch.float32, device=self.device)
            g_state=torch.from_numpy(np.concatenate([np.tile(global_state,(len(local_state),1,1,1)),np.array(hex_diffusions)],axis=1)).to(dtype=torch.float32, device=self.device)
            hrs=int(tick//3600%24)
            f_vals=self.get_f_value(local_state,g_state,hrs).cpu().numpy()
            self.current_f=f_vals
            self.current_dist=(f_vals[:,1]**2+f_vals[:,2]**2)**0.5
            # self.f_sliding+=f_vals.tolist() #sliding window to track the f values
            # # fo_vals=self.fo_network.forward(g_state,local_state).cpu().numpy()
            #
            # # print('length of local_state',len(f_vals),'Setting local values before:', self.f_median[hrs], self.f_lower[hrs], self.f_upper[hrs])
            # self.f_median[hrs]=np.median(self.f_sliding)
            # minf=np.min(self.f_sliding); maxf=np.max(self.f_sliding)
            # self.f_max[hrs]=np.maximum(abs(minf-self.f_median[hrs]),abs(maxf-self.f_median[hrs])) #set the maximum of the hours
            # self.current_max_f=self.f_max[hrs]
            #
            # print('The median of fvals is:',np.median(f_vals), ' The maximum f values is:',self.f_max[hrs])
            # self.f_median_all[hrs]+=f_vals.tolist()
            # # self.fo_median_all[hrs] += fo_vals.tolist()
            # self.current_f=np.abs(f_vals)-np.abs(np.median(f_vals))#-self.f_median[hrs]

            # self.f_lower[hrs]=np.percentile(f_vals,20)
            # self.f_upper[hrs]=np.percentile(f_vals,80)
            # print('Setting local values after:', self.f_median[hrs],self.f_lower[hrs],self.f_upper[hrs])

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


    # def is_terminal(self,states,oid):
    #     """
    #     :param states:
    #     :return: a list of bool
    #     """
    #
    #     return [True if state[1] in self.middle_terminal[(oid,int(state[0] // (60 * 60) % 24))] else False for state in states]

    def load_network(self):
        if CNN_RESUME:
            lists = os.listdir(self.path)
            lists.sort(key=lambda fn: os.path.getmtime(self.path + "/" + fn))
            newest_file = os.path.join(self.path, lists[-1])
            path_checkpoint = newest_file  #'logs/test/cnn_dqn_model/duel_dqn_69120.pkl'  #
            checkpoint = torch.load(path_checkpoint)

            self.q_network.load_state_dict(checkpoint['net'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            self.train_step = checkpoint['step']
            self.copy_parameter()
            # self.optimizer.load_state_dict(checkpoint['optimizer'])
            # print('Successfully load saved network starting from {}!'.format(str(self.train_step)))



    def get_action_mask(self, batch_state, batch_valid_action):
        """
        the action space: the first 3 is for h_network slots, then 7 relocation actions,and 5 nearest charging stations.
        :param batch_state: state
        :param batch_valid_action: info that limites to relocate to reachable neighboring hexes
        :return:
        """
        mask = np.zeros((len(batch_state), self.output_dim),dtype=bool)  # (num_state, 15)
        # for i, state in enumerate(batch_state):
        #     mask[i][(self.option_dim+ batch_valid_action[i]):(self.option_dim+self.relocation_dim)] = 1  # limited to relocate to reachable neighboring hexes
        #     if state[-1] > HIGH_SOC_THRESHOLD:
        #         mask[i][(self.option_dim+self.relocation_dim):] = 1  # no charging, must relocate
        #     elif state[-1] < LOW_SOC_THRESHOLD:
        #         mask[i][:(self.option_dim+self.relocation_dim)] = 1  # no relocation, must charge
        mask[:,self.relocation_dim+self.n_f_nets*self.option_dim:]=1 #not allowed use those f nets when they are not availabel yet.
        return mask

    def get_option_mask(self,local_state,global_state,original_state):
        """
        self.is_terminal is to judge if the state is terminal state with the info of hour and hex_id
        :param states:
        :return:
        """
        terminate_option_mask = np.zeros((len(local_state),self.output_dim),dtype=bool)
        for option in range(self.option_dim):
            terminal,_=self.is_terminal(local_state,global_state,original_state)
            terminate_option_mask[:,option] = terminal # set as 0 if not in terminal set
        # for oid in range(self.option_dim,self.option_dim):
        #     terminate_option_mask[:,oid] = 1 # mask out empty options
        return terminate_option_mask

    def get_actions(self, states, num_valid_relos, global_state,assigned_option_ids, v_idx, opt_idx,tick):
        """
        :param global_states:
        :param states: tuple of (tick, hex_id, SOC) and SOC is 0 - 100%
        :param num_valid_relos: only relocation to ADJACENT hexes / charging station is valid
        :states:
        :return:
        """
        with torch.no_grad():
            # self.decayed_epsilon=self.start_epsilon-self.beta*self.train_step
            self.decayed_epsilon = self.start_epsilon*(self.beta ** self.train_step)
            self.decayed_epsilon = max(self.final_epsilon, self.decayed_epsilon)
            # state_reps = np.array([self.state_feature_constructor.construct_state_features(state) for state in states])
            state_reps = self.state_feature_constructor.construct_f_features_batch(states)
            hex_diffusions = np.array([np.tile(self.hex_diffusion[state[1]], (1, 1, 1)) for state in
                              states])  # state[1] is hex_id
            mask = self.get_action_mask(states, num_valid_relos)  # mask for unreachable primitive actions
            # option_mask = self.get_option_mask(torch.from_numpy(np.array(state_reps)).to(dtype=torch.float32, device=self.device),
            #         torch.from_numpy(np.concatenate([np.tile(global_state,(len(states),1,1,1)),np.array(hex_diffusions)],axis=1)).to(dtype=torch.float32, device=self.device),states) # if the state is considered as terminal, we dont use it..

            action_swap=0

            take_rands=np.random.binomial(1,self.decayed_epsilon,len(states))

            action_indexes=np.zeros(len(states)).astype(np.int32)

            if 1:  # epsilon = 0.1
                full_action_values = self.q_network.forward(
                    torch.from_numpy(np.array(state_reps)).to(dtype=torch.float32, device=self.device),
                    torch.from_numpy(np.concatenate([np.tile(global_state,(len(states),1,1,1)),np.array(hex_diffusions)],axis=1)).to(dtype=torch.float32, device=self.device))
                assigned_option_ids=np.array(assigned_option_ids,dtype=int)
                #choose an action, in the following conditions:
                #1. if it is a terminal state, select the one with the maximum Q value
                #2  if it is not a terminal state, following the previous option policy to get the action (must)
                # ---- if there is no previous options, randomly select a policy with the largest value
                update_idx=assigned_option_ids>-1 #those states who are under options
                full_action_values[np.arange(full_action_values.shape[0])[update_idx], assigned_option_ids[update_idx]] = 9e10 #must take option
                # print('take a look at processed mask {}'.format(mask))
                # terminate_option_mask = torch.from_numpy(option_mask).to(dtype=torch.bool, device=self.device)
                # if self.option_dim>0:
                #     swap_list=np.random.binomial(1,1-self.decayed_epsilon,len(update_idx)) #create a list to break
                #     swap_list=update_idx
                #     acts_to_swap=torch.argmax(full_action_values[:,self.option_dim:self.option_dim+7],dim=1).cpu().numpy()
                #     action_swap=1
                #     print('Conducting action swap to break some of the options')

                ### We need to add a scheme so that proportionally the options will only use greedy q values instead of f-values, but still maintaining the option code


                # full_action_values[
                #     terminate_option_mask] = -9e10  # if the chosen policy is in a terminal state, mask it out
                full_action_values[torch.from_numpy(mask).to(dtype=torch.bool,device=self.device)] = -9e10
                # full_action_values[:,self.relocation_dim+1:]=-9e10
                # if self.option_dim>0 and self.f_train_step == 0:
                #     full_action_values[:,:self.option_dim] = -9e10  # if the chosen policy is in a terminal state, mask it out

                full_action_values[v_idx,opt_idx]=-9e10

                #lets do 50% option and 50% premitive action
                # if self.option_dim>0:
                #     full_action_values[:,0]=-9e10

                q_action_indexes = torch.argmax(full_action_values, dim=1).cpu().numpy()
                status_code=[1 for _ in range(len(states))]


                assigned_option_ids=np.array(assigned_option_ids,dtype=int)
                update_idx=assigned_option_ids>-1

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
                    if self.f_train_step==0: full_action_values[:, self.relocation_dim:] = -9e10  # if the chosen policy is in a terminal state, mask it out
                    # if random.random()>self.decayed_epsilon:
                    #     full_action_values[:,self.relocation_dim+1:]=-9e10

                    r_action_indexes = np.argmax(full_action_values, 1)
                    # log_softmax = torch.softmax(full_action_values, dim=1)
                    # action_indexes = torch.flatten(torch.multinomial(log_softmax, 1)).cpu().numpy() # choose one action
            action_indexes[take_rands==1]=r_action_indexes[take_rands==1]
            action_indexes[take_rands==0]=q_action_indexes[take_rands==0]
            status_code = [1 if s==1 else 0 for s in take_rands]
                #lets do 50% option and 50% premitive action
                # if self.option_dim>0:
                #     full_action_values[:,0]=9e10



        selected_actions=list(action_indexes)  #this is the set of actions selected by DQN, which will be used for training
        converted_action_indexes, new_assigned_opts = self.convert_option_to_primitive_action_id(np.array(action_indexes), state_reps,
                                                                                             global_state,
                                                                                             hex_diffusions, mask)
        # converted_action_indexes, new_assigned_opts = self.convert_option_to_primitive_action_id_f(np.array(action_indexes), np.array(states), state_reps,
        #                                                                                      global_state,
        #                                                                                      hex_diffusions, mask,tick)

        action_to_execute=converted_action_indexes
        # if action_swap>0:
        #     action_to_execute=acts_to_swap
        #start with false
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


    def convert_option_to_primitive_action_id_f(self, action_indexes, states,state_reps, global_state, hex_diffusions, mask,tick):
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
            #
            hex_ids=states[:,1].astype(int) #all hex ids

            local_states=np.ones((NUM_REACHABLE_HEX,3))
            local_states[:,0]=tick; local_states[:,1]=np.arange(NUM_REACHABLE_HEX)
            local_state=self.state_feature_constructor.construct_f_features_batch(local_states)
            local_state=torch.from_numpy(local_state).to(dtype=torch.float32,device=self.device)

            # converted_states=torch.from_numpy(np.array(state_reps)).to(dtype=torch.float32,device=self.device)
            hex_diffusions = np.array([np.tile(self.hex_diffusion[i], (1, 1, 1)) for i in
                                       range(NUM_REACHABLE_HEX)])  # state[1] is hex_id
            global_states=torch.from_numpy(np.concatenate([np.tile(global_state,(len(local_state),1,1,1)),np.array(hex_diffusions)],axis=1)).to(dtype=torch.float32, device=self.device)

            option_generated=[]
            dims=[[i-1,i] for i in range(self.option_dim//2)]+[[i-1,i] for i in range(self.option_dim//2)]
            signs=[1 for _ in range(self.option_dim//2)]+[-1 for _ in range(self.option_dim//2)]


            saved_f=[[] for _ in range(self.n_f_nets)]

            for option_id in range(self.n_f_nets*self.option_dim):
                if ids_require_option[option_id+self.relocation_dim]:
                    current_idx=self.neighbor_id[hex_ids]
                    d_id=option_id%self.option_dim
                    f_id=option_id//self.option_dim
                    #save time
                    if len(saved_f[f_id])==0:
                        f_vals=self.get_local_f_by_f(global_state=global_states,fid=f_id,local_state=local_state)
                        saved_f[f_id]=f_vals
                    else:
                        f_vals=saved_f[f_id]

                    # full_option_values = f_vals[current_idx,dims[d_id]].reshape(len(current_idx),self.relocation_dim)#get all the f values
                    # print('shape of idx',current_idx.shape, 'shape of full option values', full_option_values.shape)
                    # print('shape of curren tf',current_f.shape)
                    # full_option_values=signs[option_id]*np.abs(full_option_values-c[option_id])
                    # full_option_values=signs[d_id]*np.abs(full_option_values)
                    # full_option_values=full_option_values[ids_require_option[option_id+self.relocation_dim]]
                    # full_option_values[:,0]=-5e5 #no self relocation
                    # here mask is of batch x 15 dimension, we omit the first 3 columns, which should be options.
                    # primitive_action_mask = mask[ids_require_option[option_id],
                    #                         self.option_dim:self.option_dim+7]  # only primitive actions in option generator
                    # full_option_values[primitive_action_mask] = -9e10
                    # full_option_values[:,0]=-9e10 #no self relocation

                    #all negative values for H at the location
                    f1=f_vals[current_idx,dims[d_id][0]].reshape(len(current_idx),self.relocation_dim)
                    f2=f_vals[current_idx,dims[d_id][1]].reshape(len(current_idx),self.relocation_dim)
                    full_option_values=np.sqrt(f1**2+f2**2)
                    full_option_values = signs[d_id] * np.abs(full_option_values)
                    full_option_values = full_option_values[ids_require_option[option_id + self.relocation_dim]]

                    # acts=np.argmax(full_option_values, axis=1)
                    # option_generated.append(acts)
                    #lets try a softmax implementation
                    log_softmax=torch.softmax(100*torch.from_numpy(full_option_values),dim=1)
                    print(log_softmax[0,:])
                    actions=torch.flatten(torch.multinomial(log_softmax,1)) #choose one action
                    option_generated.append(actions)
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

    def get_target_Q(self, local_state, global_state):
        return self.target_q_network.forward(local_state, global_state)

    def copy_parameter(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def copy_H_parameter(self):
        for i in range(len(self.h_network_list)):
            self.h_target_network_list[i].load_state_dict(self.h_network_list[i].state_dict())

    def copy_F_parameter(self):
        self.f_target_network.load_state_dict(self.f_network[0].state_dict())

    def soft_target_update(self, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


    def train(self, record_hist):
        # print('Main buffer = {}, h buffer={}, f buffer={}'.format(len(self.memory.memory),len(self.h_memory.memory),len(self.f_memory.memory)))
        # if len(self.memory) < 5000:
        #     return
        self.train_step += 1

        transitions = self.batch_sample()
        batch = self.memory.Transition(*zip(*transitions))

        global_state_reps = [self.global_state_dict[int(state[0] / 60)] for state in
                             batch.state]  # should be list of np.array

        global_next_state_reps = [self.global_state_dict[int(state_[0] / 60)] for state_ in
                                  batch.next_state]  # should be list of np.array

        # state_reps = [self.state_feature_constructor.construct_state_features(state) for state in batch.state]
        state_reps= self.state_feature_constructor.construct_f_features_batch(batch.state)
        state_batch = torch.from_numpy(np.array(state_reps)).to(dtype=torch.float32, device=self.device)

        # next_state_reps = [self.state_feature_constructor.construct_state_features(state_) for state_ in
        #                    batch.next_state]
        next_state_reps=self.state_feature_constructor.construct_f_features_batch(batch.next_state)
        next_state_batch = torch.from_numpy(np.array(next_state_reps)).to(device=self.device, dtype=torch.float32)


        hex_diffusion = [np.tile(self.hex_diffusion[state[1]],(1,1,1)) for state in batch.state]
        hex_diffusion_ = [np.tile(self.hex_diffusion[state_[1]],(1,1,1)) for state_ in batch.next_state]


        action_batch = torch.from_numpy(np.array(batch.action)).unsqueeze(1).to(dtype=torch.int64, device=self.device)
        reward_batch = torch.from_numpy(np.array(batch.reward)).unsqueeze(1).to(dtype=torch.float32, device=self.device)
        terminal_flag= torch.from_numpy(np.array(batch.terminate_flag)).unsqueeze(1).to(dtype=torch.int64, device=self.device)
        time_step_batch = torch.from_numpy(np.array(batch.time_steps)).unsqueeze(1).to(dtype=torch.float32, device=self.device)


        global_state_batch = torch.from_numpy(np.concatenate([np.array(global_state_reps),np.array(hex_diffusion)],axis=1)).to(dtype=torch.float32, device=self.device)
        global_next_state_batch = torch.from_numpy(np.concatenate([np.array(global_next_state_reps), np.array(hex_diffusion_)],axis=1)).to(dtype=torch.float32,
                                                                                        device=self.device)

        # print('Any weired actions?', action_batch[action_batch>=self.output_dim])
        # print('Any weired actions?', action_batch[action_batch<0])
        # print(state_batch.size(),global_state_batch.size())
        q_state_action = self.get_main_Q(state_batch, global_state_batch).gather(1, action_batch.long())
        # add a mask
        all_q_ = self.get_target_Q(next_state_batch, global_next_state_batch)
        mask = self.get_action_mask(batch.next_state, batch.valid_action_num)  # action mask for next state
        #
        # if self.option_dim>0:
        #     all_q_[:,0]=-1e3
        # option_mask = self.get_option_mask(next_state_batch,global_next_state_batch,batch.next_state)
        # all_q_[torch.from_numpy(option_mask).to(dtype=torch.bool,device=self.device)] = -9e10

        all_q_[torch.from_numpy(mask).to(dtype=torch.bool,device=self.device)] = -100
        maxq = all_q_.max(1)[0].detach().unsqueeze(1)

        # print('maximum q value {}, action selected {}, reward {} and time steps for gamma {}'.format(maxq[:5],action_batch[:5],reward_batch[:5],time_step_batch[:5]))


        if 1:
            y = reward_batch + (1-terminal_flag)*maxq*torch.pow(self.gamma,time_step_batch)
            # else:
            #     fstate_reps = self.state_feature_constructor.construct_f_features_batch(batch.state)
            #     fstate_batch = torch.from_numpy(np.array(fstate_reps)).to(dtype=torch.float32, device=self.device)
            # 
            #     next_fstate_reps = self.state_feature_constructor.construct_f_features_batch(batch.next_state)
            #     next_fstate_batch = torch.from_numpy(np.array(next_fstate_reps)).to(device=self.device, dtype=torch.float32)
            # 
            #     hrs = [state[0] // 3600 % 24 for state in batch.state]
            #     hrs_ = [state[0] // 3600 % 24 for state in batch.next_state]
            #     f_median = torch.from_numpy(self.f_median_episode[hrs_]).to(dtype=torch.float32, device=self.device).unsqueeze(1)
            #     # f_max= torch.from_numpy(self.f_max_episode[hrs_]).to(dtype=torch.float32, device=self.device).unsqueeze(1)
            #     # fo_median = torch.from_numpy(self.fo_median[hrs]).to(dtype=torch.float32, device=self.device).unsqueeze(1)
            # 
            #     f_s_=torch.zeros(reward_batch.shape[0],2).to(dtype=torch.float32, device=self.device)
            #     f_s = torch.zeros(reward_batch.shape[0],2).to(dtype=torch.float32, device=self.device)
            #     # f_s_=torch.zeros_like(reward_batch)
            # 
            #
            # 
            #     # hrs=np.array(hrs)
            #     # for h in range(24):
            #     #     if self.trained[h]>0:
            #     #         idx=hrs==h #get those
            #     #         g_state_batch=global_state_batch[idx,:];g_next_state_batch=global_next_state_batch[idx,:]
            #     #         f_batch=fstate_batch[idx,:];fnext_batch=next_fstate_batch[idx,:]
            #     #
            #     #         # print(f_batch.shape)
            #     #         # print(g_state_batch.shape)
            #     #         f_s_[idx,:]=self.f_network[h].forward(g_next_state_batch, fnext_batch).detach()
            #     #         # f_s[idx,:] = self.f_network[h].forward(g_state_batch, f_batch).detach() #these are hourly values
            # 
            # 
            # 
            #     f_s_ = self.f_network[0].forward(global_next_state_batch, next_fstate_batch).detach()
            # 
            #     # f_s = self.f_network.forward(global_state_batch, fstate_batch).detach()
            # 
            #     # sample_mean=0.5*torch.mean(f_s_)+0.5*torch.mean(f_s)
            #     # f_s-=sample_mean
            #     # f_s_-=sample_mean
            # 
            #     # print('max f_s {}, min f_s {}, f_s_ mean {}, f_median {} , reward {} and reshaped reward {}'.format(torch.max(f_s_), torch.min(f_s_),f_s_.mean(), f_median.mean(),\
            #     #       reward_batch.mean(), 1 *  (f_s_-f_s)[:5] ))
            #     # y = reward_batch + (1 - terminal_flag) * maxq * torch.pow(self.gamma, time_step_batch)
            # 
            #     if self.f_train_step==0:
            #         coeff=0
            #     else:
            #         coeff=0.2  #0.1 seems to work well
            # 
            # 
            #     # reward_batch[reward_batch<0]=reward_batch[reward_batch<0]*0.5
            #     penalty=coeff*torch.norm(f_s_,dim=1,keepdim=True)
            #     print('First few fs...{}\n First penalties{}'.format(f_s_[:4,:],penalty[:4]))
            #     print('max f_s {}, min f_s {}, f_s_ mean {}, f_median {} \n reward {} \n reshaped reward {}'.format(torch.max(penalty), torch.min(penalty),penalty.mean(), penalty.mean(),\
            #           reward_batch[:4], (reward_batch-penalty)[:4] ))
            # 
            # 
            #     # y = reward_batch + coeff*abs(f_s_-f_median)+ (1 - terminal_flag) * maxq * torch.pow(self.gamma, time_step_batch)
            #     # y = reward_batch - coeff * torch.minimum(abs(f_s_-f_median-1),abs(f_s_-f_median+1)) + (1 - terminal_flag) * maxq * torch.pow(self.gamma, time_step_batch) #distance to 0,5 or -0.5
            #     y = reward_batch - penalty+ (
            #                 1 - terminal_flag) * maxq * torch.pow(self.gamma, time_step_batch)  # distance to 0,5 or -0.5

            #
            # yz=reward_batch + (1-terminal_flag)*maxq*torch.pow(self.gamma,time_step_batch)

            loss = F.smooth_l1_loss(q_state_action, y)


            #
            # self.scaler.scale(loss).backward()
            # self.scaler.unscale_(self.optimizer)
            # nn.utils.clip_grad_norm_(self.q_network.parameters(), self.clipping_value)
            # self.scaler.step(self.optimizer)
            # self.scaler.update()
            # self.optimizer.zero_grad()

            loss.backward()
            # nn.utils.clip_grad_norm_(self.q_network.parameters(), self.clipping_value)
            self.optimizer.step()

            for param in self.q_network.parameters():
                param.grad=None

            # self.lr_scheduler.step()
            self.writer.add_scalar('main_dqn/train_loss',loss, self.train_step)
            self.writer.add_scalar('main_dqn/maxQ', maxq.mean(), self.train_step)
            self.writer.add_scalar('main_dqn/R+Q', y.mean(), self.train_step)
            # self.writer.add_scalar('main_dqn/option_percent', y.mean(), self.train_step)
            # self.record_list.append([self.train_step, round(float(loss),3), round(float(reward_batch.view(-1).mean()),3),self.optimizer.state_dict()['param_groups'][0]['lr'],batch.reward[0], batch.state[0][-1]])
            print('Training step is {}; Learning rate is {}; Epsilon is {}; Average loss is {:.3f}, number of topion used={}'.format(self.train_step,self.lr_scheduler.get_lr(),round(self.decayed_epsilon,4),loss,self.n_f_nets*self.option_dim))


    def train_f(self,times=1000):
        #retrain the f network
        if LIMIT_ONE_NET:
            if self.n_f_nets>=1:
                return
        # if len(self.f_network)>5:
        #     return
        self.f_network.append(F_Network_all(INPUT_DIM,self.option_dim//2))
        self.f_network[-1].to(self.device)
        self.f_optimizer=torch.optim.Adam(self.f_network[-1].parameters(), lr=1e-3)
        # self.trained=[0 for _ in range(24)]

        # random.shuffle(self.f_memory.memory)  # random shuffle
        batch_size=32
        if 1:
            for h in range(1): #1000 iterations
                for i in range(times):
                    self.f_train_step += 1
                    if USE_RANDOM:
                        continue
                    # sample_batch=list(itertools.islice(self.f_memory, i, i+batch_size))
                    transitions=self.f_memory[h].sample(batch_size)
                    # transitions = self.f_memory.memory[i:i+batch_size]
                    sample_batch = self.f_memory[h].Transition(*zip(*transitions))
                    #first item is the first transition
                    global_state_reps = np.array([self.global_state_dict[int(state[0] / 60)] for state in
                                                          sample_batch.state])  # should be list of np.array
                    global_next_state_reps = np.array([self.global_state_dict[int(state[0] / 60)] for state in
                                                          sample_batch.next_state])  # should be list of np.array

                    state_reps = self.state_feature_constructor.construct_f_features_batch(sample_batch.state)
                    next_state_reps = self.state_feature_constructor.construct_f_features_batch(sample_batch.next_state)




                    hex_diffusion = [np.tile(self.hex_diffusion[state[1]], (1, 1, 1)) for state in sample_batch.state]
                    hex_diffusion_ = [np.tile(self.hex_diffusion[state[1]], (1, 1, 1)) for state in sample_batch.next_state]

                    state_batch = torch.from_numpy(state_reps).to(dtype=torch.float32, device=self.device)
                    next_state_batch = torch.from_numpy(next_state_reps).to(device=self.device,
                                                                                              dtype=torch.float32)


                    global_state_batch = torch.from_numpy(
                                np.concatenate([np.array(global_state_reps), np.array(hex_diffusion)], axis=1)).to(
                                dtype=torch.float32,
                                device=self.device)
                    global_next_state_batch = torch.from_numpy(
                                np.concatenate([np.array(global_next_state_reps), np.array(hex_diffusion_)], axis=1)).to(
                                dtype=torch.float32, device=self.device)


                    transitions2=self.f_memory[h].sample(batch_size)
                    # transitions = self.f_memory.memory[i:i+batch_size]
                    sample_batch2 = self.f_memory[h].Transition(*zip(*transitions2))
                    #first item is the first transition
                    global_state_reps2 = np.array([self.global_state_dict[int(state[0] / 60)] for state in
                                                          sample_batch2.state])  # should be list of np.array
                    state_reps2 = self.state_feature_constructor.construct_f_features_batch(sample_batch2.state)
                    hex_diffusion2 = [np.tile(self.hex_diffusion[state[1]], (1, 1, 1)) for state in sample_batch2.state]
                    state_batch2 = torch.from_numpy(state_reps2).to(dtype=torch.float32, device=self.device)
                    global_state_batch2 = torch.from_numpy(
                                np.concatenate([np.array(global_state_reps2), np.array(hex_diffusion2)], axis=1)).to(
                                dtype=torch.float32,
                                device=self.device)


                    if 1:
                        f_values=self.f_network[-1].forward(global_state_batch,state_batch)
                        f_values_=self.f_network[-1].forward(global_next_state_batch,next_state_batch)
                        f_values2=self.f_network[-1].forward(global_state_batch2,state_batch2)

                        loss_pos=pos_loss(f_values,f_values_)
                        loss_neg=neg_loss(f_values2)


                        # self.f_optimizer[h].zero_grad()
                    eta = 1# lagrangian multiplier, it was assumed as 1.0 in all scenarios, so we also try 1.0.
                    loss=loss_pos+eta*loss_neg


                    print('F training loss for step {} is {:.3f}, positive loss {:.3f}, negative loss {:.3f} number of samples is {}'.format(i, loss,loss_pos, loss_neg,f_values.shape[0]))
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.f_network[0].parameters(), 0.5)
                    self.f_optimizer.step()
                    self.writer.add_scalar('f_train/train_loss', loss, self.f_train_step)
                    for param in self.f_network[-1].parameters():
                        param.grad = None


        #after training, update
        self.n_f_nets+=1
        for opt in range(self.option_dim):
            h_network = OptionNetwork(self.input_dim,1+6)
            h_target_network=TargetOptionNetwork(self.input_dim, 1+6)
            self.h_network_list.append(h_network.to(self.device))
            self.h_target_network_list.append(h_target_network.to(self.device))
            self.h_optimizer.append(torch.optim.Adam(self.h_network_list[-1].parameters(), lr=1e-3))
            self.option_train_step.append(0)

        for steps in range(2000):
            self.train_h_trajectory_add(times=1,start_f=self.n_f_nets-1) #afer adding the new option networks, pretrain it for 5k times
            if steps%100==0:
                self.copy_H_parameter()

    def train_f_online(self,times=600):
        #retrain the f networks
        if self.n_f_nets<1:
            self.f_network.append(F_Network_all(INPUT_DIM,self.option_dim*2))
            self.f_network[-1].to(self.device)
            self.f_optimizer=torch.optim.Adam(self.f_network[-1].parameters(), lr=1e-3)
            for opt in range(self.option_dim):
                h_network = OptionNetwork(self.input_dim, 1 + 6)
                h_target_network = TargetOptionNetwork(self.input_dim, 1 + 6)
                self.h_network_list.append(h_network.to(self.device))
                self.h_target_network_list.append(h_target_network.to(self.device))
                self.h_optimizer.append(torch.optim.Adam(self.h_network_list[-1].parameters(), lr=1e-3))

        # else:
        #     self.f_network=[F_Network_all(INPUT_DIM,self.option_dim*2)]
        #     self.f_network[-1].to(self.device)
        #     self.f_optimizer=torch.optim.Adam(self.f_network[-1].parameters(), lr=1e-3)

        # self.trained=[0 for _ in range(24)]

        # random.shuffle(self.f_memory.memory)  # random shuffle
        if 1:
            for h in range(1): #1000 iterations
                batch_size = 32
                # for i in range(0,len(self.f_memory.memory)-batch_size,batch_size):
                # if len(self.f_memory[h].memory[0].memory)<batch_size:
                #     continue

                for i in range(times):
                    self.f_train_step += 1
                    # sample_batch=list(itertools.islice(self.f_memory, i, i+batch_size))
                    transitions=self.f_memory[h].sample(batch_size)
                    # transitions = self.f_memory.memory[i:i+batch_size]
                    sample_batch = self.f_memory[h].Transition(*zip(*transitions))
                    #first item is the first transition
                    global_state_reps = np.array([self.global_state_dict[int(state[0] / 60)] for state in
                                                          sample_batch.state])  # should be list of np.array
                    global_next_state_reps = np.array([self.global_state_dict[int(state[0] / 60)] for state in
                                                          sample_batch.next_state])  # should be list of np.array

                    state_reps = self.state_feature_constructor.construct_f_features_batch(sample_batch.state)
                    next_state_reps = self.state_feature_constructor.construct_f_features_batch(sample_batch.next_state)




                    hex_diffusion = [np.tile(self.hex_diffusion[state[1]], (1, 1, 1)) for state in sample_batch.state]
                    hex_diffusion_ = [np.tile(self.hex_diffusion[state[1]], (1, 1, 1)) for state in sample_batch.next_state]

                    state_batch = torch.from_numpy(state_reps).to(dtype=torch.float32, device=self.device)
                    next_state_batch = torch.from_numpy(next_state_reps).to(device=self.device,
                                                                                              dtype=torch.float32)


                    global_state_batch = torch.from_numpy(
                                np.concatenate([np.array(global_state_reps), np.array(hex_diffusion)], axis=1)).to(
                                dtype=torch.float32,
                                device=self.device)
                    global_next_state_batch = torch.from_numpy(
                                np.concatenate([np.array(global_next_state_reps), np.array(hex_diffusion_)], axis=1)).to(
                                dtype=torch.float32, device=self.device)

                    transitions2=self.f_memory[h].sample(batch_size)
                    # transitions = self.f_memory.memory[i:i+batch_size]
                    sample_batch2 = self.f_memory[h].Transition(*zip(*transitions2))
                    #first item is the first transition
                    global_state_reps2 = np.array([self.global_state_dict[int(state[0] / 60)] for state in
                                                          sample_batch2.state])  # should be list of np.array
                    state_reps2 = self.state_feature_constructor.construct_f_features_batch(sample_batch2.state)
                    hex_diffusion2 = [np.tile(self.hex_diffusion[state[1]], (1, 1, 1)) for state in sample_batch2.state]
                    state_batch2 = torch.from_numpy(state_reps2).to(dtype=torch.float32, device=self.device)
                    global_state_batch2 = torch.from_numpy(
                                np.concatenate([np.array(global_state_reps2), np.array(hex_diffusion2)], axis=1)).to(
                                dtype=torch.float32,
                                device=self.device)

                    if 1:
                        f_values=self.f_network[-1].forward(global_state_batch,state_batch)
                        f_values2 = self.f_network[-1].forward(global_state_batch2, state_batch2)
                        f_values_=self.f_network[-1].forward(global_next_state_batch,next_state_batch)
                        # f_values=torch.cat((f_values,t1),1)
                        # f_values_ = torch.cat((f_values_, t2), 1)
                        eta = 1 # lagrangian multiplier, it was assumed as 1.0 in all scenarios, so we also try 1.0.
                        delta = 1 #according to Yifan Wu et al. 2019
                        f1=1
                        loss_pos=pos_loss(f_values,f_values_)
                        loss_neg=neg_loss(f_values2)
                        # loss_neg = neg_loss_smooth(f_values_)
                        loss=loss_pos+eta*loss_neg


                        print('F training loss for step {} is {:.3f}, pos and neg loss are {:.3f},{:.3f}'.format(i,loss,loss_pos,loss_neg))
                        # self.f_optimizer[h].zero_grad()
                        self.f_optimizer.zero_grad()
                        loss.backward()
                        # torch.nn.utils.clip_grad_norm_(self.f_network[0].parameters(), 0.5)
                        self.f_optimizer.step()
                        self.writer.add_scalar('f_train/train_loss', loss, self.f_train_step)


        #after training, update
        self.n_f_nets=1
        # self.f_memory=[F_ReplayMemory(int(1e6))]


    def train_f_online_by_trajectory(self,times=600):
        #retrain the f networks
        if self.n_f_nets<1:
            self.f_network.append(F_Network_all(INPUT_DIM,self.option_dim*2))
            self.f_network[-1].to(self.device)
            self.f_optimizer=torch.optim.Adam(self.f_network[-1].parameters(), lr=1e-4)
            for opt in range(self.option_dim):
                h_network = OptionNetwork(self.input_dim, 1 + 6)
                h_target_network = TargetOptionNetwork(self.input_dim, 1 + 6)
                self.h_network_list.append(h_network.to(self.device))
                self.h_target_network_list.append(h_target_network.to(self.device))
                self.h_optimizer.append(torch.optim.Adam(self.h_network_list[-1].parameters(), lr=1e-3))
        # else:
        #     self.f_network=[F_Network_all(INPUT_DIM,self.option_dim*2)]
        #     self.f_network[-1].to(self.device)
        #     self.f_optimizer=torch.optim.Adam(self.f_network[-1].parameters(), lr=1e-3)

        # self.trained=[0 for _ in range(24)]

        # random.shuffle(self.f_memory.memory)  # random shuffle
        if 1:
            for h in range(1): #1000 iterations
                batch_size = 32
                # for i in range(0,len(self.f_memory.memory)-batch_size,batch_size):
                if len(self.trajectory_memory.memory)<batch_size:
                    continue

                for i in range(times):
                    t1=time.time()
                    self.f_train_step += 1
                    # sample_batch=list(itertools.islice(self.f_memory, i, i+batch_size))
                    transitions=self.trajectory_memory.sample(batch_size)
                    # transitions = self.f_memory.memory[i:i+batch_size]
                    sample_batch = self.trajectory_memory.Transition(*zip(*transitions))
                    print('sampling time cost: ', time.time()-t1)
                    t1 = time.time()
                    #first item is the first transition
                    global_state_reps = np.array([self.global_state_dict[int(state[0][0] / 60)] for trajectory in sample_batch.trajectory for state in
                                                          trajectory]) # should be list of np.array
                    global_next_state_reps = np.array([self.global_state_dict[int(state[1][0] / 60)] for trajectory in sample_batch.trajectory for state in
                                                          trajectory])

                    state_reps = self.state_feature_constructor.construct_f_features_batch([state[0] for trajectory in sample_batch.trajectory for state in trajectory])
                    next_state_reps = self.state_feature_constructor.construct_f_features_batch([state[1] for trajectory in sample_batch.trajectory for state in trajectory])


                    hex_diffusion = [np.reshape(self.hex_diffusion[state[0][1]], (1, MAP_HEIGHT, MAP_WIDTH)) for trajectory in sample_batch.trajectory for state in trajectory]
                    hex_diffusion_ = [np.reshape(self.hex_diffusion[state[1][1]],(1, MAP_HEIGHT, MAP_WIDTH)) for trajectory in sample_batch.trajectory for state in trajectory]


                    state_batch = torch.from_numpy(state_reps).to(dtype=torch.float32, device=self.device)
                    next_state_batch = torch.from_numpy(next_state_reps).to(device=self.device,
                                                                                                  dtype=torch.float32)

                    global_state_batch = torch.from_numpy(
                                    np.concatenate([global_state_reps, hex_diffusion], axis=1)).to(
                                    dtype=torch.float32,
                                    device=self.device)
                    global_next_state_batch = torch.from_numpy(
                                    np.concatenate([global_next_state_reps, hex_diffusion_], axis=1)).to(
                                    dtype=torch.float32, device=self.device)
                    print('Data prep cost: ', time.time() - t1)
                    if 1:
                        f_values=self.f_network[-1].forward(global_state_batch,state_batch)
                        f_values_=self.f_network[-1].forward(global_next_state_batch,next_state_batch)

                        loss_pos=pos_loss(f_values,f_values_)
                        loss_neg=neg_loss(f_values)


                        # self.f_optimizer[h].zero_grad()
                    eta = 1# lagrangian multiplier, it was assumed as 1.0 in all scenarios, so we also try 1.0.
                    loss=loss_pos+eta*loss_neg


                    print('F training loss for step {} is {:.3f}, positive loss {:.3f}, negative loss {:.3f} number of samples is {}'.format(i, loss,loss_pos, loss_neg,f_values.shape[0]))

                    self.f_optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.f_network[0].parameters(), 0.5)
                    self.f_optimizer.step()
                    self.writer.add_scalar('f_train/train_loss', loss, self.f_train_step)
                    #release the memory
                    # for param in self.f_network[-1].parameters():
                    #     param.grad = None

        #after training, update
        self.n_f_nets=1
        # self.f_memory=[F_ReplayMemory(int(1e6))]


    def reset_f_memory(self):
        self.f_memory = [F_ReplayMemory(int(1e6))]
        # self.trajectory_memory=Trajectory_ReplayMemory(int(5e5))

    def record_f_distance(self):
        # find the mean threshold and percentile values for each hour
        self.dist_samples=[[] for _ in range(24)]
        # self.f_median=[]
        # self.f_lower = []
        # self.f_higher = []
        with torch.no_grad():
            if 1:
                batch_size = 128
                for i in range(0,400):
                    # sample_batch=list(itertools.islice(self.f_memory, i, i+batch_size))
                    transitions = self.f_memory[0].sample(batch_size)
                    sample_batch = self.f_memory[0].Transition(*zip(*transitions))
                    #first item is the first transition
                    global_state_reps = np.array([self.global_state_dict[int(state[0] / 60)] for state in
                                                          sample_batch.state])  # should be list of np.array
                    global_next_state_reps = np.array([self.global_state_dict[int(state[0] / 60)] for state in
                                                          sample_batch.next_state])  # should be list of np.array
                    state_reps = self.state_feature_constructor.construct_f_features_batch(sample_batch.state)
                    next_state_reps = self.state_feature_constructor.construct_f_features_batch(sample_batch.next_state)
                    hex_diffusion = [np.tile(self.hex_diffusion[state[1]], (1, 1, 1)) for state in sample_batch.state]
                    hex_diffusion_ = [np.tile(self.hex_diffusion[state[1]], (1, 1, 1)) for state in sample_batch.next_state]

                    state_batch = torch.from_numpy(np.array(state_reps)).to(dtype=torch.float32, device=self.device)
                    next_state_batch = torch.from_numpy(np.array(next_state_reps)).to(device=self.device,
                                                                                              dtype=torch.float32)
                    global_state_batch = torch.from_numpy(
                                np.concatenate([np.array(global_state_reps), np.array(hex_diffusion)], axis=1)).to(
                                dtype=torch.float32,
                                device=self.device)
                    global_next_state_batch = torch.from_numpy(
                                np.concatenate([np.array(global_next_state_reps), np.array(hex_diffusion_)], axis=1)).to(
                                dtype=torch.float32, device=self.device)
                    f_values_=self.f_network[-1].forward(global_next_state_batch,next_state_batch).cpu().numpy()
                    f_values = self.f_network[-1].forward(global_state_batch, state_batch).cpu().numpy()

                    dist1=(f_values[:,1]**2+f_values[:,2]**2)**0.5
                    dist2 =(f_values_[:, 1] ** 2 + f_values_[:, 2] ** 2) ** 0.5

                    hrs = [state[0] // 3600 % 24 for state in sample_batch.state]

                    [self.dist_samples[h].append([d1,d2]) for h,d1,d2 in zip(hrs,dist1,dist2)]
            #         self.dist_samples+=list(dist1)
            #         self.dist_samples+=list(dist2)
            # self.dist_samples=np.array(self.dist_samples)
            self.dist_samples=[np.array([i for sublist in item for i in sublist]) for item in self.dist_samples]



    def reset_h(self):
        self.load_option_networks(self.option_dim)
        self.h_optimizer = [torch.optim.Adam(self.h_network_list[i].parameters(), lr=1e-4) for i in
                        range(len(self.h_network_list))]


    def train_h(self):
        if len(self.memory) < self.batch_size:
            print('batches in replay buffer is {}'.format(len(self.memory)))
            return
        if self.n_f_nets==0 or self.f_train_step==0:
            print('Not training H network due to insufficient F trainings')
            return


        self.h_train_step+=1
        transitions = self.H_batch_sample(128)
        batch = self.h_memory.Transition(*zip(*transitions))



        # hrs=[state[0]//3600%24 for state in batch.state]
        # hrs_=[state[0]//3600%24 for state in batch.next_state]

        global_state_reps = np.array([self.global_state_dict[int(state[0] / 60)] for state in
                             batch.state])  # should be list of np.array

        global_next_state_reps = np.array([self.global_state_dict[int(state_[0] / 60)] for state_ in
                                  batch.next_state]) # should be list of np.array

        next_zones = np.array([state_[1] for state_ in batch.next_state])  # zone id for choosing actions

        state_reps = self.state_feature_constructor.construct_f_features_batch(batch.state)  #[self.state_feature_constructor.construct_state_features(state) for state in batch.state]
        next_state_reps = self.state_feature_constructor.construct_f_features_batch(batch.next_state)#[self.state_feature_constructor.construct_state_features(state_) for state_ in
                           #batch.next_state]

        hex_diffusion = [np.tile(self.hex_diffusion[state[1]], (1, 1, 1)) for state in batch.state]
        hex_diffusion_ = [np.tile(self.hex_diffusion[state_[1]], (1, 1, 1)) for state_ in batch.next_state]

        state_batch = torch.from_numpy(np.array(state_reps)).to(dtype=torch.float32, device=self.device)

        action_batch = torch.from_numpy(np.array(batch.action)).unsqueeze(1).to(dtype=torch.int64, device=self.device)

        time_step_batch = torch.from_numpy(np.array(batch.time_steps)).unsqueeze(1).to(dtype=torch.float32, device=self.device)

        trip_flag=torch.from_numpy(np.array(batch.trip_flag)).unsqueeze(1).to(dtype=torch.float32, device=self.device)

        next_state_batch = torch.from_numpy(np.array(next_state_reps)).to(device=self.device, dtype=torch.float32)
        global_state_batch = torch.from_numpy(
            np.concatenate([np.array(global_state_reps), np.array(hex_diffusion)], axis=1)).to(dtype=torch.float32,
                                                                                               device=self.device)
        global_next_state_batch = torch.from_numpy(
            np.concatenate([np.array(global_next_state_reps), np.array(hex_diffusion_)], axis=1)).to(
            dtype=torch.float32, device=self.device)

        # f_median=torch.from_numpy(self.f_median[hrs]).to(dtype=torch.float32, device=self.device).unsqueeze(1)
        # fo_median = torch.from_numpy(self.fo_median[hrs]).to(dtype=torch.float32, device=self.device).unsqueeze(1)

        # f_lower=torch.from_numpy(self.f_lower[hrs]).to(dtype=torch.float32, device=self.device).unsqueeze(1)
        # f_lower_ = torch.from_numpy(self.f_lower[hrs_]).to(dtype=torch.float32, device=self.device).unsqueeze(1)
        # f_upper = torch.from_numpy(self.f_upper[hrs]).to(dtype=torch.float32, device=self.device).unsqueeze(1)
        # f_upper_ = torch.from_numpy(self.f_upper[hrs_]).to(dtype=torch.float32, device=self.device).unsqueeze(1)

        f_s=self.f_network[-1].forward(global_state_batch,state_batch).detach()
        f_s_ = self.f_network[-1].forward(global_next_state_batch, next_state_batch).detach()
        # print('the center of the median is {},{}'.format(self.median_x, self.median_y))
        #
        # f_s_[:,0]=f_s_[:,0]-self.median_x
        # f_s_[:,1]=f_s_[:,1]-self.median_y

        coeff=1

        # penalty = coeff * torch.norm(f_s_, dim=1, keepdim=True)
        #
        # print('The median of l2-dist values is {}, upper 75th is {}, lower 25th is {}'.format(torch.quantile(penalty,.5),torch.quantile(penalty,0.75),torch.quantile(penalty,0.25)))
        # print('Median of F ({}, {})'.format(torch.quantile(f_s_[:,1],0.5),torch.quantile(f_s_[:,2],0.5)))

        mask = self.get_action_mask(batch.next_state, batch.valid_action_num)  # action mask for next state
        #
        # fo_s=self.fo_network.forward(global_state_batch,state_batch).detach()
        # fo_s_ = self.fo_network.forward(global_next_state_batch, next_state_batch).detach()
        signs = [1,-1]


        dist_=f_s_[:,1:].pow(2).sum(-1).pow(0.5).unsqueeze(1)
        dist = f_s[:, 1:].pow(2).sum(-1).pow(0.5).unsqueeze(1)


        distg=(f_s_[:,1:]-f_s[:,1:]).pow(2).sum(-1).pow(0.5).unsqueeze(1)

        for opt in range(self.option_dim):
            q_state_action = self.h_network_list[opt].forward(state_batch, global_state_batch).gather(1, action_batch.long())
            all_q_ = self.h_target_network_list[opt].forward(next_state_batch, global_next_state_batch).detach()#lets change this to global state batch and see if error continues
            maxq = all_q_.max(1)[0].unsqueeze(1)

            discount=0.8

            time_step_batch[time_step_batch==0]=1

            rate = (discount ** time_step_batch - 1) / (time_step_batch * (discount - 1))  # discounted rate

            if opt==0:
                # y = -1 + 0.95 * maxq * (
                #             1 - trip_flag)  # +max_psu*trip_flag  # mix of two #use the minimum number of steps

                if self.option_dim>1: #more than 1 option
                    y=-distg+0.1*trip_flag+ discount*maxq
                else:
                    y =  -1+ discount * maxq * (1 - trip_flag)
            else:

                # dist=dist.cpu().numpy(); dist_=dist_.cpu().numpy()
                # pd=(self.dist_samples < dist[:, None]).mean(axis=1) #percentage of distance
                # pd_ =(self.dist_samples <dist_[:, None]).mean(axis=1)  # percentage of distance
                # pd=dist
                # pd_=dist_

                delta_d=signs[opt-1]*(dist-dist_) #opt 2--- move away from 0,  opt 3: move close to 0

                # pseudo_reward=torch.from_numpy(delta_d).to(self.device,dtype=torch.float32).unsqueeze(1)   #(f_s_[:,dims[opt]]-f_s[:,dims[opt]]) $d1-d works without additional items)
                pseudo_reward=delta_d.unsqueeze(1)

                y = delta_d+ 0.1*trip_flag+ maxq*discount# +max_psu*trip_flag  # mix of two #use the minimum number of steps

           #  y = pseudo_reward #
            # print(middle_terminal_flag.shape, f_s.shape, pseudo_reward.shape, maxq.shape, y.shape)
            loss = F.smooth_l1_loss(q_state_action,y)
            print('H network {} training loss for step {} is {}, average reward is {}'.format(opt,self.h_train_step,loss,y.mean()))
            #
            # if self.train_step%100==0:
            #     print('Max Q={}, Max target Q={}, Loss = {}, Gamma={}, mean diff={}, mean reward={}'.format(torch.max(q_state_action), torch.max(maxq),loss, self.gamma,mean_diff,mean_pseudo))
            #     print('Mean of main f={}, mean of target f={}'.format(f_s.mean(),f_s_.mean()))
            #     with open('saved_h/ht_train_log_{}.csv'.format(self.num_option),'a') as f:
            #         f.writelines('Train step={}, Max Q={}, Max target Q={}, Loss = {}, Mean f_diff={}, Mean pseudo-reward={}\n'.format(self.train_step,torch.max(q_state_action), torch.max(maxq),loss,mean_diff,mean_pseudo))
            self.writer.add_scalar('h_train/train_loss_{}'.format(opt), loss, self.h_train_step)
            self.writer.add_scalar('h_train/max_q_{}'.format(opt), y.mean(), self.h_train_step)
            self.h_optimizer[opt].zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(self.h_network_list[0].parameters(), self.clipping_value)
            self.h_optimizer[opt].step()
        # self.lr_scheduler.step()


    def train_h_trajectory(self):
        if len(self.memory) < self.batch_size:
            print('batches in replay buffer is {}'.format(len(self.memory)))
            return
        if self.n_f_nets==0 or self.f_train_step==0:
            print('Not training H network due to insufficient F trainings')
            return


        self.h_train_step+=1
        batch_size=32
        if len(self.trajectory_memory.memory)<batch_size*5:
            print('insufficietn sample')
            return
        transitions = self.trajectory_memory.sample(batch_size)
        # transitions = self.f_memory.memory[i:i+batch_size]
        sample_batch = self.trajectory_memory.Transition(*zip(*transitions))
        t1 = time.time()
        # first item is the first transition
        global_state_reps = np.array(
            [self.global_state_dict[int(state[0][0] / 60)] for trajectory in sample_batch.trajectory for state in
             trajectory])  # should be list of np.array
        global_next_state_reps = np.array(
            [self.global_state_dict[int(state[1][0] / 60)] for trajectory in sample_batch.trajectory for state in
             trajectory])

        state_reps = self.state_feature_constructor.construct_f_features_batch(
            [state[0] for trajectory in sample_batch.trajectory for state in trajectory])
        next_state_reps = self.state_feature_constructor.construct_f_features_batch(
            [state[1] for trajectory in sample_batch.trajectory for state in trajectory])

        hex_diffusion = [np.reshape(self.hex_diffusion[state[0][1]], (1, MAP_HEIGHT, MAP_WIDTH)) for trajectory in
                         sample_batch.trajectory for state in trajectory]
        hex_diffusion_ = [np.reshape(self.hex_diffusion[state[1][1]], (1, MAP_HEIGHT, MAP_WIDTH)) for trajectory in
                          sample_batch.trajectory for state in trajectory]

        state_batch = torch.from_numpy(state_reps).to(dtype=torch.float32, device=self.device)
        next_state_batch = torch.from_numpy(next_state_reps).to(device=self.device,
                                                                dtype=torch.float32)

        global_state_batch = torch.from_numpy(
            np.concatenate([global_state_reps, hex_diffusion], axis=1)).to(
            dtype=torch.float32,
            device=self.device)
        global_next_state_batch = torch.from_numpy(
            np.concatenate([global_next_state_reps, hex_diffusion_], axis=1)).to(
            dtype=torch.float32, device=self.device)

        trip_flag=[state[-1] for trajectory in sample_batch.trajectory for state in trajectory]
        actions=[state[-2] for trajectory in sample_batch.trajectory for state in trajectory]
        time_steps=[state[-3] for trajectory in sample_batch.trajectory for state in trajectory]

        trip_flag=torch.from_numpy(np.array(trip_flag)).unsqueeze(1).to(dtype=torch.float32, device=self.device)
        action_batch=torch.from_numpy(np.array(actions)).unsqueeze(1).to(dtype=torch.float32, device=self.device)
        time_steps = torch.from_numpy(np.array(time_steps)).unsqueeze(1).to(dtype=torch.float32, device=self.device)

        f_s=self.f_network[-1].forward(global_state_batch,state_batch).detach()
        f_s_ = self.f_network[-1].forward(global_next_state_batch, next_state_batch).detach()


        signs = [1,-1]

        dist_=f_s_[:,1:].pow(2).sum(-1).pow(0.5).unsqueeze(1)
        dist = f_s[:, 1:].pow(2).sum(-1).pow(0.5).unsqueeze(1)


        distg=(f_s_[:,1:]-f_s[:,1:]).pow(2).sum(-1).pow(0.5).unsqueeze(1)

        for opt in range(self.option_dim):
            q_state_action = self.h_network_list[opt].forward(state_batch, global_state_batch).gather(1, action_batch.long())
            all_q_ = self.h_target_network_list[opt].forward(next_state_batch, global_next_state_batch).detach()#lets change this to global state batch and see if error continues
            maxq = all_q_.max(1)[0].unsqueeze(1)

            discount=0.95

            if opt==0:
                # y = -1 + 0.95 * maxq * (
                #             1 - trip_flag)  # +max_psu*trip_flag  # mix of two #use the minimum number of steps

                if self.option_dim>1: #more than 1 option
                    y=random.random()+0*-distg*(1-trip_flag)+discount*maxq
                    # y=-1+discount*maxq*(1-trip_flag)
                    print('train opt 1')
                else:
                    y =  -1+ discount * maxq
            else:
                if opt==1:
                    delta_d=(dist-dist_)#opt 2--- move away from 0,  opt 3: move close to 0
                    # pseudo_reward=torch.from_numpy(delta_d).to(self.device,dtype=torch.float32).unsqueeze(1)   #(f_s_[:,dims[opt]]-f_s[:,dims[opt]]) $d1-d works without additional items)
                    pseudo_reward=delta_d
                    y = random.random()+0*dist_*(1+trip_flag)+discount*maxq# +max_psu*trip_flag  # mix of two #use the minimum number of steps
                    # y=1+discount*maxq*trip_flag
                    print('train opt 2')
                if opt==2:
                    delta_d=dist_-dist #opt 2--- move away from 0,  opt 3: move close to 0
                    # pseudo_reward=torch.from_numpy(delta_d).to(self.device,dtype=torch.float32).unsqueeze(1)   #(f_s_[:,dims[opt]]-f_s[:,dims[opt]]) $d1-d works without additional items)
                    pseudo_reward=delta_d
                    y = random.random()+0*-torch.abs(dist_)*(1-trip_flag)+discount*maxq# +max_psu*trip_flag  # mix of two #use the minimum number of steps
                    # y=1+discount*maxq*trip_flag
                    print('train opt 3')

           #  y = pseudo_reward #
            # print(middle_terminal_flag.shape, f_s.shape, pseudo_reward.shape, maxq.shape, y.shape)
            print(q_state_action.shape, y.shape)
            loss = F.smooth_l1_loss(q_state_action,y)
            print('H network {} training loss for step {} is {}, average reward is {}'.format(opt,self.h_train_step,loss,y.mean()))
            #
            # if self.train_step%100==0:
            #     print('Max Q={}, Max target Q={}, Loss = {}, Gamma={}, mean diff={}, mean reward={}'.format(torch.max(q_state_action), torch.max(maxq),loss, self.gamma,mean_diff,mean_pseudo))
            #     print('Mean of main f={}, mean of target f={}'.format(f_s.mean(),f_s_.mean()))
            #     with open('saved_h/ht_train_log_{}.csv'.format(self.num_option),'a') as f:
            #         f.writelines('Train step={}, Max Q={}, Max target Q={}, Loss = {}, Mean f_diff={}, Mean pseudo-reward={}\n'.format(self.train_step,torch.max(q_state_action), torch.max(maxq),loss,mean_diff,mean_pseudo))
            self.writer.add_scalar('h_train/train_loss_{}'.format(opt), loss, self.h_train_step)
            self.writer.add_scalar('h_train/max_q_{}'.format(opt), y.mean(), self.h_train_step)
            self.h_optimizer[opt].zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(self.h_network_list[0].parameters(), self.clipping_value)
            self.h_optimizer[opt].step()
        # self.lr_scheduler.step()


    def train_h_trajectory_add(self,times=1,start_f=0):
        if len(self.memory) < self.batch_size:
            print('batches in replay buffer is {}'.format(len(self.memory)))
            return
        if self.n_f_nets==0 or self.f_train_step==0:
            print('Not training H network due to insufficient F trainings')

            return
        alpha_list=[1,1,1]
        for f_net in range(start_f,self.n_f_nets):
            for j in range(times):
                alpha=alpha_list[f_net]
                self.h_train_step+=1
                transitions = self.H_batch_sample(128)
                batch = self.h_memory.Transition(*zip(*transitions))

                # hrs=[state[0]//3600%24 for state in batch.state]
                # hrs_=[state[0]//3600%24 for state in batch.next_state]

                global_state_reps = np.array([self.global_state_dict[int(state[0] / 60)] for state in
                                              batch.state])  # should be list of np.array

                global_next_state_reps = np.array([self.global_state_dict[int(state_[0] / 60)] for state_ in
                                                   batch.next_state])  # should be list of np.array

                next_zones = np.array([state_[1] for state_ in batch.next_state])  # zone id for choosing actions

                state_reps = self.state_feature_constructor.construct_f_features_batch(
                    batch.state)  # [self.state_feature_constructor.construct_state_features(state) for state in batch.state]
                next_state_reps = self.state_feature_constructor.construct_f_features_batch(
                    batch.next_state)  # [self.state_feature_constructor.construct_state_features(state_) for state_ in
                # batch.next_state]

                hex_diffusion = [np.tile(self.hex_diffusion[state[1]], (1, 1, 1)) for state in batch.state]
                hex_diffusion_ = [np.tile(self.hex_diffusion[state_[1]], (1, 1, 1)) for state_ in batch.next_state]

                state_batch = torch.from_numpy(np.array(state_reps)).to(dtype=torch.float32, device=self.device)

                action_batch = torch.from_numpy(np.array(batch.action)).unsqueeze(1).to(dtype=torch.int64,
                                                                                        device=self.device)

                time_steps = torch.from_numpy(np.array(batch.time_steps)).unsqueeze(1).to(dtype=torch.float32,
                                                                                               device=self.device)

                trip_flag = torch.from_numpy(np.array(batch.trip_flag)).unsqueeze(1).to(dtype=torch.float32,
                                                                                        device=self.device)

                next_state_batch = torch.from_numpy(np.array(next_state_reps)).to(device=self.device,
                                                                                  dtype=torch.float32)
                global_state_batch = torch.from_numpy(
                    np.concatenate([np.array(global_state_reps), np.array(hex_diffusion)], axis=1)).to(
                    dtype=torch.float32,
                    device=self.device)
                global_next_state_batch = torch.from_numpy(
                    np.concatenate([np.array(global_next_state_reps), np.array(hex_diffusion_)], axis=1)).to(
                    dtype=torch.float32, device=self.device)


                f_s=self.f_network[f_net].forward(global_state_batch,state_batch).detach()
                f_s_ = self.f_network[f_net].forward(global_next_state_batch, next_state_batch).detach()

                dist_=f_s_[:,1:].pow(2).sum(-1).pow(0.5).unsqueeze(1)
                dist = f_s[:, 1:].pow(2).sum(-1).pow(0.5).unsqueeze(1)

                dist_gap=(f_s_[:,1:]-f_s[:,1:]).pow(2).sum(-1).pow(0.5).unsqueeze(1)


                for opt in range(self.option_dim):
                    q_state_action = self.h_network_list[f_net*self.option_dim+opt].forward(state_batch, global_state_batch).gather(1, action_batch.long())
                    all_q_ = self.h_target_network_list[f_net*self.option_dim+opt].forward(next_state_batch, global_next_state_batch).detach()#lets change this to global state batch and see if error continues
                    maxq = all_q_.max(1)[0].unsqueeze(1)
                    discount=0.9 #use 0.8


                    if opt==0:
                        if self.option_dim>1: #more than 1 option
                            y=1*(dist_gap)+alpha*trip_flag+discount*maxq #torch.pow(discount,time_steps)
                            print('train opt {}'.format(f_net*self.option_dim+opt))
                    else:
                        if opt==1:
                            delta_d=(dist-dist_)#opt 2--- move away from 0,  opt 3: move close to 0
                            # pseudo_reward=torch.from_numpy(delta_d).to(self.device,dtype=torch.float32).unsqueeze(1)   #(f_s_[:,dims[opt]]-f_s[:,dims[opt]]) $d1-d works without additional items)
                            pseudo_reward=delta_d+random.random()*USE_RANDOM
                            y = 1*pseudo_reward+(alpha*trip_flag)+discount*maxq #torch.pow(discount,time_steps)# +max_psu*trip_flag  # mix of two #use the minimum number of steps
                            print('train opt {}'.format(f_net*self.option_dim+opt))
                        if opt==2:
                            delta_d=(dist_-dist)#opt 2--- move away from 0,  opt 3: move close to 0
                            # pseudo_reward=torch.from_numpy(delta_d).to(self.device,dtype=torch.float32).unsqueeze(1)   #(f_s_[:,dims[opt]]-f_s[:,dims[opt]]) $d1-d works without additional items)
                            pseudo_reward=delta_d+random.random()*USE_RANDOM
                            y = 1*pseudo_reward+alpha*trip_flag+discount*maxq #torch.pow(discount,time_steps)# +max_psu*trip_flag  # mix of two #use the minimum number of steps
                            print('train opt {}'.format(f_net*self.option_dim+opt))
                        #
                        # if opt==3:
                        #     delta_d=-5*dist_gap#opt 2--- move away from 0,  opt 3: move close to 0
                        #     # pseudo_reward=torch.from_numpy(delta_d).to(self.device,dtype=torch.float32).unsqueeze(1)   #(f_s_[:,dims[opt]]-f_s[:,dims[opt]]) $d1-d works without additional items)
                        #     pseudo_reward=delta_d*(1-USE_RANDOM)-USE_RANDOM*random.random()
                        #     y = pseudo_reward+discount*maxq*(1-trip_flag) #torch.pow(discount,time_steps)# +max_psu*trip_flag  # mix of two #use the minimum number of steps
                        #     print('train opt {}'.format(f_net*self.option_dim+opt))


                   #  y = pseudo_reward #
                    # print(middle_terminal_flag.shape, f_s.shape, pseudo_reward.shape, maxq.shape, y.shape)
                    print(q_state_action.shape, y.shape)
                    loss = F.smooth_l1_loss(q_state_action,y)
                    print('H network {} training loss for step {} is {}, average reward is {}'.format(f_net*self.option_dim+opt,self.h_train_step,loss,y.mean()))
                    #
                    # if self.train_step%100==0:
                    #     print('Max Q={}, Max target Q={}, Loss = {}, Gamma={}, mean diff={}, mean reward={}'.format(torch.max(q_state_action), torch.max(maxq),loss, self.gamma,mean_diff,mean_pseudo))
                    #     print('Mean of main f={}, mean of target f={}'.format(f_s.mean(),f_s_.mean()))
                    #     with open('saved_h/ht_train_log_{}.csv'.format(self.num_option),'a') as f:
                    #         f.writelines('Train step={}, Max Q={}, Max target Q={}, Loss = {}, Mean f_diff={}, Mean pseudo-reward={}\n'.format(self.train_step,torch.max(q_state_action), torch.max(maxq),loss,mean_diff,mean_pseudo))
                    self.writer.add_scalar('h_train/train_loss_{}'.format(f_net*self.option_dim+opt), loss, self.h_train_step)
                    self.writer.add_scalar('h_train/max_q_{}'.format(f_net*self.option_dim+opt), y.mean(), self.h_train_step)
                    loss.backward()
                    # nn.utils.clip_grad_norm_(self.h_network_list[0].parameters(), self.clipping_value)
                    self.h_optimizer[f_net*self.option_dim+opt].step()
                    for param in self.h_network_list[f_net*self.option_dim+opt].parameters():
                        param.grad = None
        # self.lr_scheduler.step()


    def save_parameter(self,trial):
        # torch.save(self.q_network.state_dict(), self.dqn_path)
        if self.option_dim>0:
            h_net=[i.state_dict() for i in self.h_network_list]
            f_net=[i.state_dict() for i in self.f_network]
        else:
            h_net=[]
            f_net=[]
        if 1:
            checkpoint = {
                "net_dqn": self.q_network.state_dict(),
                "net_f":f_net,
                "net_h":h_net,
                # 'optimizer': self.optimizer.state_dict(),
                "step": self.train_step,
            }

        if not os.path.isdir(self.path):
                os.mkdir(self.path)
            # print('the path is {}'.format('logs/dqn_model/duel_dqn_%s.pkl'%(str(self.train_step))))
        torch.save(checkpoint, 'logs/test/cnn_dqn_model/dqn_fh_{}_{}_{}_{}.pkl'.format(self.learning_rate,self.option_dim,str(self.train_step),trial))
            # record training process (stacked before)


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
