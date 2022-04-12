import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dqn_network import DQN_network, DQN_target_network
from .dqn_feature_constructor import FeatureConstructor
from .replay_buffer import ReplayMemory
from config.hex_setting import LEARNING_RATE, GAMMA, REPLAY_BUFFER_SIZE, BATCH_SIZE, RELOCATION_DIM, CHARGING_DIM, \
    INPUT_DIM, OUTPUT_DIM, FINAL_EPSILON, HIGH_SOC_THRESHOLD, LOW_SOC_THRESHOLD, CLIPPING_VALUE, START_EPSILON, \
    EPSILON_DECAY_STEPS, MODEL_SAVE_PATH, SAVING_CYCLE, DQN_RESUME
import os


class DeepQNetworkAgent:
    def __init__(self):
        """
        todo: update epsilon value later, for now keep it as 0.5
        """
        self.learning_rate = LEARNING_RATE  # 1e-5
        self.gamma = GAMMA
        self.start_epsilon = START_EPSILON
        self.final_epsilon = FINAL_EPSILON
        self.epsilon_steps = EPSILON_DECAY_STEPS
        self.memory = ReplayMemory(REPLAY_BUFFER_SIZE)
        self.batch_size = BATCH_SIZE
        self.clipping_value = CLIPPING_VALUE
        self.input_dim = INPUT_DIM
        self.relocation_dim = RELOCATION_DIM
        self.charging_dim = CHARGING_DIM
        self.output_dim = OUTPUT_DIM
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.path = MODEL_SAVE_PATH
        self.state_feature_constructor = FeatureConstructor()
        self.q_network = DQN_network(self.input_dim, self.output_dim)
        self.target_q_network = DQN_target_network(self.input_dim, self.output_dim)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.train_step = 0

        self.load_network()
        self.q_network.to(self.device)
        self.target_q_network.to(self.device)
        self.decayed_epsilon = self.start_epsilon
        self.record_list = []

    def load_network(self):
        # lists = os.listdir(self.path)
        # lists.sort(key=lambda fn: os.path.getmtime(self.path + "/" + fn))
        # newest_file = os.path.join(self.path, lists[-1])
        if DQN_RESUME:
            path_checkpoint = 'logs/dqn_model/duel_dqn_357840.pkl'
            checkpoint = torch.load(path_checkpoint)

            self.q_network.load_state_dict(checkpoint['net'])

            self.copy_parameter()

            # self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.train_step = checkpoint['step']
            print('Successfully load saved network from path {}, the training step starts from {}!',
                  format(path_checkpoint, str(self.train_step)))

    def get_actions(self, states, num_valid_relos):
        """
        :param states: tuple of (tick, hex_id, SOC) and SOC is 0 - 100%
        :param num_valid_relos: only relocation to ADJACENT hexes / charging station is valid
        :states:
        :return:
        """

        with torch.no_grad():
            self.decayed_epsilon = max(self.final_epsilon, (self.start_epsilon - self.train_step * (
                    self.start_epsilon - self.final_epsilon) / self.epsilon_steps))
            if random.random() > self.decayed_epsilon:  # epsilon = 0.1
                state_reps = [self.state_feature_constructor.construct_state_features(state) for state in states]
                full_action_values = self.q_network.forward(
                    torch.from_numpy(np.array(state_reps)).to(dtype=torch.float32, device=self.device))
                # relocation + charging dimension

                mask = np.zeros([len(states), self.output_dim])

                for i, state in enumerate(states):
                    mask[i][num_valid_relos[i]:self.relocation_dim] = 1
                    # here the SOC in state is still continuous. the categorized one is in state reps.
                    if state[-1] > HIGH_SOC_THRESHOLD:
                        mask[i][self.relocation_dim:] = 1  # no charging, must relocate
                    elif state[-1] < LOW_SOC_THRESHOLD:
                        mask[i][:self.relocation_dim] = 1  # no relocation, must charge

                mask = torch.from_numpy(mask).to(dtype=torch.bool, device=self.device)
                # print('take a look at processed mask {}'.format(mask))
                full_action_values[mask] = -9e10

                action_indexes = torch.argmax(full_action_values, dim=1).tolist()
            else:
                full_action_values = np.random.random(
                    (len(states), self.output_dim))  # generate a matrix with values from 0 to 1
                for i, state in enumerate(states):
                    full_action_values[i][num_valid_relos[i]:self.relocation_dim] = -1
                    if state[-1] > HIGH_SOC_THRESHOLD:
                        full_action_values[i][self.relocation_dim:] = -1  # no charging, must relocate
                    elif state[-1] < LOW_SOC_THRESHOLD:
                        full_action_values[i][:self.relocation_dim] = -1  # no relocation, must charge

                action_indexes = np.argmax(full_action_values, 1).tolist()

        return action_indexes

    def add_transition(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

    def batch_sample(self):
        samples = self.memory.sample(self.batch_size)  # random.sample(self.memory, self.batch_size)
        return samples
        # state, action, next_state, reward = zip(*samples)
        # return state, action, next_state, reward

    def get_main_Q(self, state):
        return self.q_network.forward(state)

    def get_target_Q(self, state):
        return self.target_q_network.forward(state)

    def copy_parameter(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def train(self, record_hist):
        self.train_step += 1
        if len(self.memory) < self.batch_size:
            print('batches in replay buffer is {}'.format(len(self.memory)))
            return

        transitions = self.batch_sample()
        batch = self.memory.Transition(*zip(*transitions))
        # print('batches are:{}'.format(batch.state))
        # state_batch = torch.cat(list(batch.state))
        # action_batch = torch.stack(batch.action)
        # reward_batch = torch.stack(batch.reward)

        state_reps = [self.state_feature_constructor.construct_state_features(state) for state in batch.state]

        next_state_reps = [self.state_feature_constructor.construct_state_features(state_) for state_ in
                           batch.next_state]

        state_batch = torch.from_numpy(np.array(state_reps)).to(dtype=torch.float32, device=self.device)
        action_batch = torch.from_numpy(np.array(batch.action)).unsqueeze(1).to(dtype=torch.int64, device=self.device)
        reward_batch = torch.from_numpy(np.array(batch.reward)).unsqueeze(1).to(dtype=torch.float32, device=self.device)

        # non_terminal_indices = torch.tensor(
        #     tuple(map(lambda s: s is not 1, batch.flag)),
        #     device=self.device,
        #     dtype=torch.bool,
        # )
        next_state_batch = torch.from_numpy(np.array(next_state_reps)).to(device=self.device, dtype=torch.float32)

        q_state_action = self.get_main_Q(state_batch).gather(1, action_batch.long())
        maxq = self.get_target_Q(next_state_batch).max(1)[0].detach()
        # additional_qs = torch.zeros(self.batch_size, device=self.device)
        # additional_qs[non_terminal_indices] = maxq
        # print('SHAPE of REWARD: {}, MAXQ: {}, Q(s,a):{} '.format(reward_batch.shape,maxq.unsqueeze(1).shape,q_state_action.shape))
        y = reward_batch + self.gamma * maxq.unsqueeze(1)
        loss = F.smooth_l1_loss(q_state_action, y)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.clipping_value)
        self.optimizer.step()
        self.record_list.append([self.train_step, loss, reward_batch.view(-1).mean()])
        self.save_parameter(record_hist)

    def save_parameter(self, record_hist):
        # torch.save(self.q_network.state_dict(), self.dqn_path)
        if self.train_step % SAVING_CYCLE == 0:
            print('step:', self.train_step)
            # print('learning rate:', self.optimizer.state_dict()['param_groups'][0]['lr'])
            print('decayed epsilon', self.decayed_epsilon)
            checkpoint = {
                "net": self.q_network.state_dict(),
                # 'optimizer': self.optimizer.state_dict(),
                "step": self.train_step
            }
            if not os.path.isdir(self.path):
                os.mkdir(self.path)
            # print('the path is {}'.format('logs/dqn_model/duel_dqn_%s.pkl'%(str(self.train_step))))
            torch.save(checkpoint, 'logs/dqn_model/duel_dqn_%s.pkl' % (str(self.train_step)))
            # record training process
            for item in self.record_list:
                record_hist.writelines('{},{},{}\n'.format(item[0], item[1], item[2]))
            self.record_list = []
