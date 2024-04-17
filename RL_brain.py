"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, total_episode=100, is_resume=False):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        # in this case, Frame vertical axis is the state(eg. 0,1,2,3...); horizon axis is the actions
        # in this case, actions are up, down, left, right
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.total_episode = total_episode
        self.is_resume = is_resume
        if self.is_resume:
            self.load_pd_dataframe()

    def choose_action(self, observation, episode):
        self.check_state_exist(observation)
        if episode > 0.7 * self.total_episode:
            self.epsilon = 1
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return np.int64(action)

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
        # if not done:
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
    # Given the current state, check if the state been reached, if not, it should be appended to q table
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def load_pd_dataframe(self):
        self.q_table = pd.read_json('my_df.json')


np.random.seed(1)
torch.manual_seed(1)


# define the network architecture
class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_feature, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class DeepQNetwork:
    def __init__(self, n_actions, n_features, n_hidden=128, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,
                 replace_target_iter=200, memory_size=500, batch_size=32, e_greedy_increment=None, total_episode=300,
                 is_resume=False
                 ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.total_episode = total_episode

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self.loss_func = nn.MSELoss()
        # self.loss_func = F.smooth_l1_loss()
        self.cost_his = []
        self.is_resume = is_resume
        self._build_net()

    def _build_net(self):
        self.q_eval = Net(self.n_features, self.n_hidden, self.n_actions)
        self.q_target = Net(self.n_features, self.n_hidden, self.n_actions)
        self.optimizer = torch.optim.RMSprop(self.q_eval.parameters(), lr=self.lr)
        if self.is_resume:
            self.q_eval.load_state_dict(torch.load("maze_dqn_299.pth"))

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation, episode):
        observation = torch.Tensor(observation[np.newaxis, :])
        if episode > 0.7 * self.total_episode:
            self.epsilon = 1
        if np.random.uniform() < self.epsilon:
            actions_value = self.q_eval(observation)

            action = np.argmax(actions_value.data.numpy())
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self, episode):
        # check to replace target parameters
        if episode % 5 == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())
            print("\ntarget params replaced\n")

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # q_next is used for getting which action would be chosen by target network in state s_(t+1)
        q_next, q_eval = self.q_target(torch.Tensor(batch_memory[:, -self.n_features:])), self.q_eval(
            torch.Tensor(batch_memory[:, :self.n_features]))
        # used for calculating y, we need to copy for q_eval because this operation could keep the Q_value that has not been selected unchanged,
        # so when we do q_target - q_eval, these Q_value become zero and wouldn't affect the calculation of the loss
        q_target = torch.Tensor(q_eval.data.numpy().copy())

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = torch.Tensor(batch_memory[:, self.n_features + 1])
        q_target[batch_index, eval_act_index] = reward + self.gamma * torch.max(q_next, 1)[0]

        # loss = self.loss_func(q_eval, q_target)
        # loss = F.smooth_l1_loss(q_eval, q_target.unsqueeze(1))
        loss = F.smooth_l1_loss(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # increase epsilon
        self.cost_his.append(loss.detach().numpy())
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()






