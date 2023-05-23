
# DQN的实现参考了以下内容：
# https://mofanpy.com/tutorials/machine-learning/torch/DQN

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_shape, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, action_shape)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class DQN:
    def __init__(self, state_shape, action_shape, lr, gamma, epsilon, memory_capacity, batch_size, target_replace_iter):
        # Train two networks: eval_net and target_net
        self.eval_net, self.target_net = Net(state_shape, action_shape), Net(state_shape, action_shape)
        self.n_actions = action_shape
        self.n_states = state_shape
        # target updating
        self.target_replace_iter = target_replace_iter
        self.learn_step_counter = 0
        # memory storing
        self.memory_capacity = memory_capacity
        self.memory_counter = 0
        # initialize memory pool
        self.memory = np.zeros((memory_capacity, state_shape * 2 + 2))  # state(n)+action(1)+reward(1)+next_state(n)
        # other parameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        # optimizer and loss function
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < self.epsilon:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]  # return the argmax index
        else:   # random
            action = np.random.randint(0, self.n_actions)
            action = action
        return action

    def store_transition(self, s, a, r, s_):
        # store the transition in memory pool, called after env.step()
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample transitions (size: batch_size) from memory pool
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.n_states])
        b_a = torch.LongTensor(b_memory[:, self.n_states:self.n_states+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.n_states+1:self.n_states+2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_states:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, no backpropagation
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
