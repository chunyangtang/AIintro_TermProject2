
from models.dqn import DQN
from models.pets import PETS, ReplayBuffer
import gymnasium as gym
import numpy as np
import torch
import tianshou as ts
from tianshou.utils.net.common import Net


def main_dqn():
    dqn_params = {
        'lr': 0.01,                     # learning rate of the optimizer
        'gamma': 0.9,                   # reward discount factor
        'epsilon': 0.9,                 # epsilon greedy policy
        'memory_capacity': 200,        # capacity of replay buffer
        'batch_size': 32,               # batch size for each sampling
        'target_replace_iter': 100,     # target network update frequency
        'train_iter': 100,              # number of training iterations
    }

    env = gym.make('CartPole-v1')
    state_shape = env.observation_space.shape[0] or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n

    dqn = DQN(state_shape, action_shape, dqn_params['lr'], dqn_params['gamma'],
              dqn_params['epsilon'], dqn_params['memory_capacity'], dqn_params['batch_size'],
              dqn_params['target_replace_iter'])

    x_threshold = env.x_threshold
    theta_threshold_radians = env.theta_threshold_radians

    for i_episode in range(dqn_params['train_iter']):
        obs, info = env.reset()
        episode_reward = 0
        while True:
            # choose action based on the current state
            action = dqn.choose_action(obs)
            # take action and get next state and reward
            next_obs, reward, terminated, truncated, info = env.step(action)
            # accumulate reward
            episode_reward += reward
            # modify reward for better training
            x, x_dot, theta, theta_dot = next_obs
            r1 = (x_threshold - abs(x)) / x_threshold - 0.8
            r2 = (theta_threshold_radians - abs(theta)) / theta_threshold_radians - 0.5
            reward = r1 + r2

            # store the transition in memory pool
            dqn.store_transition(obs, action, reward, next_obs)
            # update current state
            obs = next_obs
            # learn from the memory pool
            if dqn.memory_counter > dqn.memory_capacity:
                dqn.learn()

            if terminated or truncated:
                if i_episode % 10 == 0:
                    print('Episode %d: Finished with the reward of %d.' % (i_episode, episode_reward))
                break


def main_pets():
    pets_params = {
        'memory_capacity': 2000,        # capacity of replay buffer
        'sequence_length': 5,           # length of the sequence to be planned
        'num_network': 5,               # ensemble size
        'lr': 0.0003,                     # learning rate of the optimizer
        'train_iter': 100,              # number of training iterations
    }

    env = gym.make('CartPole-v1')
    replay_buffer = ReplayBuffer(pets_params["memory_capacity"])
    pets = PETS(env, replay_buffer, pets_params['sequence_length'], pets_params['num_network'], pets_params['lr'])

    x_threshold = env.x_threshold
    theta_threshold_radians = env.theta_threshold_radians

    for i_episode in range(pets_params['train_iter']):
        obs, info = pets.env.reset()
        episode_reward = 0
        while True:
            # choose action based on the current state
            action = pets.optimize(obs)
            # take action and get next state and reward
            next_obs, reward, terminated, truncated, info = env.step(action)
            # accumulate reward
            episode_reward += reward
            # modify reward for better training
            x, x_dot, theta, theta_dot = next_obs
            r1 = (x_threshold - abs(x)) / x_threshold - 0.8
            r2 = (theta_threshold_radians - abs(theta)) / theta_threshold_radians - 0.5
            reward = r1 + r2
            # change action to one-hot vector for estimated environment training
            action_one_hot = np.zeros(pets.env.action_space.n)
            action_one_hot[action] = 1
            # store the transition in memory pool
            pets.env_pool.add(obs, action_one_hot, reward, next_obs, terminated)
            # update current state
            obs = next_obs

            if terminated or truncated:
                if i_episode % 10 == 0:
                    print('Episode %d: Finished with the reward of %d.' % (i_episode, episode_reward))
                break

        pets.train_model()


def main_tianshou_dqn():
    task = 'CartPole-v1'
    lr, epoch, batch_size = 1e-3, 10, 64
    train_num, test_num = 10, 100
    gamma, n_step, target_freq = 0.9, 3, 320
    buffer_size = 20000
    eps_train, eps_test = 0.1, 0.05
    step_per_epoch, step_per_collect = 10000, 10

    train_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(train_num)])
    test_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(test_num)])

    env = gym.make(task)
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n

    net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[128, 128, 128])
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    policy = ts.policy.DQNPolicy(net, optim, gamma, n_step, target_update_freq=target_freq)
    train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(buffer_size, train_num),
                                        exploration_noise=True)
    test_collector = ts.data.Collector(policy, test_envs,
                                       exploration_noise=True)  # because DQN uses epsilon-greedy method

    result = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector, epoch, step_per_epoch, step_per_collect,
        test_num, batch_size, update_per_step=1 / step_per_collect,
        train_fn=lambda epoch, env_step: policy.set_eps(eps_train),
        test_fn=lambda epoch, env_step: policy.set_eps(eps_test),
        stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold)
    print(f'Finished training! Use {result["duration"]}')


if __name__ == '__main__':
    print("Hello World!")
    print("DQN Training...")
    main_dqn()
    print("PETS Training...")
    # main_pets()
    print("Tianshou DQN Training...")
    # main_tianshou_dqn()
