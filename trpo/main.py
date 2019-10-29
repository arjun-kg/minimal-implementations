import torch
import numpy as np
import gym
import os
from rl_implementations.trpo.trpo_agent import TRPOAgent
from rl_implementations.trpo.utils import evaluate, save_model
from tensorboardX import SummaryWriter
from datetime import datetime
import pdb

n_train_steps_per_epoch = 1000
n_epochs = 1000
n_eval_every = 1
env_name = 'Pendulum-v0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gamma = 0.99
save_freq = 100
dir_name = '/tmp/rl_implementations/trpo/{}-{}'.format(env_name, datetime.today().strftime("%Y-%d-%b-%H-%M-%S"))


if __name__=='__main__':

    env = gym.make(env_name)
    agent = TRPOAgent(env.observation_space.shape[0], env.action_space.shape[0])
    max_eval_rew = -1e8
    writer = SummaryWriter(dir_name)

    for ep in range(n_epochs):
        steps = 0
        dones = []
        states = []
        states2 = []
        actions = []
        rewards = []

        n_episodes = 0
        while steps < n_train_steps_per_epoch:
            n_episodes += 1
            state = env.reset()
            done = False

            states.append(state)
            states2.append(state)

            while not done and steps < n_train_steps_per_epoch:
                steps += 1

                action = agent.get_action(state)
                new_obs, rew, done, info = env.step(np.clip(action, -1, 1))
                states.append(new_obs)
                if not done:
                    states2.append(new_obs)
                actions.append(action)
                rewards.append(rew)
                dones.append(done)

        trajectories = (states, states2, actions, rewards, dones)
        agent.optimize_model(trajectories, writer)

        if ep % save_freq == 0:
            if not os.path.exists(dir_name + '/interval_models'):
                os.makedirs(dir_name + '/interval_models')

            save_path_temp = dir_name + '/interval_models/model_{}'.format(ep)
            save_model(agent, save_path_temp)
            print("Model saved after {}th epoch".format(ep))

        if ep % n_eval_every == 0:
            eval_rew = evaluate(agent, env)
            print("Epoch: {}, Eval Reward: {}".format(ep, eval_rew))

            if eval_rew > max_eval_rew:
                max_eval_rew = eval_rew
                save_model(agent, dir_name + '/model_best')
                print("Best model saved with reward of {}".format(eval_rew))








