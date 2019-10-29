import torch
import numpy as np
import gym
from rl_implementations.trpo.trpo_agent import TRPOAgent
from rl_implementations.trpo.utils import evaluate
import pdb

n_train_steps_per_epoch = 1000
n_epochs = 100
n_eval_every = 1
env_name = 'Pendulum-v0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gamma = 0.99

if __name__=='__main__':

    env = gym.make(env_name)
    agent = TRPOAgent(env.observation_space.shape[0], env.action_space.shape[0])

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
                action = agent.get_action(state)
                new_obs, rew, done, info = env.step(np.tanh(action))
                steps += 1

                states.append(new_obs)
                if not done:
                    states2.append(new_obs)
                actions.append(action)
                rewards.append(rew)
                dones.append(done)

        trajectories = (states, states2, actions, rewards, dones)
        agent.optimize_model(trajectories)

        if ep % n_eval_every == 0:
            eval_rew = evaluate(agent, env)
            print("Epoch: {}, Eval Reward: {}".format(ep, eval_rew))







