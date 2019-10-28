import torch
import gym
from rl_implementations.trpo.trpo_agent import TRPOAgent
import pdb

n_train_steps_per_epoch = 1000
n_epochs = 100
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
        actions = []
        log_probs = []
        rewards = []

        n_episodes = 0
        while steps < n_train_steps_per_epoch:
            n_episodes += 1
            state = env.reset()
            done = False

            states.append(state)

            while not done and steps < n_train_steps_per_epoch:
                action, log_prob = agent.get_action(state)
                new_obs, rew, done, info = env.step(action)
                steps += 1

                states.append(new_obs)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(rew)
                dones.append(done)

        trajectories = (states, actions, log_probs, rewards, dones)
        agent.optimize_model(trajectories)







