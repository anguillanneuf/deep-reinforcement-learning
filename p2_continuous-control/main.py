import argparse
from collections import deque
import gym
import numpy as np
import pickle
import random
import time
import torch
from unityagents import UnityEnvironment

from ddpg_agent import Agent

def main():
    env = UnityEnvironment(file_name='Reacher.app')
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]

    agent = Agent(state_size=state_size, action_size=action_size, random_seed=3)

    scores_deque = deque(maxlen=100)
    scores = []

    for i_episode in range(1, 1000):
        begin = time.time()
        curr_scores = np.zeros(num_agents)                 # initialize the score (for each agent)
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations              # get the current state (for each agent)

        agent.reset()

        for t in range(1000):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]        # send all actions to the environment
            next_states = env_info.vector_observations      # get next state (for each agent)
            rewards = env_info.rewards                      # get reward (for each agent)
            dones = env_info.local_done                     # see if episode finished

            agent.step(states, actions, rewards, next_states, dones, t)

            states = next_states
            curr_scores += rewards

            if np.any(dones):
                break

        curr_score = np.mean(curr_scores)
        scores_deque.append(curr_score)
        average_score = np.mean(scores_deque)
        scores.append(curr_score)

        print('\rEpisode {}\tTime: {:.2f}\tAvg: {:.2f}\tScore: {:.2f}\tMin {:.2f}\tMax {:.2f}'.format(
            i_episode, time.time()-begin, average_score, curr_score,
            min(curr_scores), max(curr_scores)))
        if i_episode % 10 == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        if average_score >= 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                i_episode-100, average_score))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break

    env.close()

    return

if __name__ == '__main__':
    _ = main()
