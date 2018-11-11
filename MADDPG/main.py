# main function that sets up environments
# perform training loop

import envs
from buffer import ReplayBuffer
from maddpg import MADDPG
import torch
import numpy as np
from tensorboardX import SummaryWriter
import os
from utilities import transpose_list, transpose_to_tensor

# keep training awake
from workspace_utils import keep_awake

# for saving gif
import imageio

def seeding(seed=1):
    """Set random seed for pytorch and numpy."""
    np.random.seed(seed)
    torch.manual_seed(seed)

def pre_process(entity, batchsize):
    processed_entity = []
    for j in range(3):
        list = []
        for i in range(batchsize):
            b = entity[i][j]
            list.append(b)
        c = torch.Tensor(list)
        processed_entity.append(c)
    return processed_entity


def main():
    seeding()
    parallel_envs = 4
    number_of_episodes = 1000
    episode_length = 80
    batchsize = 1000
    save_interval = 1000
    t = 0

    # amplitude of OU noise, which slowly decreases to 0
    noise = 2
    noise_reduction = 0.9999

    # how many episodes before update
    episode_per_update = 2 * parallel_envs

    log_path = os.getcwd() + "/log"
    model_dir = os.getcwd() + "/model_dir"

    os.makedirs(model_dir, exist_ok=True)

    torch.set_num_threads(parallel_envs)
    """
    `env` controls three agents, two blue, one red.
    env.observation_space: [Box(14,), Box(14,), Box(14,)]
    env.action_sapce: [Box(2,), Box(2,), Box(2,)]
    Box(14,) can be broken down into 2+3*2+3*2=14
    (2) location coordinates of the target landmark
    (3*2) the three agents' positions w.r.t. the target landmark
    (3*2) the three agents' velocities w.r.t. the target landmark
    """
    env = envs.make_parallel_env(parallel_envs)

    # keep 5000 episodes worth of replay
    buffer = ReplayBuffer(int(5000 * episode_length))

    # initialize policy and critic
    maddpg = MADDPG()
    logger = SummaryWriter(log_dir=log_path)
    agent0_reward = []
    agent1_reward = []
    agent2_reward = []

    # training loop
    # show progressbar
    import progressbar as pb
    widget = ['episode: ',
              pb.Counter(), '/', str(number_of_episodes), ' ',
              pb.Percentage(), ' ',
              pb.ETA(), ' ',
              pb.Bar(marker=pb.RotatingMarker()), ' ']

    timer = pb.ProgressBar(widgets=widget, maxval=number_of_episodes).start()

    # use keep_awake to keep workspace from disconnecting
    for episode in keep_awake(range(0, number_of_episodes, parallel_envs)):

        timer.update(episode)

        reward_this_episode = np.zeros((parallel_envs, 3))
        # Consult `env_wrapper.py` line 19.
        all_obs = env.reset()
        """
        `all_abs` is a list of size `parallel_envs`,
        each item in the list is another list of size two,
        first is env.observation_space: [Box(14,), Box(14,), Box(14,)],
        second is [Box(14,)], which is added to faciliate training
        https://goo.gl/Xtr6sF
        `obs` and `obs_full` are both lists of size `parallel_envs`,
        `obs` has the default observation space [Box(14,), Box(14,), Box(14,)]
        `obs_full` has the compounded observation space [Box(14,)]
        """
        obs, obs_full = transpose_list(all_obs)

        # for calculating rewards for one episode - addition of all time steps

        # save info or not
        save_info = ((episode) % save_interval < parallel_envs or
                     episode==number_of_episodes-parallel_envs)
        frames = []
        tmax = 0

        if save_info:
            frames.append(env.render('rgb_array'))

        for episode_t in range(episode_length):

            t += parallel_envs

            # explore = only explore for a certain number of steps
            # action input needs to be transposed
            actions = maddpg.act(transpose_to_tensor(obs), noise=noise)
            noise *= noise_reduction

            # `actions_array` has shape (3, parallel_envs, 2)
            actions_array = torch.stack(actions).detach().numpy()
            # `actions_for_env` has shape (parallel_envs, 3, 2), because
            # input to `step` requires the first index to be `parallel_envs`
            actions_for_env = np.rollaxis(actions_array, axis=1)

            # step forward one frame
            next_obs, next_obs_full, rewards, dones, info = \
                env.step(actions_for_env)

            # add data to buffer
            transition = (obs, obs_full, actions_for_env, rewards,
                          next_obs, next_obs_full, dones)

            buffer.push(transition)

            reward_this_episode += rewards

            obs, obs_full = next_obs, next_obs_full

            # save gif frame
            if save_info:
                frames.append(env.render('rgb_array'))
                tmax+=1

        # update the target network `parallel_envs`=4 times
        # after every `episode_per_update`=2*4
        if len(buffer) > batchsize and episode % episode_per_update < parallel_envs:
            # update the local network for all agents, `a_i` refers to agent no.
            for a_i in range(3):
                samples = buffer.sample(batchsize)
                maddpg.update(samples, a_i, logger)
            # soft update the target network towards the actual networks
            maddpg.update_targets()

        for i in range(parallel_envs):
            agent0_reward.append(reward_this_episode[i,0])
            agent1_reward.append(reward_this_episode[i,1])
            agent2_reward.append(reward_this_episode[i,2])

        if episode % 100 == 0 or episode == number_of_episodes-1:
            avg_rewards = [np.mean(agent0_reward),
                           np.mean(agent1_reward),
                           np.mean(agent2_reward)]
            agent0_reward = []
            agent1_reward = []
            agent2_reward = []
            for a_i, avg_rew in enumerate(avg_rewards):
                logger.add_scalar(
                    'agent%i/mean_episode_rewards' % a_i, avg_rew, episode)

        # Saves the model.
        save_dict_list =[]
        if save_info:
            for i in range(3):
                save_dict = {'actor_params' : maddpg.maddpg_agent[i].actor.state_dict(),
                             'actor_optim_params': maddpg.maddpg_agent[i].actor_optimizer.state_dict(),
                             'critic_params' : maddpg.maddpg_agent[i].critic.state_dict(),
                             'critic_optim_params' : maddpg.maddpg_agent[i].critic_optimizer.state_dict()}
                save_dict_list.append(save_dict)

                torch.save(save_dict_list,
                           os.path.join(model_dir, 'episode-{}.pt'.format(episode)))

            # Save gif files.
            imageio.mimsave(os.path.join(model_dir, 'episode-{}.gif'.format(episode)),
                            frames, duration=.04)

    env.close()
    logger.close()
    timer.finish()

if __name__=='__main__':
    main()
