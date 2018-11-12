from collections import deque
import imageio
import numpy as np
import os
from tensorboardX import SummaryWriter
import torch
from unityagents import UnityEnvironment

from buffer import ReplayBuffer
from maddpg import MADDPG
from utilities import transpose_list, transpose_to_tensor


def seeding(seed=1):
    """Set random seed for pytorch and numpy."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():

    seeding()

    number_of_episodes = 3000
    episode_length     = 1000
    batchsize          = 512
    save_interval      = 1000
    rewards_deque      = deque(maxlen=100)
    rewards_all        = []
    noise              = 2
    noise_reduction    = 0.9999
    episode_per_update = 10

    log_path = os.getcwd() + "/log"
    model_dir = os.getcwd() + "/model_dir"

    os.makedirs(model_dir, exist_ok=True)

    """ Info about the UnityEnvironment
    brain_name: 'TennisBrain'
    brain: ['brain_name', 'camera_resolutions',
           'num_stacked_vector_observations', 'number_visual_observations',
           'vector_action_descriptions', 'vector_action_space_size',
           'vector_action_space_type', 'vector_observation_space_size',
           'vector_observation_space_type']]
    """

    env         = UnityEnvironment(file_name="Tennis.app")
    brain_name  = env.brain_names[0]
    brain       = env.brains[brain_name]

    buffer      = ReplayBuffer(int(1e4))

    # initialize policy and critic
    maddpg = MADDPG()
    logger = SummaryWriter(log_dir=log_path)
    max_scores_deque = deque(maxlen=100)

    # ------------------------------ training ------------------------------ #
    # show progressbar
    import progressbar as pb
    widget = ['episode: ',
              pb.Counter(), '/', str(number_of_episodes), ' ',
              pb.Percentage(), ' ',
              pb.ETA(), ' ',
              pb.Bar(marker=pb.RotatingMarker()), ' ']

    timer = pb.ProgressBar(widgets=widget, maxval=number_of_episodes).start()

    for episode in range(1, number_of_episodes+1):

        timer.update(episode)
        rewards_this_episode = np.zeros((2,))

        """ Info about the UnityEnvironment
        env_info: ['agents', 'local_done', 'max_reached', 'memories',
                  'previous_text_actions', 'previous_vector_actions', 'rewards',
                  'text_observations', 'vector_observations', 'visual_observations']
        actions: List(num_agents=2, action_size=2)
        states: List((24,), (24,))
        rewards: List(2,)
        dones: List(2,)
        """
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations

        for episode_t in range(episode_length):

            actions = maddpg.act(states, noise=noise)
            env_info = env.step(actions)[brain_name]
            noise *= noise_reduction

            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            # add data to buffer
            transition = (states, actions, rewards, next_states, dones)
            buffer.push(transition)

            rewards_this_episode += rewards

            states = next_states

            if any(dones):
                break

        # update the local and target network 20 times every 10 episodes
        if len(buffer) > batchsize and episode % episode_per_update == 0:
            # update the local network
            for _ in range(20):
                for a_i in range(2):
                    samples = buffer.sample(batchsize)
                    maddpg.update(samples, a_i, logger)
            # soft update the target network
            maddpg.update_targets()

        max_scores_deque.append(np.max(rewards_this_episode))
        average_score = np.mean(max_scores_deque)

        # --------------------- Logging for TensorBoard --------------------- #
        for a_i, a_reward in enumerate(rewards_this_episode):
            logger.add_scalar(
                'agent%i/reward' % a_i, a_reward, episode)

        logger.add_scalars('global',
                           {'score': np.max(rewards_this_episode),
                            'average_score': average_score},
                           episode)
        # -------------------------- Save the model -------------------------- #
        save_dict_list =[]

        if episode % save_interval == 0 or average_score >= 0.5:
            for i in range(2):
                save_dict = \
                    {'actor_params' : maddpg.maddpg_agent[i].actor.state_dict(),
                     'actor_optim_params': maddpg.maddpg_agent[i].actor_optimizer.state_dict(),
                     'critic_params' : maddpg.maddpg_agent[i].critic.state_dict(),
                     'critic_optim_params' : maddpg.maddpg_agent[i].critic_optimizer.state_dict()}
                save_dict_list.append(save_dict)

                torch.save(
                    save_dict_list,
                    os.path.join(model_dir, 'episode-{}.pt'.format(episode)))

            if average_score >= 0.5:
                print('\nEnvironment solved in {} episodes!'.format(episode-100))
                print('\nAverage Score: {:.2f}'.format(average_score))
                break

    env.close()
    logger.close()
    timer.finish()

if __name__=='__main__':
    main()
