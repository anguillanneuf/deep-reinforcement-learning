import numpy as np
import torch

from ddpg import DDPGAgent
from utilities import soft_update, transpose_to_tensor, transpose_list


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MADDPG:
    """policy + critic updates"""

    def __init__(self, discount_factor=0.95, tau=0.02):
        super(MADDPG, self).__init__()

        # critic input = obs_full + actions = 24 + 24 + 2 + 2
        # in_actor=24, hidden_in_actor=16, hidden_out_actor=8, out_actor=2,
        # in_critic=52, hidden_in_critic=32, hidden_out_critic=16,
        self.maddpg_agent = [DDPGAgent(24, 256, 128, 2, 52, 256, 128),
                             DDPGAgent(24, 256, 128, 2, 52, 256, 128)]

        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent
                         in self.maddpg_agent]
        return target_actors

    def act(self, states, noise=0.0):
        """get actions from all agents in the MADDPG object
        Args:
            states: Array(2,24)
        Returns:
            actions: Array(2,2)
        """
        actions = []

        for agent, state in zip(self.maddpg_agent, states):
            state = torch.tensor(state).float().to(device)
            action = agent.act(state, noise)
            actions.append(action)

        actions = np.vstack(actions[i].detach().numpy() for i in range(2))
        return actions

    def target_act(self, states, noise=0.0):
        """get target network actions from all agents in the MADDPG object
        Args:
            states: List(Tensor(batchsize,24), Tensor(batchsize,24))
        Returns:
            target_actions: List(Tensor(batchsize,2), Tensor(batchsize,2))
        """
        target_actions = []

        for agent, state in zip(self.maddpg_agent, states):
            temp = agent.target_act(state, noise)
            target_actions.append(temp)
        return target_actions

    def update(self, samples, agent_number, logger):
        """update the critics and actors of all the agents"""

        # `samples`: a list of batchsize, List(5,)
        # `states` & next_states: a list of batchsize, Array(2,24)
        # `actions`: a list of batchsize, Array(2,2)
        # `rewards` & `dones`: a list of batch size, List(2,)
        states, actions, rewards, next_states, dones = zip(*samples)

        # -------------------------- preprocessing -------------------------- #
        # `states` & `next_states`: a list of size 2, Tensor(batchsize,24)
        states = transpose_to_tensor(states)
        next_states = transpose_to_tensor(next_states)

        # `states_full` & `next_states_full`: Tensor(batchsize,48)
        states_full = torch.cat(states, dim=1)
        next_states_full = torch.cat(next_states, dim=1)

        # `actions`: Tensor(batchsize,4)
        actions = transpose_to_tensor(actions)
        actions = torch.cat(actions, dim=1)

        # `dones` & `rewards`: a list of 2, Tensor(batchsize,)
        dones = transpose_to_tensor(transpose_list(zip(*dones)))
        rewards = transpose_to_tensor(transpose_list(zip(*rewards)))

        # -------------------------- update critic -------------------------- #
        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        # critic loss = batch mean of (y - Q(s,a) from target network)^2
        # y = current reward + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act(next_states)
        target_actions = torch.cat(target_actions, dim=-1)
        target_critic_input = torch.cat(
            (next_states_full, target_actions), dim=1).to(device)

        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)

        y = rewards[agent_number].view(-1, 1) + \
            self.discount_factor * q_next * \
            (1 - dones[agent_number].view(-1, 1))
        critic_input = torch.cat((states_full, actions), dim=1).to(device)
        q = agent.critic(critic_input)

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()

        # -------------------------- update actor -------------------------- #
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = [ self.maddpg_agent[i].actor(state) if i == agent_number \
                   else self.maddpg_agent[i].actor(state).detach()
                   for i, state in enumerate(states) ]

        q_input = torch.cat(q_input, dim=1)
        # combine all the actions and observations for input to critic
        q_input2 = torch.cat((states_full, q_input), dim=1)

        # get the policy gradient
        actor_loss = -agent.critic(q_input2).mean()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

        # for TensorBoard
        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        logger.add_scalars('agent%i/losses' % agent_number,
                           {'critic loss': cl,
                            'actor_loss': al},
                           self.iter)

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
