import torch
from model import Actor, Critic
from noise import OUNoise
from torch.nn import MSELoss
from torch.optim import Adam
from utils import hard_update, soft_update, transpose_to_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPGAgent:
    def __init__(self, state_size, action_size, agents, lr_actor=1.0e-2, lr_critic=1.0e-2):
        super(DDPGAgent, self).__init__()

        self.actor = Actor(state_size, action_size).to(device)
        self.critic = Critic(state_size, action_size, agents).to(device)
        self.target_actor = Actor(state_size, action_size).to(device)
        self.target_critic = Critic(state_size, action_size, agents).to(device)

        self.noise = OUNoise(action_size, scale=1.0)

        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=1.e-5)

    def act(self, state, noise=0.0):
        state = state.to(device)
        action = self.actor(state) + (noise * self.noise.noise()).to(device)
        return action

    def target_act(self, state, noise=0.0):
        state = state.to(device)
        action = self.target_actor(state) + (noise * self.noise.noise()).to(device)
        return action


class MADDPG:
    def __init__(self, state_size, action_size, agents, discount_factor=0.95, tau=0.02):
        super(MADDPG, self).__init__()
        self.action_size = action_size
        self.agents = agents
        self.maddpg_agent = [DDPGAgent(state_size=state_size, action_size=action_size, agents=agents) for _ in
                             range(agents)]
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = []

        for i in range(2):
            target_actions.append(self.maddpg_agent[i].target_act(obs_all_agents[:, i, :], noise))

        return target_actions

    '''
    def update(self, buffer):
        # do not train until exploration is enough
        if self.episode_done <= self.episodes_before_train:
            return None, None

        ByteTensor = th.cuda.ByteTensor if self.use_cuda else th.ByteTensor
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor

        c_loss = []
        a_loss = []
        for agent in range(self.n_agents):

            samples = buffer.sample(BATCH_SIZE)

            transitions = self.memory.sample(self.batch_size)
            batch = Experience(*zip(*transitions))

            non_final_mask = ByteTensor(list(map(lambda s: s is not None,
                                                 batch.next_states)))
            # state_batch: batch_size x n_agents x dim_obs
            state_batch = th.stack(batch.states).type(FloatTensor)
            action_batch = th.stack(batch.actions).type(FloatTensor)
            reward_batch = th.stack(batch.rewards).type(FloatTensor)
            # : (batch_size_non_final) x n_agents x dim_obs
            non_final_next_states = th.stack(
                [s for s in batch.next_states
                 if s is not None]).type(FloatTensor)

            # for current agent
            whole_state = state_batch.view(self.batch_size, -1)
            whole_action = action_batch.view(self.batch_size, -1)
            self.critic_optimizer[agent].zero_grad()
            current_Q = self.critics[agent](whole_state, whole_action)

            non_final_next_actions = [
                self.actors_target[i](non_final_next_states[:,
                                      i,
                                      :]) for i in range(
                    self.n_agents)]
            non_final_next_actions = th.stack(non_final_next_actions)
            non_final_next_actions = (
                non_final_next_actions.transpose(0,
                                                 1).contiguous())

            target_Q = th.zeros(
                self.batch_size).type(FloatTensor)

            target_Q[non_final_mask] = self.critics_target[agent](
                non_final_next_states.view(-1, self.n_agents * self.n_states),
                non_final_next_actions.view(-1,
                                            self.n_agents * self.n_actions)
            ).squeeze()
            # scale_reward: to scale reward in Q functions

            target_Q = (target_Q.unsqueeze(1) * self.GAMMA) + (
                    reward_batch[:, agent].unsqueeze(1) * scale_reward)

            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
            loss_Q.backward()
            self.critic_optimizer[agent].step()

            self.actor_optimizer[agent].zero_grad()
            state_i = state_batch[:, agent, :]
            action_i = self.actors[agent](state_i)
            ac = action_batch.clone()
            ac[:, agent, :] = action_i
            whole_action = ac.view(self.batch_size, -1)
            actor_loss = -self.critics[agent](whole_state, whole_action)
            actor_loss = actor_loss.mean()
            actor_loss.backward()
            self.actor_optimizer[agent].step()
            c_loss.append(loss_Q)
            a_loss.append(actor_loss)

        if self.steps_done % 100 == 0 and self.steps_done > 0:
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)

        return c_loss, a_loss
    '''

    def update(self, samples, agent_number):
        """update the critics and actors of all the agents """
        print('update the critics and actors of all the agents ')

        obs = []
        obs_full = []
        action = []
        reward = []
        next_obs = []
        next_obs_full = []
        done = []

        for sample in samples:
            o, o_f, a, r, n_o, n_o_f, d = transpose_to_tensor(sample)

            obs.append(o)
            obs_full.append(o_f)
            action.append(a)
            reward.append(r)
            next_obs.append(n_o)
            next_obs_full.append(n_o_f)
            done.append(d)

        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        # critic loss = batch mean of (y- Q(s,a) from target network)^2
        # y = reward of this timestep + discount * Q(st+1,at+1) from target network
        next_obs = torch.stack(next_obs)
        target_actions = self.target_act(next_obs)
        target_actions = torch.cat(target_actions, dim=1)

        next_obs_full = torch.stack(next_obs_full)
        with torch.no_grad():
            q_next = agent.target_critic(next_obs_full, target_actions)

        target_Q = torch.stack(reward)[:, agent_number].view(-1, 1) + self.discount_factor * q_next * (
                    1 - torch.stack(done)[:, agent_number].view(-1, 1))

        obs_full = torch.stack(obs_full)
        action = torch.stack(action).view(-1, self.action_size * self.agents)
        current_Q = agent.critic(obs_full, action)

        #huber_loss = torch.nn.SmoothL1Loss()
        #critic_loss = huber_loss(curent_q, target_q.detach())
        critic_loss = MSELoss()(current_Q, target_Q.detach())
        critic_loss.backward()

        #torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()

        # update actor network using policy gradient
        agent.actor_optimizer.zero_grad()

        # make input to agent
        # detach the other agents to save computation, saves some time for computing derivative
        '''
        q_input = [self.maddpg_agent[i].actor(ob) if i == agent_number \
                       else self.maddpg_agent[i].actor(ob).detach()
                   for i, ob in enumerate(obs)]
        '''
        q_input = []
        for i in range(self.agents):
            if i == agent_number:
                q_input.append(self.maddpg_agent[i].actor(torch.stack(obs)[:, i, :]))
            else:
                q_input.append(self.maddpg_agent[i].actor(torch.stack(obs)[:, i, :]).detach())

        q_input = torch.cat(q_input, dim=1)

        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already

        # get the policy gradient
        actor_loss = -agent.critic(obs_full, q_input).mean()
        actor_loss.backward()

        # torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
