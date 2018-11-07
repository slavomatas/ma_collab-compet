import os
import imageio
import numpy as np
import torch

from agent import MADDPG
from buffer import ReplayBuffer
from utils import transpose_list, transpose_to_tensor
from unityagents import UnityEnvironment

BUFFER_SIZE = 100000  # replay buffer size
BATCH_SIZE = 1000 # minibatch size

def seeding(seed=1):
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


def maddpg():
    seeding()

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    print("GPU available: {}".format(torch.cuda.is_available()))
    print("GPU tensor test: {}".format(torch.rand(3, 3).cuda()))

    env = UnityEnvironment(file_name='/home/slavo/Dev/deep-rl-projects/ma_collab-compet/Tennis_Linux/Tennis.x86_64', no_graphics=True)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    state = env_info.vector_observations[0]  # get the current state

    agents = len(env_info.agents)

    # number of training episodes.
    # change this to higher number to experiment. say 30000.
    number_of_episodes = 30000
    episode_length = 80

    # how many episodes to save policy
    save_interval = 1000
    t = 0

    # amplitude of OU noise
    # this slowly decreases to 0
    noise = 2
    noise_reduction = 0.9999

    # how many episodes before update
    episode_per_update = 2

    log_path = os.getcwd() + "/log"
    model_dir = os.getcwd() + "/model_dir"

    os.makedirs(model_dir, exist_ok=True)

    torch.set_num_threads(4)

    # keep 5000 episodes worth of replay
    buffer = ReplayBuffer(int(5000 * episode_length))

    # initialize policy and critic
    maddpg = MADDPG(state_size, action_size, agents)

    agent0_reward = []
    agent1_reward = []

    for episode in range(0, number_of_episodes):

        reward_this_episode = np.zeros(agents)

        env_info = env.reset(train_mode=True)[brain_name]
        obs = env_info.vector_observations
        obs_full = torch.cat(list(torch.tensor(env_info.vector_observations)))

        # for calculating rewards for this particular episode - addition of all time steps
        tmax = 0

        for episode_t in range(episode_length):

            t += 1

            # explore = only explore for a certain number of episodes

            # action input needs to be transposed
            actions = maddpg.act(transpose_to_tensor(list(obs)), noise=noise)
            noise *= noise_reduction

            actions = torch.stack(actions).detach().cpu().numpy()

            # transpose the list of list
            # flip the first two indices
            # input to step requires the first index to correspond to number of parallel agents
            #actions_for_env = np.rollaxis(actions_array, 1)

            # step forward one frame
            #env_info = env.step(np.concatenate(actions))[brain_name]
            env_info = env.step(actions)[brain_name]

            #next_obs, next_obs_full, rewards, dones, info = env.step(actions_for_env)

            # add data to buffer
            transition = (obs, obs_full, actions_for_env, rewards, next_obs, next_obs_full, dones)

            buffer.push(transition)

            reward_this_episode += rewards

            obs, obs_full = next_obs, next_obs_full

        # update once after every episode_per_update
        if len(buffer) > BATCH_SIZE and episode % episode_per_update == 0:
            for agent_idx in range(3):
                samples = buffer.sample(BATCH_SIZE)
                maddpg.update(samples, agent_idx)
            maddpg.update_targets()  # soft update the target network towards the actual networks

        agent0_reward.append(reward_this_episode[0])
        agent1_reward.append(reward_this_episode[1])

        if episode % 100 == 0 or episode == number_of_episodes - 1:
            avg_rewards = [np.mean(agent0_reward), np.mean(agent1_reward)]
            agent0_reward = []
            agent1_reward = []
            for agent_idx, avg_rew in enumerate(avg_rewards):
                print('agent%i/mean_episode_rewards' % agent_idx, avg_rew, episode)

        '''
        # saving model
        save_dict_list = []
        if save_info:
            for i in range(3):
                save_dict = {'actor_params': maddpg.maddpg_agent[i].actor.state_dict(),
                             'actor_optim_params': maddpg.maddpg_agent[i].actor_optimizer.state_dict(),
                             'critic_params': maddpg.maddpg_agent[i].critic.state_dict(),
                             'critic_optim_params': maddpg.maddpg_agent[i].critic_optimizer.state_dict()}
                save_dict_list.append(save_dict)

                torch.save(save_dict_list,
                           os.path.join(model_dir, 'episode-{}.pt'.format(episode)))
        '''

    env.close()

if __name__ == '__main__':
    maddpg()
