import os
import numpy as np
import torch

from agent import MADDPG
from buffer import ReplayBuffer
from utils import transpose_list, transpose_to_tensor
from unityagents import UnityEnvironment

BUFFER_SIZE = 100000  # replay buffer size
BATCH_SIZE = 1024 # minibatch size


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
    episode_length = 2000

    # how many episodes to save policy
    save_interval = 1000

    # amplitude of OU noise
    # this slowly decreases to 0
    noise = 1
    noise_reduction = 0.9999

    # how many steps before update
    steps_per_update = 100

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
        obs_full = np.concatenate(obs)

        # for calculating rewards for this particular episode - addition of all time steps
        for episode_t in range(episode_length):

            actions = maddpg.act(transpose_to_tensor(list(obs)), noise=noise)
            noise *= noise_reduction

            actions = torch.stack(actions).view(-1).detach().cpu().numpy()
            env_info = env.step(actions)[brain_name]

            next_obs = env_info.vector_observations  # get the next state
            next_obs_full = np.concatenate(next_obs)
            rewards = env_info.rewards  # get the reward
            dones = env_info.local_done  # see if episode has finished

            # add experiences to buffer
            transition = (obs, obs_full, actions, rewards, next_obs, next_obs_full, dones)

            buffer.push(transition)

            reward_this_episode += rewards

            obs, obs_full = next_obs, next_obs_full

            # update once after every steps_per_update
            if len(buffer) > BATCH_SIZE and (episode_t % steps_per_update == 0):
                # print("mapddpg update {}".format(episode_t))
                for agent_idx in range(2):
                    samples = buffer.sample(BATCH_SIZE)
                    maddpg.update(samples, agent_idx)
                maddpg.update_targets()  # soft update the target network towards the actual networks

        print("episode rewards {}".format(reward_this_episode))
        agent0_reward.append(reward_this_episode[0])
        agent1_reward.append(reward_this_episode[1])

        if episode % 10 == 0 or episode == number_of_episodes - 1:
            avg_rewards = [np.mean(agent0_reward), np.mean(agent1_reward)]
            agent0_reward = []
            agent1_reward = []
            for agent_idx, avg_rew in enumerate(avg_rewards):
                print('agent%i/mean_episode_rewards' % agent_idx, avg_rew, episode)

    env.close()


if __name__ == '__main__':
    maddpg()
