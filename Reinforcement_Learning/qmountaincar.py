# Authors: Dimitrios Gagatsis // Christos Kaparakis
import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # ; sns.set_theme()

gym.envs.register(
    id='MountainCarMyEasyVersion-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=10000,  # MountainCar-v0 uses 200
)
env = gym.make('MountainCarMyEasyVersion-v0')

# Action space is [0, 1, 2] where: 0=move left, 1=don't move, 2=move right

########################################################################################

# rewards
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': [], 'total': []}


########################################################################################


def qlearning(q_table, episodes, show_every, gamma, save, plotheatmap, plotrewards):
    for i in range(episodes):

        ep_rewards = []
        if i % show_every == 0 and i > 0:
            render = True
            print('Rendering episode ' + str(i))
        else:
            render = False

        episode_reward = 0
        observation = env.reset()
        done = False
        timesteps = 0

        index = (observation - env.observation_space.low) / discrete_os_win_size
        index1 = int(index[0])
        index2 = int(index[1])

        while not done:

            if render:
                env.render()

            action = np.argmax(q_table[index1, index2])
            observation, reward, done, info = env.step(action)
            timesteps += 1
            episode_reward += reward

            new_index = (observation - env.observation_space.low) / discrete_os_win_size
            new_index1 = int(new_index[0])
            new_index2 = int(new_index[1])

            # print(timesteps)
            q_table[index1, index2, action] = reward + gamma * np.max(q_table[new_index1, new_index2])

            index1, index2 = new_index1, new_index2
            ep_rewards.append(reward)

        env.close()

        aggr_ep_rewards['ep'].append(i)
        aggr_ep_rewards['avg'].append(sum(ep_rewards) / len(ep_rewards))
        aggr_ep_rewards['min'].append(min(ep_rewards))
        aggr_ep_rewards['max'].append(max(ep_rewards))
        aggr_ep_rewards['total'].append(sum(ep_rewards))

        print(f"Episode {i} finished after {timesteps} timesteps with reward {episode_reward}.")

    if save:
        np.save('qvalues50', q_table)

    if plotheatmap:
        fig, ax = plt.subplots()
        x = np.linspace(-0.07, 0.07, 25)
        y = np.linspace(0.6, 1.2, 25)
        sns.heatmap(np.max(q_table, axis=2))
        ax.set_xticklabels(np.around(x, 3))
        ax.set_yticklabels(np.around(y, 3))
        ax.set_title('Value function of states with '+str(episodes)+' episodes. Gamma: '+str(gamma))
        plt.show()

    if plotrewards:
        # plt.plot(aggr_ep_rewards['avg'], label="Average reward")
        # plt.plot(aggr_ep_rewards['min'], label="Minimum reward")
        # plt.plot(aggr_ep_rewards['max'], label="Maximum reward")
        plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['total'], label="Total reward per episode")
        plt.legend()
        plt.show()


######################
# First Q-Table

# q_table = np.random.uniform(low=-1, high=1, size=(50,50,env.action_space.n))
q_table = np.zeros((100, 100, env.action_space.n))
# print(q_table)

###################

# DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
DISCRETE_OS_SIZE = [100, 100]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# Set hyperparameters
episodes = 10000
show_every = 1000
gamma = 0.995  # discount factor
save = True
plotheatmap = True
plotrewards = True

qlearning(q_table, episodes, show_every, gamma, save, plotheatmap, plotrewards)
