import numpy as np
from utils import *
import matplotlib.pyplot as plt
from tqdm import tqdm

agents = [Agent(lambda x: 0.1, eps=0.1),
              Agent(lambda x: 1/x, eps=0.1), ]

n_episodes = 2000
n_steps = 10000
mean_rewards = np.zeros((len(agents), n_steps))
for j in tqdm(range(n_episodes)):
    for agent in agents:
        agent.reset()

    envs = [KBandits(10, stationary=False), KBandits(10, stationary=False)] 

    for i in range(n_steps):
        for idx, (env, agent) in enumerate(zip(envs, agents)):
            action = agent.act()
            rw = env.step(action)
            agent.update(action, rw)

            mean_rewards[idx, i] += rw/n_episodes

plt.plot(mean_rewards[0, :], label=r'Constant $\alpha=0.1$')
plt.plot(mean_rewards[1, :], label='Mean sampling')

plt.xlabel('Step')
plt.xlabel('Reward')

plt.grid()
plt.legend()

plt.show()




    


