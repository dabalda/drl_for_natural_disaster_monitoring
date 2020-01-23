from scipy import stats
import numpy as np

rew_hist1 = np.load('../../../../Results/models/wildfire/indiv/dqn/test_100_episode_rewards.npy')
rew_hist2 = np.load('../../../../Results/models/wildfire/indiv/drqn/trace_08/test_100_episode_rewards.npy')
rew_hist3 = np.load('../../../../Results/models/wildfire/indiv/drqn/trace_20/test_100_episode_rewards.npy')

print('Model 1 mean:' + str(np.mean(rew_hist1)))
print('Model 2 mean:' + str(np.mean(rew_hist2)))
print('Model 3 mean:' + str(np.mean(rew_hist3)))

import matplotlib.pyplot as plt

n, bins, patches = plt.hist(x=(rew_hist1, rew_hist2, rew_hist3),
                            bins=30,
                            color=('#aa0405', '#05aa04','#0504aa'),
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Episode reward')
plt.ylabel('Frequency')
plt.title('')
maxfreq = n[0].max()
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

plt.show()

t_stat, p_val = stats.ttest_ind(rew_hist1, rew_hist3, equal_var=False)

print('T-statistic: ' + str(t_stat))
print('P-value: ' + str(p_val))