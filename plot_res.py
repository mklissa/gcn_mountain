import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
import numpy as np
import pdb
from collections import deque
sns.set(style='ticks')



def add_b(bias,seeds):
	for i in range(len(seeds)):
		seeds[i]+=bias
	return seeds

data=[]
axes=[]
seeds = []
seeds+= add_b(140,[2,3,4,6,7,9])
seeds += add_b(150,[4,6,9])
seeds += add_b(160,[3,9])
# seeds += add_b(170,[2,4])
seeds += add_b(180,[2])
# pdb.set_trace()
# seeds= range(10)
for seed in seeds:


	dat = np.genfromtxt('res/mountain_graph1_seed{}.csv'.format(seed), delimiter=',')[:999]
	print(len(dat))
	rewbuffer = deque(maxlen=100)

	real_dat=[]
	tot=0
	for i in range(len(dat)-1):
		rew=0
		if dat[i+1] - dat[i] <399:
			rew=1

	
		rewbuffer.append(rew)
		real_dat.append(np.mean(rewbuffer))

	# 	rewards.append(tot)
	# dat=rewards
	data.append(real_dat)

axes.append(sns.tsplot(data=data,legend=True,condition='Diffusion-Based Approximate VF',color='red'))



data=[]
for seed in seeds:



	dat = np.genfromtxt('res/mountain_graph0_seed{}.csv'.format(seed), delimiter=',')[:999]
	print(len(dat))
	rewbuffer = deque(maxlen=100)

	real_dat=[]
	tot=0
	for i in range(len(dat)-1):
		rew=0
		if dat[i+1] - dat[i] <399:
			rew=1

	
		rewbuffer.append(rew)
		real_dat.append(np.mean(rewbuffer))

	# 	rewards.append(tot)
	# dat=rewards
	data.append(real_dat)
axes.append(sns.tsplot(data=data,legend=True,condition='Actor-Critic',color='blue'))


plt.xlabel('Episodes',fontsize=18)
plt.ylabel('Average Rewards',fontsize=18)
plt.legend()
plt.title("Results on SparseMountainCar-v0")
plt.savefig('mountain_results_rewards.png')
plt.clf()
