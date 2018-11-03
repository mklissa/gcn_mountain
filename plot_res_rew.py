import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
import numpy as np
import pdb
sns.set(style='ticks')



def add_b(bias,seeds):
	for i in range(len(seeds)):
		seeds[i]+=bias
	return seeds

data=[]
axes=[]
seeds = []
seeds+= add_b(140,[2,3,4,6,7,9])
# seeds += add_b(150,[4,6,9])
# seeds += add_b(160,[3,9])
# seeds += add_b(180,[2])
# pdb.set_trace()
# seeds = [156]
for seed in seeds:


	dat = np.genfromtxt('res/mountain_graph1_seed{}.csv'.format(seed), delimiter=',')[:999]
	print(len(dat))
	rewards=[]
	tot=0
	for i in range(len(dat)-1):
		if dat[i+1] - dat[i] <399:
			# pdb.set_trace()
			tot+=1
		rewards.append(tot)
	dat=rewards
	data.append(dat)

axes.append(sns.tsplot(data=data,legend=True,condition='GCN',color='red'))



data=[]
for seed in seeds:



	dat = np.genfromtxt('res/mountain_graph0_seed{}.csv'.format(seed), delimiter=',')[:999]
	print(len(dat))
	rewards=[]
	tot=0
	for i in range(len(dat)-1):
		if dat[i+1] - dat[i] <399:
			tot+=1
		rewards.append(tot)
	dat=rewards
	data.append(dat)
axes.append(sns.tsplot(data=data,legend=True,condition='Primitive actions',color='blue'))


plt.xlabel('Iterations',fontsize=18)
plt.ylabel('Cumulative Rewards',fontsize=18)
plt.legend()
plt.title("Results on MountainCar-v0")
plt.savefig('mountain_results_rewards.png')
plt.clf()
