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

seeds += add_b(150,[4,6,9])
seeds += add_b(160,[3,9])
seeds += add_b(170,[2,4])
seeds += add_b(180,[2])

# seeds = seeds1 + seeds2 + seeds3 + seeds4 + seeds5
# seeds= range(10)
for seed in seeds:


	dat = np.genfromtxt('res/mountain_graph1_seed{}.csv'.format(seed), delimiter=',')[:950]
	print(len(dat))
	data.append(dat)

axes.append(sns.tsplot(data=data,legend=True,condition='GCN',color='red'))



data=[]
for seed in seeds:



	dat = np.genfromtxt('res/mountain_graph0_seed{}.csv'.format(seed), delimiter=',')[:950]
	print(len(dat))
	data.append(dat)
axes.append(sns.tsplot(data=data,legend=True,condition='Primitive actions',color='blue'))


plt.xlabel('Iterations',fontsize=18)
plt.ylabel('Cumulative Steps',fontsize=18)
plt.legend()
plt.title("Results on MountainCar-v0")
plt.savefig('mountain_results_steps.png')
plt.clf()
