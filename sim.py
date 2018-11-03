import numpy as np
import pdb
import networkx as nx
import matplotlib.pyplot as plt
import os

def simulate(seed):
	# seed=223

	np.random.seed(seed)

	row=10
	col=10
	size=row*col


	walls=set()
	for i in range(size):
		if i < col or i >=((row-1)*col):
			walls.add(i)
		if i %col ==0 or i%col==(col-1) or i%col==(col/2):

			if (i//col>(row/3) and i//col< (row-row/3)) and i%col==(col/2):
				continue
			walls.add(i)
	walls=list(walls)

	allstates = range(size)
	allstates = [s for s in allstates if s not in walls]	
	G = nx.Graph()
	G.add_nodes_from(allstates)

	state=np.random.randint(size)
	while state in walls:
		state=np.random.randint(size)



	states=set()
	states.add(state)
	actions = [-col,1,col,-1] # up, right, down, left 
	steps=100
	for _ in range(steps):
		A=np.random.randint(4)
		next_state = state + actions[A]

		while next_state in walls:
			next_state = state + actions[np.random.randint(4)]

		G.add_edge(state,next_state)
		state=next_state
		states.add(state)





	# pdb.set_trace()
	source = -1
	sink = -1
	title='rand'
	with open("gcn/data/{}_edges.txt".format(title),"w+") as f:
		for line in nx.generate_edgelist(G, data=False):
			f.write(line+"\n")

	with open("gcn/data/{}_info.txt".format(title),"w+") as f:
		f.write("{} {}\n".format(row,col))
		f.write("{} {}\n".format(source,sink))
		for state in states:
			f.write("{} -1\n".format(state))

	return steps,size


	