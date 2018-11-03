import gym
import argparse
import numpy as np
from fourrooms import Fourrooms
from utils import *
import matplotlib.pyplot as plt
import matplotlib
from pylab import *
import os 
import networkx as nx

import pdb
from scipy.special import expit
from scipy.misc import logsumexp
import dill
colors = [(0,0,0)] + [(cm.viridis(i)) for i in xrange(1,256)]
new_map = matplotlib.colors.LinearSegmentedColormap.from_list('new_map', colors, N=256)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--discount', help='Discount factor', type=float, default=0.99)
    parser.add_argument('--lr_term', help="Termination gradient learning rate", type=float, default=1e-3)
    parser.add_argument('--lr_intra', help="Intra-option gradient learning rate", type=float, default=1e-3)
    parser.add_argument('--lr_critic', help="Learning rate", type=float, default=1e-2)
    parser.add_argument('--epsilon', help="Epsilon-greedy for policy over options", type=float, default=1e-2)
    parser.add_argument('--nepisodes', help="Number of episodes per run", type=int, default=1)
    parser.add_argument('--nruns', help="Number of runs", type=int, default=1)
    parser.add_argument('--nsteps', help="Maximum number of steps per episode", type=int, default=1000)
    parser.add_argument('--noptions', help='Number of options', type=int, default=1)
    parser.add_argument('--baseline', help="Use the baseline for the intra-option gradient", action='store_false', default=True)
    parser.add_argument('--temperature', help="Temperature parameter for softmax", type=float, default=1e-2)
    parser.add_argument('--primitive', help="Augment with primitive", default=False, action='store_true')

    args = parser.parse_args()

    seed = 123
    rng = np.random.RandomState(seed)
    # env = gym.make('Fourrooms-v0')
    env = Fourrooms()


    fname = '-'.join(['{}_{}'.format(param, val) for param, val in sorted(vars(args).items())])
    fname = 'optioncritic-fourrooms-' + fname + '.npy'

    possible_next_goals = [68, 69, 70, 71, 72, 78, 79, 80, 81, 82, 88, 89, 90, 91, 92, 93, 99, 100, 101, 102, 103]

    history = np.zeros((args.nruns, args.nepisodes, 2))
    for run in range(args.nruns):
        
        
        features = Tabular(env.observation_space.n)
        nfeatures, nactions = len(features), env.action_space.n

        # pdb.set_trace()
        G = nx.Graph()
        G.add_nodes_from(range(len(env.occupancy.flatten())))
        grid_dict = dict(zip(range(env.observation_space.n), 
                                np.argwhere(env.occupancy.flatten()==0).squeeze() ))

        # The intra-option policies are linear-softmax functions
        option_policies = [SoftmaxPolicy(rng, nfeatures, nactions, args.temperature) for _ in range(args.noptions)]
        if args.primitive:
            option_policies.extend([FixedActionPolicies(act, nactions) for act in range(nactions)])

        # The termination function are linear-sigmoid functions
        option_terminations = [SigmoidTermination(rng, nfeatures) for _ in range(args.noptions)]
        if args.primitive:
            option_terminations.extend([OneStepTermination() for _ in range(nactions)])


        policy = SoftmaxPolicy(rng, nfeatures, args.noptions, args.temperature)
        critic = IntraOptionQLearning(args.discount, args.lr_critic, option_terminations, policy.weights)
        action_weights = np.zeros((nfeatures, args.noptions, nactions)) # Learn Q(s,o,a) separately
        action_critic = IntraOptionActionQLearning(args.discount, args.lr_critic, option_terminations, action_weights, critic)

        termination_improvement= TerminationGradient(option_terminations, critic, args.lr_term)
        intraoption_improvement = IntraOptionGradient(option_policies, args.lr_intra)

        for episode in range(args.nepisodes):
            if episode == 1000:
                env.goal = rng.choice(possible_next_goals)
                print('************* New goal : ', env.goal)


            allobs = set()
            observation = env.reset()
            start = grid_dict.get(observation)
            allobs.add(start)

            phi = features(observation)
            option = policy.sample(phi)
            action = option_policies[option].sample(phi)
            critic.start(phi, option)
            action_critic.start(phi, option, action)

            
            cumreward = 0.
            duration = 1
            option_switches = 0
            avgduration = 0.
            for step in range(args.nsteps):
                next_observation, reward, done, _ = env.step(action)
                if observation != next_observation:
                    G.add_edge(grid_dict.get(observation),grid_dict.get(next_observation))
                observation=next_observation

                phi = features(observation)
                allobs.add(grid_dict.get(observation))
                # pdb.set_trace()

                action = option_policies[option].sample(phi)
                update_target = critic.update(phi, option, reward, done)
                action_critic.update(phi, option, action, reward, done)

                if isinstance(option_policies[option], SoftmaxPolicy):
                    # Intra-option policy update
                    critic_feedback = action_critic.value(phi, option, action)
                    if args.baseline:
                        critic_feedback -= critic.value(phi, option)
                    intraoption_improvement.update(phi, option, action, critic_feedback)

                    # Termination update
                    termination_improvement.update(phi, option)

                cumreward += reward
                duration += 1
                if done:
                    break


            history[run, episode, 0] = step
            history[run, episode, 1] = avgduration
            print('Run {} episode {} steps {} cumreward {} avg. duration {} switches {}'.format(run, episode, step, cumreward, avgduration, option_switches))
            allobs = list(allobs)


            # pdb.set_trace()
            # Plot the V function
            plan = env.occupancy.flatten().astype(float)
            plot_critic = critic.weights[:,0].copy()
            plot_critic[plot_critic == 0] = -0.01
            plan[plan == 0] = plot_critic
            plan[plan == 1] = -.02


            # #Plot the path
            # mypath = np.zeros_like(critic.weights[:,0])
            # mypath[allobs] = 0.5
            # mypath[start] = 0.25
            # mypath[env.goal] = 0.75
            path = env.occupancy.flatten().astype(float)
            # path[path == 0] = mypath
            path[path == 1] = -.5
            path[allobs] = 0.5
            path[start] = 0.25
            path[grid_dict.get(env.goal)] = 0.75



            # fig,ax = plt.subplots(2)
            # ax[0].imshow(path.reshape(env.occupancy.shape),cmap=new_map)
            # ax[1].imshow(plan.reshape(env.occupancy.shape),cmap=new_map)
            # plt.show()



            ## Plot the Q functions
            # acts = ['up','down','left','right']
            # for i,act in zip(range(env.action_space.n),acts):
                
            #     plan = env.occupancy.flatten().astype(float)
            #     plot_critic = action_critic.weights[:,0,i].copy()
            #     plot_critic[plot_critic == 0] = -0.01
            #     plan[plan == 0] = plot_critic
            #     plan[plan == 1] = -.02   
            #     ax[i+2].imshow(plan.reshape(env.occupancy.shape),cmap=new_map)
 


            # directory = "randwalk/"
            # if not os.path.exists(directory):
            #     os.makedirs(directory)
            # plt.savefig("{}epoch{}_seed{}.png".format(directory,episode,seed))
            # # plt.close()


            interpol = make_interpolater(min(critic.weights),max(critic.weights),0.,1.)
            features =[]
            for w in critic.weights:
                features.append(interpol(w)[0])


            allfeats = np.zeros_like(env.occupancy).flatten()
            feat_indices = (env.occupancy == 0).flatten()
            allfeats[feat_indices] = features

            row,col = env.occupancy.shape
            source = start 
            sink = grid_dict.get(env.goal)
            title='fourroom'
            with open("gcn/data/{}_edges.txt".format(title),"w+") as f:
                for line in nx.generate_edgelist(G, data=False):
                    f.write(line+"\n")

            # pdb.set_trace()
            with open("gcn/data/{}_info.txt".format(title),"w+") as f:
                f.write("{} {}\n".format(row,col))
                f.write("{} {}\n".format(source,sink))
                for state,feat in zip(allobs,allfeats[allobs]):
                    f.write("{} {}\n".format(state,feat))

            # pdb.set_trace()
                

