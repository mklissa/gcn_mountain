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
from graph import *

import pdb
from scipy.special import expit
from scipy.misc import logsumexp
import dill
colors = [(0,0,0)] + [(cm.viridis(i)) for i in xrange(1,256)]
new_map = matplotlib.colors.LinearSegmentedColormap.from_list('new_map', colors, N=256)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

seeds = [140]
# seeds = range(140,150)
want_graph=1
plotsteps=0

"""
(Pdb) print(step)
0
(Pdb) episode
101
(Pdb) seed
144
"""

flags.DEFINE_integer('epochs', 50, 'Number of epochs to train.')
flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
flags.DEFINE_float('weight_decay', 1e-2, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('nf', 1, 'Create features or not.')
flags.DEFINE_integer('f', 0, 'Create features or not.')
flags.DEFINE_string('fig', '', 'Figure identifier.')
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_string('app', '', 'For data file loading') 
flags.DEFINE_integer('max_degree', 2, 'Maximum Chebyshev polynomial degree.')




flags.DEFINE_integer('ngraph', 5, "Number of episodes before graph generation")
flags.DEFINE_integer('nepisodes', 400, "Number of episodes per run")
flags.DEFINE_integer('nruns',1, "Number of runs")
flags.DEFINE_integer('nsteps',10000, "Maximum number of steps per episode")
flags.DEFINE_integer('noptions',1, 'Number of options')
flags.DEFINE_integer('baseline',True, "Use the baseline for the intra-option gradient")
flags.DEFINE_integer('primitive',False, "Augment with primitive")



flags.DEFINE_float('temperature',1e-3, "Temperature parameter for softmax")
flags.DEFINE_float('discount',0.99, 'Discount factor')
flags.DEFINE_float('lr_intra',1e-1, "Intra-option gradient learning rate")
flags.DEFINE_float('lr_critic',1e-1, "Learning rate")





# flags.DEFINE_integer('hidden1', 500, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden2', 250, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden3', 32, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden4', 28, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden5', 18, 'Number of units in hidden layer 1.')



# flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden2', 64, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden3', 46, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden4', 28, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden5', 18, 'Number of units in hidden layer 1.')


flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 46, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden3', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden4', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden5', 8, 'Number of units in hidden layer 1.')



# flags.DEFINE_integer('hidden1', 256, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden2', 128, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden3', 64, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden4', 32, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden5', 16, 'Number of units in hidden layer 1.')





for seed in seeds:
    print('seed:',seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    rng = np.random.RandomState(seed)

    
    env = Fourrooms()
    walls = np.argwhere(env.occupancy.flatten()==1)
    # possible_next_goals = [68, 69, 70, 71, 72, 78, 79, 80, 81, 82, 88, 89, 90, 91, 92, 93, 99, 100, 101, 102, 103]


    # for run in range(FLAGS.nruns):
    run=0
    option = 0


    features = Tabular(env.observation_space.n)
    nfeatures, nactions = len(features), env.action_space.n

    # pdb.set_trace()
    observations = set()
    G = nx.Graph()
    G.add_nodes_from(range(len(env.occupancy.flatten())))
    full_grid_dict = dict(zip(range(env.observation_space.n), 
                            np.argwhere(env.occupancy.flatten()==0).squeeze() ))

    # pdb.set_trace()

    options_dict = [dict(zip( np.argwhere(env.occupancy.flatten()==0).squeeze(),
                            range(env.observation_space.n) )), ]

    option_policies = [SoftmaxPolicy(rng, nfeatures, nactions, FLAGS.temperature), ]
    intraoption_improvement = IntraOptionGradient(option_policies, FLAGS.lr_intra)

    critics = [StateValue(FLAGS.discount, FLAGS.lr_critic, np.zeros((nfeatures)) ), ]
    action_critics = [ActionValue(FLAGS.discount, FLAGS.lr_critic, np.zeros((nfeatures,nactions)) ), ]

    done=False
    cumsteps = 0.
    optionsteps = 0.
    myrand = 1.
    myrandinit =1.
    sources=[]
    init_set=[]
    goals = []
    allstates=[]
    for episode in range(FLAGS.nepisodes):

        phis = []
        rewards = []
        pos = env.occupancy.flatten().astype(float)
        pos[pos == 1] = -.5
        
        observation = env.reset()
        # start = full_grid_dict.get(observation)
        start=observation
        sources.append(start)
        observations.add(start)
        
        if observation in allstates:
            option =1
        else:
            option =0


        # if observation in init_set:
        #     next_option = 1
        # else:
        #     next_option = 0



        last_phi = phi = options_dict[option].get(observation)
        phis.append(phi)

        action = option_policies[option].sample(phi,0)
        critics[option].start(phi)
        action_critics[option].start(phi, action)

        
        cumreward = 0.
        duration = 1
        option_switches = 0
        avgduration = 0.
        for step in range(FLAGS.nsteps):

            next_observation, reward, done, _ = env.step(action)
            real_reward = reward
            rewards.append(reward)
            if observation != next_observation:
                G.add_edge(observation,next_observation)
            observation=next_observation
            observations.add(observation)
            



            pos[observation] += 0.1    
            if option ==1 or observation in map(env.state_dict.get,env.init_states):
                optionsteps += 1

            if observation in allstates:
                next_option = 1
            else:
                next_option = 0  

            # if observation in init_set:
            #     next_option = 1
            # else:
            #     next_option = 0

            # if goals:
            #     if observation in goals and option ==1:
            #         reward = 1.






            # if (done or step==999) and plotsteps:
            if init_set and (done or step==FLAGS.nsteps-1) and plotsteps:
            # if init_set and step>0 and plotsteps:

                # pdb.set_trace()

                # pos = env.occupancy.flatten().astype(float)
                # pos[pos == 1] = -.5
                # pos[observation] = 0.5

                critic_map0 = env.occupancy.flatten().astype(float)
                critic_vals0 = critics[0].weights.copy()
                critic_vals0[critic_vals0 == 0] = -0.01
                critic_map0[critic_map0 == 1] = -.02
                critic_map0[critic_map0 == 0] = critic_vals0


                critic_vals1 = critics[1].weights.copy()
                critic_map1 = env.occupancy.flatten().astype(float)
                critic_map1[critic_map1 == 1] = -.02
                critic_map1[allstates] = critic_vals1
                critic_map1[-1] = -0.01

                fig,ax = plt.subplots(3)
                ax[0].imshow(pos.reshape(env.occupancy.shape),cmap=new_map)
                ax[1].imshow(critic_map0.reshape(env.occupancy.shape),cmap=new_map)
                ax[2].imshow(critic_map1.reshape(env.occupancy.shape),cmap=new_map)


                # Plot the option policy
                # acts = ['up','down','left','right']
                # for i,act in zip(range(env.action_space.n),acts):
                    
                #     # pdb.set_trace()
                #     plan = env.occupancy.flatten().astype(float)
                #     pol_values = option_policies[1].weights[:,i].copy()
                #     plan[plan == 1] = min(pol_values)
                #     plan[plan == 0] = min(pol_values)                  
                    
                #     pol_values[pol_values == 0] = min(pol_values)
                #     plan[allstates] = pol_values

                #     ax[i+3].imshow(plan.reshape(env.occupancy.shape),cmap=new_map)


                # # Plot the Q functions
                # acts = ['up','down','left','right']
                # for i,act in zip(range(env.action_space.n),acts):
                    
                #     plan = env.occupancy.flatten().astype(float)
                #     plot_critic = action_critics[1].weights[:,i].copy()
                #     plot_critic[plot_critic == 0] = -0.01
                #     plan[allstates] = plot_critic
                #     plan[plan == 1] = -.02   
                #     ax[i+3].imshow(plan.reshape(env.occupancy.shape),cmap=new_map)


                # plt.show()
                # plt.close()



                directory = "afteroption/"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                plt.savefig("{}episode{}step{}_seed{}_option{}.png".format(directory,episode,step,seed,next_option))
                plt.close()

                # pdb.set_trace()







            phi = options_dict[option].get(observation)
            if phi is None:
                # pdb.set_trace()
                phi=last_phi


            critics[option].update(phi, reward, done)
            action_critics[option].update(phi, reward, done, critics[option].value(phi))

            option = next_option 
            phi = options_dict[option].get(observation)
            action = option_policies[option].sample(phi,step) if np.random.rand() > 0.1 else np.random.randint(4)
            phis.append(phi)

            if option ==1 and True:

                # myrand = myrandinit  * .1 **(episode/100)
                myrand=.1
                acts = [-col,col,-1,1]
                if np.random.rand() < myrand:
                    
                    action = np.random.randint(4)
                else:

                    vals = []
                    for a in range(4):
                        # pdb.set_trace()
                        newpos = options_dict[option].get(observation+acts[a])
                        if newpos is not None:
                            vals.append(critics[option].weights[newpos])
                        # elif observation+acts[a] in walls:
                        #     vals.append(-inf)
                        else:
                            vals.append(critics[option].weights[phi])
                        # print(vals)

                    # pdb.set_trace()
                    v = np.array(vals)/1e-3
                    probs =np.exp(v - logsumexp(v))
                    action =int(np.random.choice(4, p=probs))

                newpos = options_dict[option].get(observation+acts[action])
                newobs = observation+acts[action]
                if newpos is None and newobs not in walls:
                    allstates.append(newobs)
                    # allstates.sort()
                    critics[option].add(critics[option].weights[phi])
                    action_critics[option].add(action_critics[option].weights[phi])
                    option_policies[option].add(option_policies[option].weights[phi])
                    options_dict[option][newobs] = len(critics[option].weights) -1




            critics[option].start(phi)
            action_critics[option].start(phi, action)

            critic_feedback = action_critics[option].value(phi, action) 
            critic_feedback -= critics[option].value(phi)
            intraoption_improvement.update(phi, option, action, critic_feedback)

            cumreward += real_reward
            duration += 1
            last_phi = phi
            if done:
                break
        cumsteps += step
        print('Episode {} steps {} cumreward {} cumsteps {} optionsteps {}'.format(episode, step, cumreward, cumsteps,optionsteps))
        # pdb.set_trace()

        # if init_set:

        #     for i,phi in enumerate(phis[:-1]):
        #         gammas = np.array([.99**n for n in range(1,len(rewards) - i + 1)] )
        #         disc_return = np.array(rewards[i:]).dot(gammas) 
        #         critics[1].update2(disc_return,phi)

        #         # pdb.set_trace()
















        
        # if episode == FLAGS.nepisodes-1:
        # if episode == FLAGS.ngraph -1:
        if done and not init_set:

            # pdb.set_trace()
            allobs = list(observations)
            allobs.sort()

            if full_grid_dict.get(env.goal) not in allobs:
                print("No sink this time")
                continue

            # # pdb.set_trace()
            # option=0
            # # Plot the V function

            # critic_map = env.occupancy.flatten().astype(float)
            # critic_vals = critics[option].weights.copy()
            # critic_vals[critic_vals == 0] = -0.01
            # critic_map[critic_map == 1] = -.02
            # critic_map[critic_map == 0] = critic_vals


            # #plot feats
            # interpol = make_interpolater(min(critics[option].weights),max(critics[option].weights),0.,1.)
            # xtra_feats =[]
            # for w in critics[option].weights:
            #     xtra_feats.append(interpol(w))
            # feats_map = env.occupancy.flatten().astype(float)
            # feats_vals = np.array(xtra_feats).copy()
            # feats_map[feats_map == 1] = -.02
            # feats_map[feats_map == 0] = feats_vals


            # # #Plot the path
            # path = env.occupancy.flatten().astype(float)
            # path[path == 1] = -.5
            # path[allobs] = 0.5
            # path[start] = 0.25
            # path[full_grid_dict.get(env.goal)] = 0.75


            # fig,ax = plt.subplots(3)
            # ax[0].imshow(path.reshape(env.occupancy.shape),cmap=new_map)
            # ax[1].imshow(critic_map.reshape(env.occupancy.shape),cmap=new_map)
            # ax[2].imshow(feats_map.reshape(env.occupancy.shape),cmap=new_map)
            # plt.show()


            # # Plot the Q functions
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
            # plt.close()


            # pdb.set_trace()
            option=0
            interpol = make_interpolater(min(critics[option].weights),max(critics[option].weights),0.,1.)
            critic_features =[]
            for w in critics[option].weights:
                critic_features.append(interpol(w))
            critic_features[env.goal] = 1. # little hack

            # pdb.set_trace()
            allfeats = np.zeros_like(env.occupancy).flatten().astype(float)
            feat_indices = (env.occupancy == 0).flatten()
            allfeats[feat_indices] = critic_features


            row,col = env.occupancy.shape

            # pdb.set_trace()
            # sources = [start]
            # sources = map(env.state_dict.get,env.init_states)

            # sources = [sources[0] for _ in range(5) ]
            # for counter,source in enumerate(sources):
            #     if source not in allobs:
            #         continue

            source = sources[0]
            sink = full_grid_dict.get(env.goal)
            title='fourroom'
            with open("gcn/data/{}_edges.txt".format(title),"w+") as f:
                for line in nx.generate_edgelist(G, data=False):
                    f.write(line+"\n")

            with open("gcn/data/{}_info.txt".format(title),"w+") as f:
                f.write("{} {}\n".format(row,col))
                f.write("{} {}\n".format(source,sink))
                for state,feat in zip(allobs,allfeats[allobs]):
                    f.write("{} {}\n".format(state,feat))
            last_obs = allobs






            if want_graph:
                # pdb.set_trace()
                sess = tf.Session()
                # init_set,goals,sinks,V_weights = get_graph(seed,sess,sources[:10],[sink],0)
                init_set,goals,sinks,V_weights = get_graph(seed,sess,[sources[0]],[sink],0)
                # sess = tf.Session()
                # init_set,goals,_ = get_graph(seed,sess,sources[:3],sinks,1)

                # def replace(x):
                #     if x < .3:
                #         return 0.
                #     else:
                #         return x
                # V_weights = map(replace,V_weights)

                allstates = init_set + goals
                allstates.sort()
                nstates = len(allstates)

                # pdb.set_trace()
                option_policies.append(SoftmaxPolicy(rng, nstates, nactions, FLAGS.temperature) )
                intraoption_improvement.add_option(option_policies)

                critics.append(StateValue(FLAGS.discount, FLAGS.lr_critic, V_weights,) )
                action_critics.append(ActionValue(FLAGS.discount, FLAGS.lr_critic, np.zeros((nstates,nactions)),) )

                options_dict.append( dict(zip( allstates, range(nstates) )) )
                # FLAGS.nsteps = 10000



            # break
                    



# simfour(121)