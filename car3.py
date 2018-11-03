
import pdb

import networkx as nx
import gym
import itertools
import matplotlib
import numpy as np
import sys
import tensorflow as tf
import collections

import sklearn.pipeline
import sklearn.preprocessing
from sklearn.metrics import pairwise_distances
import scipy.sparse as sp

from cont_utils import *
from graph import *


from sklearn.kernel_approximation import RBFSampler
import sklearn.neighbors as nn
import matplotlib.pyplot as plt



first=0
seed= 3
plots=0



flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_integer('gen', 1, 'Do you want to generate a graph?')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('epochs', 50, 'Number of epochs to train.')
flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
flags.DEFINE_float('weight_decay', 1e-2, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('nf', 1, 'Create features or not.')
flags.DEFINE_integer('f', 0, 'Create features or not.')
flags.DEFINE_string('fig', '', 'Figure identifier.')
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_string('app', '', 'For data file loading') 
flags.DEFINE_integer('max_degree', 2, 'Maximum Chebyshev polynomial degree.')




flags.DEFINE_integer('ngraph', 10, "Number of episodes before graph generation")
flags.DEFINE_integer('nepisodes', 500, "Number of episodes per run")
flags.DEFINE_integer('nruns',1, "Number of runs")
flags.DEFINE_integer('nsteps',1000, "Maximum number of steps per episode")
flags.DEFINE_integer('noptions',1, 'Number of options')
flags.DEFINE_integer('baseline',1, "Use the baseline for the intra-option gradient")
flags.DEFINE_integer('primitive',0, "Augment with primitive")



flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 46, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden3', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden4', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden5', 8, 'Number of units in hidden layer 1.')






# pdb.set_trace()
env = gym.envs.make("MountainCar-v0")

env._max_episode_steps= 1000


env.seed(seed)
np.random.seed(seed)

print('seed: {} '.format(seed))

# pdb.set_trace()


observation_examples = np.array([env.observation_space.sample() for _ in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)


featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100, random_state=seed)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100, random_state=seed)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100, random_state=seed)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100, random_state=seed))
        ])
featurizer.fit(scaler.transform(observation_examples))


feats=400


def featurize_state(state):
    """
    Returns the featurized representation for a state.
    """
    # pdb.set_trace()

    scaled = scaler.transform(state)
    featurized = featurizer.transform(scaled)
    # print(featurized)
    return featurized


    # return scaled



class PolicyEstimator():
    """
    Policy Function approximator. 
    """
    
    def __init__(self, learning_rate=0.01, scope="policy_estimator"):
        # pdb.set_trace()
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [None,feats], "state")
            self.action = tf.placeholder(tf.int32, name="action")
            self.advantage = tf.placeholder(dtype=tf.float32, name="advantage")

            # This is just linear classifier
            hid = self.state

            hid = tf.contrib.layers.fully_connected(
                inputs=self.state,
                num_outputs=400,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.contrib.layers.xavier_initializer()
                )
            
            
            self.actions = tf.contrib.layers.fully_connected(
                inputs=hid,
                num_outputs=env.action_space.n,
                activation_fn=tf.nn.softmax,
                weights_initializer=tf.contrib.layers.xavier_initializer()
                )[0]
            # pdb.set_trace()
            entropy = - tf.reduce_sum( tf.log(self.actions)*self.actions )

            # pdb.set_trace()
            # sess = tf.Session();sess.run(tf.global_variables_initializer());pdb.set_trace()
            # sess.run(self.action,feed_dict={self.action:0})

            self.loss = - tf.log( tf.maximum(self.actions[self.action], 1e-7)) * self.advantage
            self.loss -= 1e-2 * entropy


            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.train.get_global_step())
    
    def predict(self,state,sess=None):
        # pdb.set_trace()
        sess = sess or tf.get_default_session()
        state = featurize_state(state)
        actions = sess.run(self.actions, { self.state: state })
        return np.random.choice(len(actions),p=actions)

    def update(self, state, advantage, action, sess=None):
        # pdb.set_trace()
        sess = sess or tf.get_default_session()
        state = featurize_state(state)
        feed_dict = { self.state: state, self.advantage: advantage, self.action: action  }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss




class ValueEstimator():
    """
    Value Function approximator. 
    """
    
    def __init__(self, learning_rate=0.1, scope="value_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [None,feats], "state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            hid= self.state

            hid = tf.contrib.layers.fully_connected(
                inputs=self.state,
                num_outputs=400,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.contrib.layers.xavier_initializer()
                )

            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=hid,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer()
                )


            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)


            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            # self.optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
            # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.train.get_global_step())        
    
    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        
        state = featurize_state(state)
        val = sess.run(self.value_estimate, { self.state: state })
        # pdb.set_trace()
        return val 

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        # pdb.set_trace()
        state = featurize_state(state)
        feed_dict = { self.state: state, self.target: target }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


# In[15]:

def actor_critic(sess,env, estimator_policy, estimator_value, num_episodes, discount_factor=1.0):
    global first

    G = nx.Graph()
    colors= []

    totsteps =0
    positions = np.arange(-1.2,0.6,0.01)
    velolicties = np.arange(-0.07,0.07,0.001)

    vinput= []
    for vel in velolicties:
        for pos in positions:
            vinput.append([pos,vel])
    vinput = np.array(vinput)

    # v_preds=estimator_value.predict(vinput).reshape(len(velolicties),len(positions)) 
    # fig2,ax2 = plt.subplots()
    # ax2.imshow(v_preds, interpolation='nearest')
    # plt.savefig("vpreds/vpred{}.png".format(0))
    # plt.clf();plt.close()    
    # pdb.set_trace()



    states= []
    node=-1
    done=False
    curr_episode = -1
    for i_episode in range(num_episodes):

        if i_episode % 1 ==0:
            print('new graph')
            G = nx.Graph()
            states= []
            node=-1
            curr_episode=i_episode-1

        state = env.reset()
        states.append(state) 

        # episode = []
        rewards =0

        # fig,ax = plt.subplots()
        # states= []

        for t in itertools.count():
            

            # pdb.set_trace()
            action = estimator_policy.predict([state])
            next_state, reward, done, _ = env.step(action)


            # print(reward)

            if curr_episode == i_episode:
                G.add_edge(node,node+1)
            else:
                curr_episode= i_episode
            node+=1

            rewards += reward

            
            # Calculate TD Target
            value_next = estimator_value.predict([next_state])
            td_target = reward + (1-done) * discount_factor * value_next
            advantage = td_target - estimator_value.predict([state])
            
            # print(td_target)

            # if first:
            estimator_value.update([state], td_target)
            estimator_policy.update([state], advantage, action)
            
            state = next_state
            if not done:
                states.append(state) 

            if done:
                totsteps+=t 
                print("\rEpisode {}/{} Stesp {} Total Steps {} ({})".format(i_episode, num_episodes, t,totsteps, rewards) )
                rewards =0

                pos = {i:(states[i][0],states[i][1]) for i in range(len(states))}
                this_color = [i_episode+1] * (t+1)
                colors += this_color
                # v_preds=estimator_value.predict(vinput).reshape(len(velolicties),len(positions)) 
                # minv = np.min(v_preds)
                v_preds=estimator_value.predict(vinput).reshape(len(velolicties),len(positions)) 
                minv = np.min(v_preds)

                if plots:
                    # pdb.set_trace()
                    # plt.autoscale(False)
                    plt.xlim((-1.2,0.6))
                    plt.ylim((-0.07,0.07))   
                    # plt.xlim((0,180))
                    # plt.ylim((0,140))   

                    nx.draw(G,pos, with_labels=False, font_size=7, node_size=5,node_color='blue')

                    # plt.show()

                    plt.savefig("graphs/graph{}.png".format(i_episode+1))
                    plt.clf();plt.close()




                    fig,ax = plt.subplots()

                    ax.imshow(v_preds, interpolation='nearest', alpha=1.)
                    # ax.autoscale(False)
                    # nx.draw(G,pos, with_labels=False, font_size=7, node_size=5,node_color=colors)

                    # plt.show()

                    plt.savefig("vpreds/vpred{}.png".format(i_episode+1))
                    plt.clf();plt.close()


                # pdb.set_trace()
                if  t<env._max_episode_steps-1 and first:

                    first=False
                    # pdb.set_trace()
                    aspect = (0.6 + 1.2) / (2*0.07)
                    metric = lambda p0, p1: np.sqrt((p1[0] - p0[0]) * (p1[0] - p0[0]) + (p1[1] - p0[1]) * (p1[1] - p0[1]) * aspect)
                    # dist='euclidean'


                    radius = 0.02
                    # real_states = np.array(states)[-(t+1):]
                    real_states = np.array(states)

                    adj = nn.radius_neighbors_graph(real_states,radius,metric=metric)
                    adj = adj+nx.adjacency_matrix(G)
                    gg = nx.from_scipy_sparse_matrix(adj)

                    
                    # nx.draw(gg,pos, with_labels=False, font_size=10, node_size=5,node_color=colors)
                    # plt.show()

                    
                    source = 0 
                    sink = len(real_states) -1
                    max_sources = 40
                    max_sinks=40
                    other_sources =range(max_sources)
                    other_sinks =range(len(real_states)-max_sinks,len(real_states))

                    # pdb.set_trace()
                    features = np.eye(len(real_states), dtype=np.float32)
                    features = sparse_to_tuple(sp.lil_matrix(features))

                    labels = np.zeros((len(real_states)))
                    labels[-max_sinks:] = 1
                    labels = encode_onehot(labels)

                    
                    V_weights = get_graph(gg.edges(),adj,features,labels,source,sink,other_sources,other_sinks)


                    # pdb.set_trace()
                    # pos = {i:(real_states[i][0],real_states[i][1]) for i in range(len(real_states))}
                    # nx.draw(gg,pos, with_labels=False, font_size=10, node_size=25,node_color=V_weights)
                    # plt.show()
                    # plt.clf();plt.close()

                    # pdb.set_trace()
                    minv=-10 if minv >-10 else minv
                    interpol = make_interpolater(min(V_weights),max(V_weights),0,1.)
                    targets = interpol(V_weights)

                    # oldlr = estimator_value.optimizer._lr
                    estimator_value.optimizer._lr= 0.1
                    # estimator_value.optimizer._learning_rate= 0.1
                    # pdb.set_trace()
                    for epo in range(10):
                        estimator_value.update(real_states, targets)

                        # fig,ax = plt.subplots()
                        # v_preds=estimator_value.predict(vinput).reshape(len(velolicties),len(positions)) 
                        # ax.imshow(v_preds, interpolation='nearest', alpha=1.)
                        # ax.autoscale(False)
                        # nx.draw(G,pos, with_labels=False, font_size=7, node_size=5,node_color=colors)

                        # # plt.show()

                        # plt.savefig("newpreds{}.png".format(epo))
                        # plt.clf();plt.close()


                    
                    fig,ax = plt.subplots()
                    v_preds=estimator_value.predict(vinput).reshape(len(velolicties),len(positions)) 

                    ax.imshow(v_preds, interpolation='nearest', alpha=1.)
                    # ax.autoscale(False)
                    # nx.draw(G,pos, with_labels=False, font_size=7, node_size=5)
                    # nx.draw(G,pos, with_labels=False, font_size=7, node_size=5,node_color=colors)

                    plt.show()

                    # pdb.set_trace()


                    # plt.show()
                    # pdb.set_trace()
                    estimator_value.optimizer._lr=1e-5
                    # estimator_value.optimizer._learning_rate=1e-5
                    env.env.reward = 0.

                    

                break
            
            
    
    return stats




# tf.reset_default_graph()


with tf.Session() as sess:
    tf.set_random_seed(seed)
    global_step = tf.Variable(0, name="global_step", trainable=False)

    policy_estimator = PolicyEstimator(learning_rate=0.001)
    value_estimator = ValueEstimator(learning_rate=0.001)

    sess.run(tf.global_variables_initializer())

    # pdb.set_trace()
    stats = actor_critic(sess,env, policy_estimator, value_estimator, 1000, discount_factor=0.95)








