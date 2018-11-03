from __future__ import division
from __future__ import print_function


import time
import tensorflow as tf
import pdb
import matplotlib.pyplot as plt
import networkx as nx
import os
from sklearn.cluster import SpectralClustering
from sklearn import metrics

import gcn.globs as g
from cont_utils import *
from gcn.models import GCN
from sim import *
from transfer import *
colors = [(0,0,0)] + [(cm.viridis(i)) for i in xrange(1,256)]
new_map = matplotlib.colors.LinearSegmentedColormap.from_list('new_map', colors, N=256)

flags = tf.app.flags
FLAGS = flags.FLAGS

lastoutputs= None
def get_graph(edges,gg,real_states,adj,features,labels,source,sink,other_sources,other_sinks):

    sess = tf.Session()

    y_train, y_val, train_mask, val_mask = get_splits(labels, source, sink, other_sources, other_sinks)



    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN


    adj =adj.toarray()
    deg = np.diag(np.sum(adj,axis=1))
    laplacian = deg - adj



    # Define placeholders
    placeholders = {
        'adj': tf.placeholder(tf.float32, shape=(None, None)) , #unnormalized adjancy matrix
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
        'learning_rate': tf.placeholder(tf.float32)
    }




    model = model_func(placeholders,edges,laplacian, input_dim=features[2][1], logging=True,FLAGS=FLAGS)

    # remain_vars= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gcn')
    
    gcn_vars = []
    for var in tf.global_variables():
        if 'gcn' in var.name:
            gcn_vars.append(var)
    # pdb.set_trace()
    sess.run(tf.variables_initializer(gcn_vars))
    # sess.run(tf.variables_initializer(tf.global_variables()[16:]))
    
        

    feed_dict = construct_feed_dict(adj, features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['learning_rate']: FLAGS.learning_rate})

    cost_val = []


    pos = {i:(real_states[i][0],real_states[i][1]) for i in range(len(real_states))}
    start = time.time()
    for epoch in range(FLAGS.epochs):

        outputs = sess.run([tf.nn.softmax(model.outputs)], feed_dict=feed_dict)[0]

        # pdb.set_trace()
        fig,ax = plt.subplots()
        nx.draw(gg,pos, with_labels=False, font_size=10, node_size=25,node_color=outputs[:,1])
        plt.savefig("updated_graph/iter{}.png".format(epoch))
        plt.clf();plt.close()

        t = time.time()
        # pdb.set_trace()
        if epoch >-1:
            feed_dict = construct_feed_dict(adj, features, support, y_train, train_mask, placeholders)
        else:
            feed_dict = construct_feed_dict(adj, features, support, y_val, val_mask, placeholders)

        feed_dict.update({placeholders['learning_rate']: FLAGS.learning_rate})

        outs = sess.run([model.opt_op, model.loss, model.accuracy,model.learning_rate], feed_dict=feed_dict)




    print("Total time for gcn {}".format(time.time()-start))
    print("Optimization Finished!")



    outputs = sess.run([tf.nn.softmax(model.outputs)], feed_dict=feed_dict)[0]

    return outputs[:,1]








# get_graph([132])
# get_graph(range(130,135))