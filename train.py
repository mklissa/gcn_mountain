from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import pdb
import matplotlib.pyplot as plt
import networkx as nx
import os


import gcn.globs as g
from gcn.utils import *
from gcn.models import GCN, MLP
from sim import *
from transfer import *
colors = [(0,0,0)] + [(cm.viridis(i)) for i in xrange(1,256)]
new_map = matplotlib.colors.LinearSegmentedColormap.from_list('new_map', colors, N=256)



# simulate()

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_string('app', '', 'For data file loading') 
flags.DEFINE_float('learning_rate', 0.008, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 10, 'Number of epochs to train.')



# flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden2', 64, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden3', 32, 'Number of units in hidden layer 1.')

flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden3', 24, 'Number of units in hidden layer 1.')

# flags.DEFINE_integer('hidden1', 256, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden2', 128, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden3', 64, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden4', 32, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden5', 16, 'Number of units in hidden layer 1.')

flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 1e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 5000, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 2, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('nf', 1, 'Create features or not.')
flags.DEFINE_integer('f', 0, 'Create features or not.')
force_feats=FLAGS.f
force_nofeats=FLAGS.nf



# Set random seed
seeds = range(130,135)
# seeds=[121]
for seed in seeds:
    # steps,size =simulate(seed)
    simfour(seed)
    # pdb.set_trace()
    np.random.seed(seed)
    # seed = 125
    tf.set_random_seed(seed)



    features, featplot, adj, labels, vertices, edges, row, col, source, sink,other_sinks, grid_dict = load_other(append=FLAGS.app,force_feats=force_feats,force_nofeats=force_nofeats)
    y_train, y_val, train_mask, val_mask = get_splits(labels, source, sink,other_sinks)

    if sink is None:
        print("Sink not there, skipping this seed.")
        continue

    G=nx.Graph()
    G.add_nodes_from(range(len(vertices)))
    G.add_edges_from(edges,capacity=1)
    # pdb.set_trace()
    start = time.time()
    cut = nx.minimum_cut(G,source,sink)
    print("Total time for mincut {}".format(time.time()-start))
    # Some preprocessing
    # features = preprocess_features(features)
    if FLAGS.model == 'gcn':
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = GCN
    elif FLAGS.model == 'gcn_cheby':
        support = chebyshev_polynomials(adj, FLAGS.max_degree)
        num_supports = 1 + FLAGS.max_degree
        model_func = GCN
    elif FLAGS.model == 'dense':
        support = [preprocess_adj(adj)]  # Not used
        num_supports = 1
        model_func = MLP
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))
    # adj=normalize_adj(adj).toarray()
    # pdb.set_trace()
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
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }
    # pdb.set_trace()

    ### Setting up some default values for checkups
    g.feedz =  construct_feed_dict(adj, features, support, y_train, train_mask, placeholders)
    g.feedz.update({placeholders['dropout']: False})
    ###



    model = model_func(placeholders,edges,laplacian, input_dim=features[2][1], logging=True)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    # Define model evaluation function
    def evaluate(adj,features, support, labels, mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(adj,features, support, labels, mask, placeholders)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], (time.time() - t_test)


    feed_dict = construct_feed_dict(adj, features, support, y_train, train_mask, placeholders)
    cost_val = []

    start = time.time()
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # pdb.set_trace()
        feed_dict = construct_feed_dict(adj, features, support, y_train, train_mask, placeholders)
     
        # pdb.set_trace()
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        # cost, acc, duration = evaluate(adj, features, support, y_val, val_mask, placeholders)
        # cost_val.append(cost)
        # print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
        #       "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
        #       "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

        # if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        #     print("Early stopping...")
        #     break
    print("Total time for gcn {}".format(time.time()-start))
    print("Optimization Finished!")







    outputs = sess.run([tf.nn.softmax(model.outputs)], feed_dict=feed_dict)[0]

    # pdb.set_trace()

    X = np.ones((row*col)) *.5
    Xround = np.ones((row*col)) *.5
    X[vertices] = outputs[:,1]
    Xround[vertices] = np.round(X[vertices])
    if (Xround[vertices] == 0.).all():
        X[0] = Xround[0] = 1.
    X=X.reshape(row,col)
    Xround = Xround.reshape(row,col)


    path = np.ones((row*col))*0.25
    path[vertices] = .5
    path[grid_dict.get(source)] = 0.
    path[grid_dict.get(sink)] = 1.
    path=path.reshape(row,col)

    # pdb.set_trace()


    A=map(grid_dict.get,cut[1][0])
    B=map(grid_dict.get,cut[1][1])
    view_cut=np.ones((row*col))*0.5
    view_cut[A] = 0
    view_cut[B] = 1
    view_cut=view_cut.reshape(row,col)

    # pdb.set_trace()

    if featplot is not None:
        fig, ax = plt.subplots(5,1)
        ax[0].imshow(path, interpolation='nearest')
        ax[1].imshow(featplot.reshape(row,col), interpolation='nearest')
        ax[2].imshow(Xround, interpolation='nearest')
        ax[3].imshow(X, interpolation='nearest')
        ax[4].imshow(view_cut, interpolation='nearest')
        
    else:
        fig, ax = plt.subplots(4,1)
        ax[0].imshow(path, interpolation='nearest')
        ax[1].imshow(Xround, interpolation='nearest')
        ax[2].imshow(X, interpolation='nearest')
        ax[3].imshow(view_cut, interpolation='nearest')



    plt.show()

    # directory = "{}_{}/".format(FLAGS.app,row*col)
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # plt.savefig("{}_seed{}.png".format(directory,seed))




    plt.close()


