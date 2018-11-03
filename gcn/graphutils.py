import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import matplotlib.pyplot as plt
import pdb




def get_source_sink(source,sink,feats,vertices,graph_dict):
    if source == -1 or sink == -1:
        if feats.shape == np.eye(feats.shape[0]).shape and (feats == np.eye(feats.shape[0])).all():
            choices = np.random.choice(len(vertices),2,replace=False)
            source=choices[0]
            sink = choices[1]     
        else:
            source = np.argmin(feats[:,1])
            sink = np.argmax(feats[:,1])         
    else:   
        source = graph_dict.get(source)
        sink = graph_dict.get(sink)

    return source, sink

def get_splits(y,source,sink,other_sources,other_sinks):
    pdb.set_trace()
    idx_train = [source,sink]
    # other_sinks = other_sinks[-2:]
    if len(other_sinks):
        idx_train += other_sinks
    if len(other_sources):
        idx_train += other_sources
    # pdb.set_trace()
    print([source] + other_sources)
    print([sink] + other_sinks)

    # my_idx= [185,186]
    # idx_train += my_idx

    # else:
    #     idx_train = [source,sink]
    #     print(idx_train)
    idx_val = range(len(y))

    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)

    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]

    train_mask = sample_mask(idx_train, y.shape[0])
    val_mask = sample_mask(idx_val, y.shape[0])

    return y_train, y_val, train_mask, val_mask


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    # pdb.set_trace()
    # val=10
    # for i in range(5):
    #     adj[739-i,740-i] =val
    # adj[739,740] = val
    # adj[779,780] = val
    # adj = np.triu(adj.toarray(),k=1) + np.triu(adj.toarray(),k=1).T
    adj = sp.csr_matrix(adj)
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj_normalized= adj + sp.eye(adj.shape[0])
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(adj,features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""

    feed_dict = dict()

    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj})
    # pdb.set_trace()
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    # feed_dict.update({placeholders['learning_rate']: 0.01})
    return feed_dict



def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    # pdb.set_trace()
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    # pdb.set_trace()
    return features
    # return sparse_to_tuple(features),features


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # pdb.set_trace()
    # d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
    # a_norm = adj.dot(d).transpose().dot(d).toarray()

    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot
