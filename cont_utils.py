import numpy as np
import scipy.sparse as sp
import pdb


def get_splits(y,source,sink,other_sources,other_sinks):
    # pdb.set_trace()

    idx_train = [source,sink]

    if len(other_sinks):
        idx_train += other_sinks
    if len(other_sources):
        idx_train += other_sources


    print([source] + other_sources)
    print([sink] + other_sinks)


    idx_val = range(len(y))

    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)

    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]

    train_mask = sample_mask(idx_train, y.shape[0])
    val_mask = sample_mask(idx_val, y.shape[0])

    return y_train, y_val, train_mask, val_mask



def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)



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


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot




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


    



def make_interpolater(left_min, left_max, right_min, right_max): 
    if left_min == left_max:
        return lambda x:x
    # Figure out how 'wide' each range is  
    leftSpan = left_max - left_min  
    rightSpan = right_max - right_min  

    # Compute the scale factor between left and right values 
    scaleFactor = float(rightSpan) / float(leftSpan) 

    # create interpolation function using pre-calculated scaleFactor
    def interp_fn(value):
        return right_min + (value-left_min)*scaleFactor

    return interp_fn
    