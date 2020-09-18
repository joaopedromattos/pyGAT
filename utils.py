import numpy as np
import scipy.sparse as sp
import torch
import spektral as spk


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[
        i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(
        list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def new_load_data(path="./pyGAT/data/cora/", dataset='cora'):
    print(f"[LOAD DATA]: {dataset}")

    if (dataset == "cora" or dataset == 'citeseer' or dataset == 'pubmed'):
        adj, features, labels, train, val, test = spk.datasets.citation.load_data(
            dataset_name=dataset, normalize_features=True, random_split=True)
    elif (dataset == 'ppi' or dataset == 'reddit'):
        adj, features, labels, train, val, test = spk.datasets.graphsage.load_data(
            dataset_name=dataset, max_degree=1, normalize_features=True)
    else:
        raise ValueError(
            "Dataset not supported. List of supported datsets: ['cora', 'citeseer', 'pubmed', 'ppi', 'reddit']")

    if (dataset == "citeseer"):

        missing_labels = np.where(np.sum(labels, axis=1) == 0)

        # Deleting missing nodes' rows and columns

        adj = np.delete(, missing_labels, 0)
        adj = np.delete(adj, missing_labels, 1)

        print(adj.shape)

        # Deleting feature vectors of nodes with missing labels.
        features = np.delete(features, missing_labels, 0)

        train = np.delete(train, missing_labels, 0)
        val = np.delete(val, missing_labels, 0)
        test = np.delete(test, missing_labels, 0)

        # Removing nodes without labels...
        labels = np.delete(labels, missing_labels, 0)

    # Converting one-hot encoding into categorical
    # values with the indexes of each dataset partition
    idx_train, idx_val, idx_test = np.where(train)[0], np.where(val)[
        0], np.where(test)[0]

    # Normalizing our features and adjacency matrices
    # features = normalize_features(features)
    # print(adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    adj = torch.FloatTensor(adj.todense())
    features = torch.FloatTensor(features.todense())
    labels = torch.LongTensor(np.where(labels)[1])
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    print(f"Adjacency Matrix: \t {adj.shape}")
    print(f"Features Matrix: \t {features.shape}")
    print(f"Labels Matrix: \t {labels.shape}")
    print(f"Train Matrix: \t {idx_train.shape}")
    print(f"Validation Matrix: \t {idx_val.shape}")
    print(f"Test Matrix: \t {idx_test.shape}")

    return adj, features, labels, idx_train, idx_val, idx_test


def original_load_data(path="./pyGAT/data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Test {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt(
        "{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(
        "{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(
        labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # Normalizing our features and adjacency matrices
    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
