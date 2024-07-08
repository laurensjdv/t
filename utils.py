import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from scipy.sparse import coo_matrix

def balanced_random_split(dataset, lengths):
    """
    Args:
      dataset: A torch.utils.data.Dataset instance.
      lengths: A list of ints, specifying the lengths of the splits to make.
    Returns:
      A list of torch.utils.data.SubsetRandomSampler instances, one for each split.
    """
    # Get the number of items in the dataset
    n = len(dataset)

    # get index of first instance of class 1
    m = int(n/2)

    class_0_idx = list(range(m))
    class_1_idx = list(range(m, n))
    # Create a list of indices for the dataset

    c0_lengths = [int(l/2) for l in lengths]
    c1_lengths = [l - c0 for l, c0 in zip(lengths, c0_lengths)]
    c0_split_indices = [class_0_idx[i:i + l] for i, l in enumerate(c0_lengths)]
    c1_split_indices = [class_1_idx[i:i + l] for i, l in enumerate(c1_lengths)]



    # combine the indices 
    split_indices = [c0 + c1 for c0, c1 in zip(c0_split_indices, c1_split_indices)]

    # shuffle the indices
    for i in range(len(split_indices)):
        np.random.shuffle(split_indices[i])

    # create a list of subset samplers
    samplers = [Subset(dataset, indices) for indices in split_indices]

    return samplers

# def balanced_random_split_v2(dataset, subset_lengths, num_classes=4):
#     """
#     Args:
#       dataset: A torch.utils.data.Dataset instance, assumed to have an equally balanced class distribution.
#       lengths: A list of ints, specifying the lengths of the splits to make.
#       num_classes: The number of classes in the dataset.
#     Returns:
#       A list of torch.utils.data.SubsetRandomSampler instances, one for each split.
#     """
#     n = len(dataset)

#     print(f"Dataset length: {n}")
    
#     n_per_class = int(n / num_classes)
#     indices = list(range(n))

#     class_indices = []
#     for i in range(num_classes):
#         class_indices.append([idx for idx in indices if dataset[idx][1] == i])

        
#     for i in range(num_classes):
#         np.random.shuffle(class_indices[i])

#     subsets_idxs = []
#     start_idx = 0
#     for l in subset_lengths:
#         l_per_class = int(l / num_classes)
#         subset_idx = []
#         for i in range(num_classes):
#             subset_idx.extend(class_indices[i][start_idx:start_idx + l_per_class])

#         start_idx += l_per_class
#         subsets_idxs.append(subset_idx)
            
#     # calculate overlap between class_indices lists
#     overlap = [len(set(class_indices[i]) & set(class_indices[i+1])) for i in range(len(class_indices) - 1)]
#     print(overlap)

#     # calculate overlap between subeset_idxs lists
#     overlap = [len(set(subsets_idxs[i]) & set(subsets_idxs[i+1])) for i in range(len(subsets_idxs) - 1)]
#     print(f"Overlap: {overlap}")

#     # len of subse idxs
#     print(f"Length of subset idxs: {[len(subset) for subset in subsets_idxs]}")
#     samplers = [Subset(dataset, indices) for indices in subsets_idxs]

#     return samplers

def balanced_random_split_v2(dataset, subset_lengths, num_classes=4):
    """
    Args:
      dataset: A torch.utils.data.Dataset instance, assumed to have an equally balanced class distribution.
      lengths: A list of ints, specifying the lengths of the splits to make.
      num_classes: The number of classes in the dataset.
    Returns:
      A list of torch.utils.data.SubsetRandomSampler instances, one for each split.
    """
    n = len(dataset)
    
    n_per_class = int(n / num_classes)
    indices = list(range(n))

    class_indices = []
    for i in range(num_classes):
        if dataset.__class__.__name__ == 'GraphDataset':
            class_indices.append([idx for idx in indices if dataset[idx].y == i])
        else:
            class_indices.append([idx for idx in indices if dataset[idx][1] == i])

    print(f"Length of class indices: {[len(class_indices[i]) for i in range(num_classes)]}")

        
    for i in range(num_classes):
        np.random.shuffle(class_indices[i])

    subsets_idxs = []
    start_idx = 0
    for l in subset_lengths:
        l_per_class = int(l / num_classes)
        subset_idx = []
        for i in range(num_classes):
            subset_idx.extend(class_indices[i][start_idx:start_idx + l_per_class])

        start_idx += l_per_class
        subsets_idxs.append(subset_idx)

    print(f"Length of subset idxs: {[len(subset) for subset in subsets_idxs]}")
            
    # calculate overlap between class_indices lists
    overlap = [len(set(class_indices[i]) & set(class_indices[i+1])) for i in range(len(class_indices) - 1)]

    # calculate overlap between subeset_idxs lists
    overlap = [len(set(subsets_idxs[i]) & set(subsets_idxs[i+1])) for i in range(len(subsets_idxs) - 1)]



    samplers = [Subset(dataset, indices) for indices in subsets_idxs]

    return samplers

def compute_KNN_graph(matrix, k_degree=10):
    """ Calculate the adjacency matrix from the connectivity matrix."""

    matrix = np.abs(matrix)
    idx = np.argsort(-matrix)[:, 0:k_degree]
    matrix.sort()
    matrix = matrix[:, ::-1]
    matrix = matrix[:, 0:k_degree]

    A = adjacency(matrix, idx).astype(np.float32)

    return A

def fc_to_matrix(fc, D):
    sq_fc = np.zeros((D,D))
    sq_fc[np.triu_indices(D,1)] = fc
    sq_fc += sq_fc.T
    return sq_fc

def adjacency(dist, idx):

    m, k = dist.shape
    assert m, k == idx.shape
    assert dist.min() >= 0

    # Weight matrix.
    I = np.arange(0, m).repeat(k)
    J = idx.reshape(m * k)
    V = dist.reshape(m * k)
    W = coo_matrix((V, (I, J)), shape=(m, m))

    # No self-connections.
    W.setdiag(0)

    # Non-directed graph.
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)

    return W.todense()
