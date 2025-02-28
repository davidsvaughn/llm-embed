"""
Embedding layer aggregation utilities
"""

import numpy as np

def select_layer_indices(M, N, method=1, seed=1234):
    """
    Embedding Aggregation Helper Function
    - Implements multiple algorithms for randomly selecting layer indices
    - Based on random seed, so every call with same seed will return same layer indices selections
    - M is the number of layers, N is the number of embeddings dimensions in each layer
    
    Args:
        M (int): Number of layers.
        N (int): Number of embedding dimensions in each layer.
        method (int): Method for selecting layer indices. Possible values are:
            0 - Linear probability distribution.
            1 - Uniform random selection.
            2 - Fixed number of random indices per layer.
            3 - Combination of uniform random selection and fixed number of random indices.
        seed (int): Random seed for reproducibility.
    
    Returns:
        list: List of selected layer indices.
    """
    np.random.seed(seed)
    
    if method == 0:
        p = np.linspace(1, min(5, M), M)
        p /= np.sum(p)
        r = np.random.choice(M, size=N, p=p)
        I = [r == i for i in range(M)]
    
    elif method == 1:
        r = np.random.randint(0, M, size=N)
        I = [r == i for i in range(M)]
    
    elif method == 2:
        Q = 1.5
        I, s = [], int(Q * N // M)
        for i in range(M):
            p = np.random.permutation(N)
            I.append(p[:s])

    elif method == 3:
        r = np.random.RandomState(seed).randint(0, M, size=N)
        I, s = [], int(1 * N//M)
        for i in range(M):
            v = np.where(r==i)[0]
            p = np.random.RandomState(seed+i).permutation(N)[:s]
            p = np.unique(np.concatenate((v,p)))
            I.append(p)
            
    return I

def aggregate_layers(L, **kwargs):
    """
    Embedding Aggregation Functions
    - Takes a dictionary of layer embeddings and aggregates them
    - Outputs a single numpy array of embeddings
    """
    X = list(L.values())
    M, N = len(X), X[0].shape[1]
    I = select_layer_indices(M, N, **kwargs)
    return np.concatenate([x[:, i] for x, i in zip(X, I)], axis=1)
