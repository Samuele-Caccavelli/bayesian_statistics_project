import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_t
from scipy.stats import entropy
import random
import copy
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering

def compute_entropy(labels, base=None, label_format="list"):
    if label_format == "list":
        counts = []
        for cluster in labels:
            counts.append(len(cluster))
        return entropy(counts, base=base)
    if label_format == "array":
        _, counts = np.unique(labels, return_counts=True)
        return entropy(counts, base=base)

  

def sampled_sim_matrix(MC_part, n_obs):
    """
    Compute A, the similarity matrix of dimensions n_obs * n_obs, where A[i,j] is the frequency for which the observations
    i and j are assigned to the same cluster.

    Args:
        MC_part: list of lists, MCMC sampling of partitions. Each partition is a list of clusters, 
                 and each cluster is a list of observation indices.
        n_obs: int, total number of observations.

    Returns:
        A: np.ndarray, similarity matrix of size (n_obs, n_obs).
    """
    
    A = np.zeros((n_obs, n_obs), dtype=float)
    n_sample = len(MC_part)

    for clusters in MC_part:
        for cluster in clusters:
            for k, i in enumerate(cluster):
                for j in cluster[k:]:
                    if i != j:  # Avoid double increment for diagonal
                        A[i, j] += 1
                        A[j, i] += 1

    # Normalize by the number of samples
    A /= n_sample

    # Ensure the diagonal is 1 (observations are always in the same cluster with themselves)
    np.fill_diagonal(A, 1.0)

    return A


def binder_loss_label_format(labels, S, alpha=1.0, beta=1.0):
    """
    Compute the Binder loss for a clustering in labels format and a similarity matrix.

    Parameters:
    - labels: array of size N
        labels[i] = cluster allocation for observation i
    - S: array-like, shape (N, N)
        Posterior similarity matrix (symmetric).
    - alpha: float
        Weight for within-cluster disagreements.
    - beta: float
        Weight for between-cluster disagreements.

    Returns:
    - loss: float
        The Binder loss value.
    """
    loss = 0.0
    N = S.shape[0]
    for i in range(N):
        for j in range(i + 1, N):
            same_cluster = labels[i] == labels[j]
            loss += alpha * same_cluster * (1 - S[i, j]) + beta * (not same_cluster) * S[i, j]
    return loss

def binder_loss_list_format(clustering, S, alpha=1.0, beta=1.0):
    """
    Compute the Binder loss for a clustering in list-of-lists format and a similarity matrix.

    Parameters:
    - clustering: list of lists
        Each sublist contains the indices of data points in the same cluster.
    - S: array-like, shape (N, N)
        Posterior similarity matrix (symmetric).
    - alpha: float
        Weight for within-cluster disagreements.
    - beta: float
        Weight for between-cluster disagreements.

    Returns:
    - loss: float
        The Binder loss value.
    """
    # Convert clustering to label format
    N = S.shape[0]
    labels = np.zeros(N, dtype=int)
    for cluster_id, indices in enumerate(clustering):
        for index in indices:
            labels[index] = cluster_id
    return binder_loss_label_format(labels, S)


def clusters_from_matrix(S, loss=binder_loss_label_format, max_clusters=None):
    """
    Finds the best clustering using the similarity matrix by comparing the binder loss of the clustering obtained with AgglomerativeClustering for different number of clusters.

    Args:
        S: array (N, N), similarity matrix of the observation
        loss: function(labels, S), loss function to use to compare clusterings
        max_clusters: int, maximum number of cluster. Defaults as N // 3
    Returns:
        best_clustering: array(N), best clustering in labels format
        best_n_clusters: int, number of clusters in the best clustering
        scores: loss for each number of clusters. scores[i] = loss for i+1 clusters
    """
    if max_clusters is None:
        max_clusters = S.shape[0] // 3  # reasonable assumption for the maximum number of clusters

    affinity_matrix = 1-S

    best_clustering = None
    best_n_clusters = None
    min_loss = np.infty # initialize loss at infinity
    scores = np.zeros(max_clusters)

    for n_clusters in range(1, max_clusters): # iterate on all possible number of clusters
        # compute the best clustering for this number of clusters based on the similarity matrix
        result = AgglomerativeClustering(
        affinity='precomputed',
        n_clusters=n_clusters,
        linkage='complete'
        ).fit(affinity_matrix)
        
        # compute the loss for the obtained clustering
        scores[n_clusters] = loss(result.labels_)
        
        # check if this clustering is the best so far
        if scores[n_clusters] < min_loss:
            best_clustering = copy.deepcopy(result.labels_)
            best_n_clusters = n_clusters
            min_loss = scores[n_clusters]

    return best_clustering, best_n_clusters, scores[1:]

