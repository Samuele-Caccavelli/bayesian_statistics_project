import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import random
import copy
from scipy.stats import multivariate_t
from scipy.spatial.distance import mahalanobis
from main_code.partitions_analysis import compute_entropy, sampled_sim_matrix


class Neal_3:
    """
    Class for Neal Algorithm 3
    """

    def __init__(self, alpha=0.1, lamb_0=1):
        
        # Initialize attributes

        # These use default values that can be changed
        self.alpha = alpha
        self.lamb_0 = lamb_0

        # These require data format to initialize
        self.Y = None
        self.D = None
        self.nu_0 = None  # Should be > D
        self.mu_0 = None  # will default to mean of data
        self.inv_scale_mat_0 = None # requires D

        # attributes for computed when fitting
        self.history = None
        self.similatity_matrix = None
        
        self.metrics = {"entropy":[],
                        "binder loss":[],
                        "n clusters":[]}
        return 


    # Model hyper-parameter Functions -- Can be changed to see what works best
    def compute_nu_0(self):
        """
        Computes and set the value for nu_0 depending on the number of dimensions D of Y
        """
        if self.D is None:
            raise ValueError("No dimension D provided to compute nu_0")

        self.nu_0 = self.D + 3
    
    def compute_mu_0(self):
        """
        Computes and set the value for mu_0 as the mean of the data
        """
        if self.Y is None:
            raise ValueError("No data Y provided to compute mu_0")
        
        self.mu_0 = np.mean(self.Y, axis=0)

    def compute_inv_scale_mat_0(self):
        """
        Computes the inverse scale matrix hyper parameter for the NIW.
        Defaults as the identity, might need to be changed (TODO?)
        """
        self.inv_scale_mat_0 = np.eye(self.D)


    # Integrals based on data distribution assumptions -- /!\ to change if assumptions change /!\
    def integral_func_1(self, cluster, i):
        """
        Compute the first integral using student_t distribution based on Murphy (2007) parameters
        """
        n = len(cluster)    # number of element currently in cluster (used n to be consistent with Murphy (2007) notation)
        
        cluster_Y = self.Y[np.isin(np.arange(self.n_obs), cluster)]
        cluster_mean = np.mean(cluster_Y, axis=0)

        # based on Murphy (2007). These should not be changed
        mu_n = (self.lamb_0 * self.mu_0 + n * cluster_mean) / (self.lamb_0 + n)
        lamb_n = self.lamb_0 + n
        nu_n = self.nu_0 + n
        
        # compute scatter matrix
        S = np.zeros((self.D,self.D))
        for j in range(n):
            temp = self.Y[j] - cluster_mean
            S += np.outer(temp, temp)
        temp = cluster_mean - self.mu_0
        inv_scale_mat_n = self.inv_scale_mat_0 + S + ((self.lamb_0 * n) / (self.lamb_0 + n)) * np.outer(temp, temp)

        # Computes integral using pdf of student t
        student_df = nu_n - self.D + 1
        integral = multivariate_t.pdf(self.Y[i],
                                    mu_n,
                                    inv_scale_mat_n * ((lamb_n+1) / (lamb_n * student_df)),
                                    student_df)
        return integral
    
    def integral_func_2(self, i):
        """
        Compute the second integral using student_t distribution based on Murphy (2007) parameters
        """
        student_df = self.nu_0 - self.D + 1
        # Computes integral using pdf of student t
        integral = multivariate_t.pdf(self.Y[i],
                                    self.mu_0,
                                    self.inv_scale_mat_0 * ((self.lamb_0 + 1) / (self.lamb_0 * student_df)),
                                    student_df)
        return integral


    # Functions for Neal Algorithm 3  -- Should not be changed (create another class to implement another algorithm)
    def cluster_probabilities(self, i, clusters):
        """
        Computes the probabilities of observation i joining each clusters or creating a new one. The output is not sctictly probabilities but weights.
        
        Args:
            i: int, index of the observation to add to the clustering
            clusters: list of list, current partition
        
        Returns:
            probabilities array, probabilities[j] = probability of joining cluster j. probabilities[-1] = probability of creating a new cluster
        """
        n_clusters = len(clusters)
        probabilities = np.zeros(n_clusters+1)

        # probabilities of joining existing cluster
        for c in range(n_clusters):
            probabilities[c] = self.integral_func_1(clusters[c], i)
            probabilities[c] *= (len(clusters[c]) / (self.n_obs - 1 + self.alpha))

        # probability of creating new cluster
        probabilities[-1] = self.integral_func_2(i)
        probabilities[-1] *= self.alpha / (self.n_obs - 1 + self.alpha)

        return probabilities

    def fit(self, Y, n_steps, metrics=[]):
        """
        Performs a markov chain using algorithm 3 from Neal (2000).
        Main function of this class

        Args:
            Y: 2D array, of observations. Each observation is a vetor of size D. So Y is of shape (n_observation, D)
            n_steps: int, number of step of the markov chain to do (one step is defined as randomly moving each observation 1 time)
            metrics: list of strings, names of the metrics to compute at runtime (e.g. for the traceplots)
        
        Returns:
            history, list of partitions
        """

        # Set basic attributes
        self.Y = Y
        self.n_obs = len(Y)
        self.D = Y.shape[1]
        self.compute_mu_0()
        self.compute_inv_scale_mat_0()
        self.compute_nu_0()

        # Initialize clusters
        clusters = [[i] for i in range(self.n_obs)]

        self.history = [copy.deepcopy(clusters)]

        # update_metrics
        self.update_metrics(metrics, clusters)

        # Initialize progress bar
        progress_bar = tqdm(total=n_steps, desc="MCMC Progress", unit="step")

        for step in range(n_steps):  # Markov chain
            for i in range(self.n_obs):  # 1 step of the Markov chain
                # 1. Find in which cluster the observation is
                c = 0
                for index in range(len(clusters)):  # one step 
                    if i in clusters[index]:
                        c = index
                        break
                # 2. Remove observation i from clusters:
                if len(clusters[c]) == 1:  # case 1: i is the only element of the cluster -> remove cluster
                    del clusters[c]
                else:  # case 2: cluster has more than 1 element -> remove i from the cluster
                    clusters[c].remove(i)

                # 3. Compute probabilities of adding i to each cluster
                weights = self.cluster_probabilities(i, clusters)
                transitions = list(range(len(weights)))
                transition = random.choices(transitions, weights=weights)[0]

                # 4. Apply transition 
                if transition == len(clusters):  # add new cluster
                    clusters.append([i])
                else:
                    clusters[transition].append(i)
            
            # All elements have moved once -> one step of the Markov chain
            self.history.append(copy.deepcopy(clusters))
            
            # update_metrics
            self.update_metrics(metrics, clusters)

            # Update progress bar
            progress_bar.update(1)

        # Close progress bar
        progress_bar.close()

        return self.history
    

    # Funcions for metrics
    def update_metrics(self, metrics, clusters):
        """
        Updates metrics listed in "metrics". Used to compute metrics at runtime while computing the MCMC
        """
        if "entropy" in metrics:
            entropy = compute_entropy(clusters, label_format="list")
            self.metrics["entropy"].append(entropy)
        # possibly modify to use other metrics
        return


    # Post Processing functions :
    def compute_similarity_matrix(self, burn_in=0):
        """
        Computes the similarity matrix based on the history of the MCMC.

        Args:
            burn_in: int, specify from which iteration of the chain compute the similarity matrix. Default to 0

        Returns:
            array (n_obs, n_obs), similarity matrix
        
        Raises RuntimeError if the model has not been fitted to data
        """
        if self.history is None:
            raise RuntimeError("No MCMC history to compute the similarity matrix")
        A = np.zeros((self.n_obs, self.n_obs), dtype=float)
        n_samples = len(self.history)

        # Initialize progress bar
        progress_bar = tqdm(total=len(self.history[burn_in:]), desc="Similarity Matrix Progress", unit="step")

        for clusters in self.history[burn_in:]:
            for cluster in clusters:
                for k, i in enumerate(cluster):
                    for j in cluster[k:]:
                        if i != j:  # Avoid double increment for diagonal
                            A[i, j] += 1
                            A[j, i] += 1

            # Update progress bar
            progress_bar.update(1)

        # Close progress bar
        progress_bar.close()

        # Normalize by the number of samples
        A /= n_samples

        # Ensure the diagonal is 1 (observations are always in the same cluster with themselves)
        np.fill_diagonal(A, 1.0)

        # Both save the matrix and return it
        self.similatity_matrix = A
        return A
    
    def save(self, file_path):
        """
        Save this object to a file given by file_path
        """
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, file_path):
        """
        Loads an object from file given by file_path
        """
        with open(file_path, 'rb') as file:
            loaded_object = pickle.load(file)
        return loaded_object


class PPMx(Neal_3):
    """
    Extends Neal_3 class by including covariates
    """

    def __init__(self, alpha=0.1, lamb_0=1):
        super().__init__(alpha=alpha, lamb_0=lamb_0)

        # attributes specific to algorithm with covariates
        self.lambda_penalty = None
        self.X = None

    def compute_mahalanobis_penalty(self, cluster, i):

        # Combine current cluster observations and the new observation
        cluster_data = np.array([self.X[idx] for idx in cluster] + [self.X[i]])
        cluster_mean = np.mean(cluster_data, axis=0)
        cov_matrix = np.cov(cluster_data.T)

        penalty = mahalanobis(self.X[i], cluster_mean, cov_matrix)
        return penalty

    def cluster_probabilities(self, i, clusters):

        probabilities = super().cluster_probabilities(i, clusters)

        n_clusters = len(clusters)
        for c in range(n_clusters):
            penalty = self.compute_mahalanobis_penalty(clusters[c], i)
            probabilities[c] *= np.exp(-self.lambda_penalty * penalty)
        
        return probabilities
    
    def fit(self, Y, X, n_steps, lambda_penalty=0.1, metrics=[]):

        self.X = X
        self.lambda_penalty = lambda_penalty
        return super().fit(Y, n_steps, metrics=metrics)
    