import matplotlib.pyplot as plt
import numpy as np

def plot_clusters(data, optimal_clustering, title, xlabel="X1", ylabel="X2"):
    """
    Plots clusters based on the given clustering information.

    Parameters:
        data (np.ndarray): The dataset, where rows are observations and columns are features.
        optimal_clustering (list of lists): The clustering information as a list of lists, 
                                            where each sublist contains indices of points in a cluster.
        title (str): The title of the graph.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
    """
    colors = plt.cm.rainbow(np.linspace(0, 1, len(optimal_clustering)))
    plt.figure(figsize=(8, 6))

    for cluster_id, observations in enumerate(optimal_clustering):
        cluster_data = data[observations]
        plt.scatter(cluster_data[:, 0], 
                    cluster_data[:, 1], 
                    label=f"Cluster {cluster_id}", 
                    color=colors[cluster_id],
                    s=100,
                    alpha=0.8,
                    edgecolor="k")

    # Add labels, legend, and grid
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show()
