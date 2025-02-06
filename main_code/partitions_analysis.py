import numpy as np
from scipy.stats import entropy

def compute_entropy(labels, base=None, label_format="list"):
    if label_format == "list":
        counts = []
        for cluster in labels:
            counts.append(len(cluster))
        return entropy(counts, base=base)
    if label_format == "array":
        _, counts = np.unique(labels, return_counts=True)
        return entropy(counts, base=base)  