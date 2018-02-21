"""
Module to generate synthetic data.
"""

#######################################################################
## Imports
#######################################################################
import numpy as np
#######################################################################
## Data generators
#######################################################################


def generate_sample(N, n, k):
    """
    Generates a sample of N synthetic samples dustributed as a NBM with binary outcomes
    Returns the generated samples X, with N rows an n columns

    @param N: The number of synthetic samples
    @param n: the dimension of the mixture
    @param k: the number of components
    """

    #pi are the mixing weights
    pi = np.random.uniform(0, 1, k)
    pi = pi / np.sum(pi)

    #x are the cluster assignments
    x = np.random.multinomial(1, pi, N)
    x = np.argmax(x, 1)

    #M are the centers of the mixture
    M = np.random.exponential(10, (n, k))

    #X is the synthetic dataset
    X = np.zeros((N, n))

    #According to the selected distribution we generate the data
    M = M / (1 + M.max())
    for i in range(k):
        X[x == i, :] = np.random.binomial(1, M[:, i], [int(sum(x == i)), n])

    return X.astype(float), x

