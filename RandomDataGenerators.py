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


def generate_sample(N, n, k, distr='mixtbin'):
    """
    Generates a sample of N synthetic samples dustributed as a NBM
    Returns the generated samples X, with N rows an n columns

    @param N: The number of synthetic samples
    @param n: the dimension of the mixture
    @param k: the number of components
    @param distr: the data type: mixtbin --> Bernoulli, poiss--> Poisson
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
    if distr == 'poiss':
        X = np.random.poisson(np.transpose(M[:, x]), [N, n])
    elif distr == 'mixtbin':
        M = M / (1 + M.max())
        for i in range(k):
            X[x == i, :] = np.random.binomial(1, M[:, i], [int(sum(x == i)), n])

    return X.astype(float), x


def generate_sample_mixed(N, n1,n2, k):
    """
    Generates a sample of N synthetic samples dustributed as a NBM with mixed data type
    Returns the generated samples X, with N rows an n columns

    @param N: The number of synthetic samples
    @param n1: the number of Poisson features
    @param n2: the number of Bernoulli features
    @param k: the number of components
    """

    #pi are the mixing weights
    pi = np.random.uniform(0, 1, k)
    pi = pi / np.sum(pi)

    #x are the cluster assignments
    x = np.random.multinomial(1, pi, N)
    x = np.argmax(x, 1)

    #the total size of the mixture
    n = n1+n2

    #X is the synthetic dataset
    X = np.zeros((N, n))

    #M1 are the centers of the mixture of the Poisson data
    M1 = np.random.exponential(10, (n1, k))

    X[:,:n1] = np.random.poisson(np.transpose(M1[:, x]), [N, n1])

    #M2 are the centers of the mixture of the Bernoulli data
    M2 = np.random.exponential(10, (n2, k))
    M2 = M2 / (1 + M2.max())

    for i in range(k):
        X[x == i, n1:] = np.random.binomial(1, M2[:, i], [int(sum(x == i)), n2])

    #Shuffle the columns
    vidx = np.arange(n)
    np.random.shuffle(vidx)
    X = X[:,vidx]

    return X.astype(float),x
