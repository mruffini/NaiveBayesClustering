"""
Core functions to perform clustering
"""

#######################################################################
## Imports
#######################################################################
import numpy as np
#######################################################################
## Learning functions
#######################################################################


def NaiveBayesClustering(X,k):
    """
    Performs clustering of the dataset
    @param X: the dataset
    @param k: the number of clusters
    """

    #First calculate the parameters of the model
    M,P = LearnNBM(X,k)

    #Use the parameters to perform naive bayes clustering
    CL = PerformClustering(M,X)

    return CL

def LearnNBM(X, k):
    """
    Returns the parameters of the NBM generating the data
    @param X: the dataset
    @param k: the number of clusters
    """

    N, n = X.shape
    E = np.mean(X,0)
    ML = np.zeros([n,k])

    n1 = int(n/3)
    n2 = int(2*n/3)

    X1 = X[:,:n1]
    X2 = X[:,n1:n2]
    X3 = X[:,n2:]

    M2 = np.transpose(X).dot(X) / N
    M2[range(n),range(n)] = Get_DiagM2(X1,X2,X3,k)

    u, s, v = np.linalg.svd(M2)
    u = u[:,:k]
    s = s[:k]
    pu1 = np.linalg.pinv((u.dot(np.diag(np.sqrt(s))))[:n1,:])
    pu2 = np.linalg.pinv((u.dot(np.diag(np.sqrt(s))))[n1:n2,:])
    pu3 = np.linalg.pinv((u.dot(np.diag(np.sqrt(s))))[n2:,:])

    H, O, HMin = Get_H(X3, X2, X1, k, pu1, H = [])
    H, O, HMin = Get_H(X1, X3, X2, k, pu2, H = H, HMin = HMin, O = O)
    H, O, HMin = Get_H(X1, X2, X3, k, pu3, H = H, HMin = HMin, O = O)

    for i in range(0, len(H)):
        s = np.diag(np.transpose(O).dot(H[i]).dot(O))
        ML[i,:] = s

    POF = np.linalg.pinv(ML).dot(E)

    POF = POF / sum(POF)

    return ML, POF

def Get_DiagM2(X1, X2, X3, k):
    """
    From X1, X2 and X3, gets the diagonal of M2
    @param X1: the first view
    @param X2: the second view
    @param X3: the first view
    @param k: the number of clusters
    """

    D1 = GetPartial_M2(X3, X2, X1, k)
    D2 = GetPartial_M2(X1, X3, X2, k)
    D3 = GetPartial_M2(X1, X2, X3, k)

    return np.concatenate((np.diag(D1),np.diag(D2),np.diag(D3)))

def Get_H(X1, X2, X3, k, pu, H= [], HMin = 0, O = 1):
    """
    Get the whitened slices of M3
    @param X1: the first view
    @param X2: the second view
    @param X3: the first view
    @param k: the number of clusters
    @param H: the whitened slices of M3
    @param HMin: the smallest eigengap
    @param O: the optimal rotation
    """

    N, n = X3.shape

    P12 = X1.T.dot(X2) / N
    u, s, v = np.linalg.svd(P12)
    u = u[:, :k]
    s = s[:k]
    v = v[:k,:]
    A = np.diag(1 / s ** 0.5).dot(u.T)
    B = (v.T.dot(np.diag(1 / s ** 0.5))).T

    X1A = X1.dot(A.T)
    X2B = X2.dot(B.T)

    zX3 = X3.dot(pu.T)

    C13 = (X1A).T.dot(zX3) / N
    C12 = np.linalg.pinv((X1A).T.dot(X2B) / N)
    C23 = (X2B).T.dot(zX3) / N

    for i in range(n):
        Y = X3[:, i].reshape((N, 1))
        D = (X1A * Y).T.dot(X2B) / N
        wH = C23.T.dot(C12).dot(D).dot(C12).dot(C13)
        H.append(wH)
        h,s,v = np.linalg.svd(wH)
        if np.min(-np.diff(s)) > HMin:
            HMin = np.min(-np.diff(s))
            O = h

    return H, O, HMin


            #P12 = X1.T.dot(X2) / N
    #u, s, v = randomized_svd(P12, n_components=k, n_iter=5, random_state=None)

def GetPartial_M2(X1, X2, X3, k):
    """
    From X1, X2 and X3, gets the submatrix of M23
    @param X1: the first view
    @param X2: the second view
    @param X3: the first view
    @param k: the number of clusters
    """

    N1,n1 = X1.shape

    N = N1

    P12 = X1.T.dot(X2) / N
    u, s, v = np.linalg.svd(P12)
    u = u[:, :k]
    s = s[:k]
    v = v[:k,:]
    A = np.diag(1 / s ** 0.5).dot(u.T)
    B = (v.T.dot(np.diag(1 / s ** 0.5))).T

    X1A = X1.dot(A.T)
    X2B = X2.dot(B.T)

    C13 = (X1A).T.dot(X3)/N
    C12 = (X1A).T.dot(X2B)/N
    C23 = (X2B).T.dot(X3)/N

    M2 = C13.T.dot(np.linalg.pinv(C12.T)).dot(C23)

    return M2

def PerformClustering(M, X):
    """
    Clusters the rows of X accordng to the similarity to the cols of M
    @param M: the centers of the mixture
    @param X: The dataset
    """
    N,n = X.shape
    n,k = M.shape

    Dist = np.ones((N,k))
    Dist[:] = np.inf

    #Calculates the distance of each row to each center of the mixture
    for i in range(k):
        Dist[:,i] = ((X-M[:,i].reshape(1,n))**2).sum(1)

    #Calculates the distance of each row to each center of the mixture
    CL = np.argmin(Dist,1)

    return CL