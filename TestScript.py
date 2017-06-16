"""
Script to test the proposed clustering method.
"""
#######################################################################################################
## Imports
#######################################################################################################

import RandomDataGenerators as rng
import CoreFunctions as cf
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans


#######################################################################################################
## First experiment: data generated as a mixture of independent bernoulli
#######################################################################################################

N = 10000           #N is the sample size
n = 200             #n is the number of features
k = 20              #k is the number of clusters

#Random data generation. X is the dataset, TrueCL the true clusters
X, TrueCL  = rng.generate_sample(N,n,k, distr='mixtbin')

#Run the proposed clustering algorithm
CL = cf.NaiveBayesClustering(X,k)

# Print the clustering accuracy compared with that of k-means
print "MOM Accuracy:", adjusted_rand_score(TrueCL,CL)
print "KMeans Accuracy:",  adjusted_rand_score(KMeans(k).fit(X).labels_,CL)


#######################################################################################################
## Second experiment: data generated as a mixture with Poisson AND Bernoulli outcomes (mixed data type)
#######################################################################################################

N = 10000           #N is the sample size
n1 = 200            #n1 is the number of Poisson features
n2 = 10            #n2 is the number of Bernoulli features
k = 36              #k is the number of clusters

#Random data generation. X is the dataset, TrueCL the true clusters
X,TrueCL  = rng.generate_sample_mixed(N,n1,n2,k)

#Run the proposed clustering algorithm
CL = cf.NaiveBayesClustering(X,k)

# Print the clustering accuracy compared with that of k-means
print "MOM Accuracy:", adjusted_rand_score(TrueCL,CL)
print "KMeans Accuracy:",  adjusted_rand_score(KMeans(k).fit(X).labels_,CL)