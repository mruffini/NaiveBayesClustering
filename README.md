# NaiveBayseClustering

Python implementation of the algorithms to perform Naive Bayes Clustering using the three views approach.

Based on SVTD, described here

    https://arxiv.org/abs/1612.03409

## Content of the project:

### RandomDataGenerator.py

A module that allows the user to generate synthetic Naive Bayes Models with:
- Bernoulli outcomes
- Poisson outcomes
- Mixed Bernoulli - Poisson outcomes

### CoreFunctions.py

A module that contains the core functions to perform clustering


### TestScript.py

A script to generate synthetic Naive Bayes Models using the module "RandomGenerator", perform clustering
 using the module "CoreFunctions" and compare the results with k-means.
