#!/usr/bin/env python
# coding: utf-8

# In[54]:


import numpy as np
import random
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.manifold import SpectralEmbedding


# In[55]:


def pltPer(X, y, W):
    f = plt.figure()
        
    for n in range(len(y)):
            plt.plot(X[n,1],X[n,2],'b.')
            
    m, b = -W[1]/W[2], -W[0]/W[2]
    l = np.linspace(min(X[:,1]),max(X[:,1]))
    plt.plot(l, m*l+b, 'k-')
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("Linear Regression")


# In[56]:


def main():
    itlst = []
    for x in range(1):
        N = 2000
        
        dataset = genfromtxt('features.csv', delimiter=' ')
        y = dataset[:, 0]
        X = dataset[:, 1:]
        y[y!=4] = -1    #rest of numbers are negative class
        y[y==4] = +1    #number zero is the positive class
        
        y[y==0] = -1  # replace the zeros    
        X = np.append(np.ones((N,1)), X, 1)   # add a column of ones
        
        # linear regression
        Xs = np.linalg.pinv(X.T.dot(X)).dot(X.T)
        wIr = Xs.dot(y)
        pltPer(X,y,wIr)
        print("Our W is:")
        print(wIr)
        return


# In[57]:


main()


# In[ ]:




