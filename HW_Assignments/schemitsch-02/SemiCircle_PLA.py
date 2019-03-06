#!/usr/bin/env python
# coding: utf-8

# In[37]:


from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.manifold import SpectralEmbedding

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs


# In[38]:


def pltPer(X, y, W):
    f = plt.figure()
    if X.shape[1] > 3:
        # reduce dimensions for display purposes
        Xw = np.append(X, np.reshape(W, (1, len(W))), 0)   # add a column of ones
        #Xs = TSNE(n_components=2, random_state=0, verbose=1).fit_transform(Xw)
        #Xs = Isomap(n_components=2).fit_transform(Xw)
        Xs = SpectralEmbedding(n_components=2).fit_transform(Xw)
        #Xs = PCA(n_components=2, random_state=0).fit_transform(Xw)

        Xs = np.append(np.ones((Xs.shape[0],1)), Xs, 1)   # add a column of ones
        X = Xs[:-1,:]
        W = Xs[-1,:]
        #print(W.shape)
        
    for n in range(len(y)):
        if y[n] == -1:
            plt.plot(X[n,1],X[n,2],'r*')
        else:
            plt.plot(X[n,1],X[n,2],'b.')
    m, b = -W[1]/W[2], -W[0]/W[2]
    l = np.linspace(min(X[:,1]),max(X[:,1]))
    plt.plot(l, m*l+b, 'k-')
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("Perceptron Learning Algorithm")


# In[39]:


def classification_error(w, X, y):
    err_cnt = 0
    N = len(X)
    for n in range(N):
        s = np.sign(w.T.dot(X[n])) # if this is zero, then :(
        if y[n] != s:
            err_cnt += 1
    print(err_cnt)
    return err_cnt

def choose_miscl_point(w, X, y):
    mispts = []
    # Choose a random point among the misclassified
    for n in range(len(X)):
        if np.sign(w.T.dot(X[n])) != y[n]:
            mispts.append((X[n], y[n]))
    #print(len(mispts))
    return mispts[random.randrange(0,len(mispts))]


# In[40]:


def make_semi_circles(n_samples=2000, thk=5, rad=10, sep=5, plot=True):
    """Make two semicircles circles
    A simple toy dataset to visualize classification algorithms.
    Parameters
    ----------
    n_samples : int, optional (default=2000)
        The total number of points generated.
    thk : int, optional (default=5)
        Thickness of the semi circles.
    rad : int, optional (default=10)
        Radious of the circle.
    sep : int, optional (default=5)
        Separation between circles.
    plot : boolean, optional (default=True)
        Whether to plot the data.
    Returns
    -------
    X : array of shape [n_samples, 2]
        The generated samples.
    y : array of shape [n_samples]
        The integer labels (-1 or 1) for class membership of each sample.
    """

    
    noisey = np.random.uniform(low=-thk/100.0, high=thk/100.0, size=(n_samples // 2))
    
    noisex = np.random.uniform(low=-rad/100.0, high=rad/100.0, size=(n_samples // 2))
    
    separation = np.ones(n_samples // 2)*((-sep*0.1)-0.6)
    
    
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out
    
    # generator = check_random_state(random_state)
    
    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out)) + noisex
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out)) + noisey
    inner_circ_x = (1 - np.cos(np.linspace(0, np.pi, n_samples_in))) + noisex
    inner_circ_y = (1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - .5) + noisey + separation
    
    X = np.vstack((np.append(outer_circ_x, inner_circ_x),
                   np.append(outer_circ_y, inner_circ_y))).T
    y = np.hstack([np.ones(n_samples_in, dtype=np.intp)*-1,
                   np.ones(n_samples_out, dtype=np.intp)])
    
    if plot:
        plt.plot(outer_circ_x, outer_circ_y, 'r.')
        plt.plot(inner_circ_x, inner_circ_y, 'b.')
        plt.show()
        
    return X, y


# In[43]:


def main():
    itlst = []
    for x in range(1):
        N = 2000
        
        X, y = make_semi_circles(n_samples=N, thk=5, rad=10, sep=5, plot=True)

        print(X)
        print(y)

        # data    
        #X, y = make_blobs(n_samples=N, centers=2, n_features=10)
        y[y==0] = -1  # replace the zeros    
        X = np.append(np.ones((N,1)), X, 1)   # add a column of ones

        # initialize the weigths to zeros
        w = np.zeros(3)
        it = 0
        pltPer(X,y,w)  # initial solution (bad!)
        
        stopIt = 1000
        currIt = 0
        bestW = w
        bestE = N
        nerr = classification_error(w, X, y)
        
        # Iterate until all points are correctly classified
        while nerr != 0:
            it += 1
            currIt += 1
            if currIt > stopIt:
                print("Early stop, no progress!")
                break
            # Pick random misclassified point
            x, s = choose_miscl_point(w, X, y)
            # Update weights
            w += s*x
            nerr = classification_error(w, X, y)
            if nerr < bestE:
                currIt = 0
                bestE = nerr
                bestW = w
                
        w = bestW
        pltPer(X,y,w)
        print("Total iterations: " + str(it - stopIt))
        itlst.append(it)
    print(itlst)
    f = plt.figure()
    plt.hist(itlst)


# In[44]:


main()


# In[ ]:




