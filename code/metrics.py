from math import log2
import numpy as np
import torch
from scipy.stats import wasserstein_distance

def compute_wass_hist(a_hist, b_hist):
    ''' Compute Wasserstein distance between 2 histograms
    Histograms are of shape (number of samples, dims). 
    Usage: given an input, compute distance between softmax and approx softmax
    If the vectors are not normalized. Normalize them 
    '''
    a_hist = a_hist/torch.sum(a_hist,dim=1).unsqueeze(-1)
    b_hist = b_hist/torch.sum(b_hist, dim=1).unsqueeze(-1)
    assert a_hist.shape[0]==b_hist.shape[0]

    wass_dist = []
    for i in range(a_hist.shape[0]):
        wass_dist.append(wasserstein_distance(a_hist[i].numpy(),b_hist[i].numpy()))
    return sum(wass_dist)/len(wass_dist)


def kl_divergence(a_hist, b_hist):
    ''' kl_divergence between 2 histograms. The histograms are of shape(number of samples, dim)
    Keeping the normalization just as before in case things are not normalized.
    '''
    a_hist = a_hist/torch.sum(a_hist,dim=1).unsqueeze(-1)
    b_hist = b_hist/torch.sum(b_hist, dim=1).unsqueeze(-1)
    assert a_hist.shape[0]==b_hist.shape[0]
    
    div = []
    for j in range(a_hist.shape[0]):
         div.append(sum(a_hist[j][i] * log2(a_hist[j][i]/b_hist[j][i]) for i in range(a_hist.shape[1])))
               
    return sum(div)/len(div)

#TODO Add Wasserstein or Sliced Wasserstein distances between softmax vectors and the approx softmax vectors. 
# But this time we do not normalize the approx softmax vectors and just treat them as vectors and the points (i.e. word probabilities are either uniform 
# or the relative frequency of the word)
