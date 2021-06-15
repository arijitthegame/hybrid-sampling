from math import log2
import numpy as np
import torch
from scipy.stats import wasserstein_distance
import ot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

def compute_sliced_wass(source_dist, target_dist, weights='uniform', n_seed=42, n_proj = 25):
    """
     Sliced Wasserstein distances between softmax vectors and the approx softmax vectors. 
     This time we do not normalize the approx softmax vectors and just treat them as vectors (locations) and an empirical measure on the vectors 
     (i.e. probabilities are either uniform)
    """
    n_samples = source_dist.shape[0]
    a, b = np.ones((n_samples,)) / n_samples, np.ones((n_samples,)) / n_samples
    #TODO: FIX start and end of logspace
    n_projections_arr = np.logspace(0, 4, n_proj, dtype=int)
    res = np.empty((n_seed, 25))

    for seed in range(n_seed):
        for i, n_projections in enumerate(n_projections_arr):
            res[seed, i] = ot.sliced.sliced_wasserstein_distance(xs, xt, a, b, n_projections, seed)

    res_mean = np.mean(res, axis=0)
    res_std = np.std(res, axis=0)
    return res_mean, res_std

class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
            
    Usage: sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, reduction=None).to(device)
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).to(device).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).to(device).squeeze()

        u = torch.zeros_like(mu).to(device)
        v = torch.zeros_like(nu).to(device)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

      #  return cost, pi, C
        return cost

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

