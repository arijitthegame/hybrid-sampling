import numpy as np


# Perfomer attention implementation using some random feature map phi
def att_hat(q, k, phi, normalize=True):
    l, d = k.shape
  
    normalizer = 1 / (d ** 0.25) if normalize else 1
    q_prime = phi(q * normalizer)
   
    k_prime = phi(k * normalizer)
    
    d_inv = np.diag(1 / (q_prime @ (k_prime.T @ np.ones(l))))
    return d_inv @ (q_prime @ k_prime.T)
   
# random feature map
def phi(h, fs, random_feats):
    return lambda x: (
        h(x)
        / np.sqrt(m)
        * np.concatenate(
            [f(np.einsum("...d,md->...m", x, random_feats)) for f in fs],
            axis=-1,
        )
    )


# Performer "sin/cos" attention
def sincos_att_hat(q, k, random_feats, normalize=True):
    def h(x):
        return np.exp(np.square(x).sum(axis=-1, keepdims=True) / 2)

    sin = lambda x: np.sin(2 * np.pi * x)
    cos = lambda x: np.cos(2 * np.pi * x)

    kernel = phi(h, [sin, cos], random_feats)
    return att_hat(q, k, kernel, normalize)


# Performer "positive" attention
def positive_att_hat(q, k, random_feats, normalize=True):
    def h(x):
        return np.exp(-np.square(x).sum(axis=-1, keepdims=True) / 2)

    kernel = phi(h, [np.exp], random_feats)
 
    return att_hat(q, k, kernel, normalize)


# generate IID Gaussian random features
def iid_gaussian(m, d):
    return np.random.normal(size=(m, d))


# generate orthogonal Gaussian random features
def orthogonal_gaussian(m, d):
    '''
  Creates block orthogonal matrices
  '''
    def orthogonal_square():
        # create orthogonal square matrix using Gram-Schmidt
        q, _ = np.linalg.qr(iid_gaussian(d, d))
        return q.T

    num_squares = int(m / d)
    blocks = [orthogonal_square() for _ in range(num_squares)]

    remainder = m - d * num_squares
    if remainder:
        blocks.append(orthogonal_square()[:remainder])

    matrix = np.vstack(blocks)
    matrix /= np.sqrt(num_squares + remainder / d)
    # matrix = np.diag(np.sqrt(d) * np.ones(m)) @ matrix

    return matrix

