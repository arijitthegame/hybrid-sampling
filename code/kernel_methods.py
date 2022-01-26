import math
import numpy as np

'''
Code for all the kernel methods.
'''

def gram_schmidt_columns(X):
    '''
    Using QR decomoposition to obtain orthogonal matrix.
    
    Parameters
    ----------
    X : matrix, dimension = m * d, where m <= d
        Random feature matrix with l2 normalized row.
    Returns
    -------
    Q : matrix, dimension = m * d, where m <= d
        Orthogonal random feature matrix with l2 normalized row.
    '''
    Q, R = np.linalg.qr(X)
    return Q

def orthgonalize(V):
    '''
    Generate matrix with multiple orthogonal blocks
    Parameters
    ----------
    V : matrix, dimension = m * d, where m > d
        Random feature matrix with l2 normalized row.
    Returns
    -------
    V_ : TYPE
        Random feature matrix with l2 normalized row and multiple
        blocks.
    '''
    N = V.shape[0]
    d = V.shape[1]
    turns = int(N/d)
    remainder = N%d

    if turns:
        V_ = np.zeros_like(V)

        for i in range(turns):
            v = gram_schmidt_columns(V[i*d:(i+1)*d, :].T).T
            V_[i*d:(i+1)*d, :] = v
        if remainder != 0:
            V_[(i+1)*d:,:] = gram_schmidt_columns(V[(i+1)*d:,:].T).T
    else:
        V_ = gram_schmidt_columns(V.T).T

    return V_

def orthogonal_gau(dim_0, dim_1):

    V = np.random.normal(0, 1, (dim_0, dim_1))
    norms = np.linalg.norm(V, axis = 1)[:, np.newaxis]
    V_orth = orthgonalize(V)
    
    return V_orth*norms

def trig_att(x, y, random_feats_sfm, normalize=False):
    
    l, d = x.shape
  
    normalizer = 1 / (d ** 0.25) if normalize else 1
    
    x = x * normalizer
    y = y * normalizer
    
    x_feat = np.sqrt(1/(random_feats_sfm.shape[0])) *\
                 np.exp(np.linalg.norm(x, axis = 1)**2/2)[:, np.newaxis] *\
                 np.vstack((np.sin(random_feats_sfm.dot(x.T)), \
                            np.cos(random_feats_sfm.dot(x.T)))).T

    y_feat = np.sqrt(1/(random_feats_sfm.shape[0])) *\
                 np.exp(np.linalg.norm(y, axis = 1)**2/2)[:, np.newaxis] *\
                 np.vstack((np.sin(random_feats_sfm.dot(y.T)), \
                            np.cos(random_feats_sfm.dot(y.T)))).T
  
    return np.dot(x_feat, y_feat.T)


def pos_att(x, y, random_feats_sfm, normalize=False):
    
    l, d = x.shape
  
    normalizer = 1 / (d ** 0.25) if normalize else 1
    
    x = x * normalizer
   
    
    x_feat = np.sqrt(1/(2*random_feats_sfm.shape[0])) * \
                    np.exp(-np.linalg.norm(x, axis = 1)**2/2)[:, np.newaxis] *\
                    np.vstack((np.exp(random_feats_sfm.dot(x.T)), \
                                np.exp(-random_feats_sfm.dot(x.T)))).T
    del x
    
    y = y * normalizer
    y_feat = np.sqrt(1/(2*random_feats_sfm.shape[0])) * \
                    np.exp(-np.linalg.norm(y, axis = 1)**2/2)[:, np.newaxis] *\
                    np.vstack((np.exp(random_feats_sfm.dot(y.T)), \
                                np.exp(-random_feats_sfm.dot(y.T)))).T
     
    
    del y
    return np.dot(x_feat, y_feat.T)

def ang_hyb_lambda(x, y, random_feats_lambda, normalize=False):
    
    l, d = x.shape
  
    normalizer = 1 / (d ** 0.25) if normalize else 1
    
    x = x * normalizer
    
    x_feat = np.hstack((np.repeat(np.sqrt(1/2), x.shape[0])[:, np.newaxis],\
                                      (1j*np.sqrt(1/(2*random_feats_lambda.shape[0])) *\
                                      np.sign(random_feats_lambda.dot(x.T))).T))
    #print('x_feat shape ', x_feat.shape)  
    del x 
    
    y = y * normalizer
    y_feat = np.hstack((np.repeat(np.sqrt(1/2), y.shape[0])[:, np.newaxis],\
                                      (1j*np.sqrt(1/(2*random_feats_lambda.shape[0])) *\
                                      np.sign(random_feats_lambda.dot(y.T))).T))
    #print('y_feat shape ', y_feat.shape) 
    del y
  
    return np.dot(x_feat, y_feat.T).real


def gau_hyb_lambda(x, y, random_feats_lambda, lambda_=1, normalize=False):
    
    l, d = x.shape
  
    normalizer = 1 / (d ** 0.25) if normalize else 1
    
    x = x * normalizer

    x_feat = (1*np.sqrt(1/(random_feats_lambda.shape[0])) *\
                      np.vstack((np.sin(lambda_*random_feats_lambda.dot(x.T)), \
                                np.cos(lambda_*random_feats_lambda.dot(x.T))))).T
     
    del x
    
    y = y * normalizer
    y_feat = (1*np.sqrt(1/(random_feats_lambda.shape[0])) *\
                      np.vstack((np.sin(lambda_*random_feats_lambda.dot(y.T)), \
                                np.cos(lambda_*random_feats_lambda.dot(y.T))))).T
    
    del y  
  
    return np.dot(x_feat, y_feat.T)


def ang_hyb_att(x, y, random_feats_sfm, random_feats_lambda, normalize=True):

    approx_softmax_trig_hyb = trig_att(x, y, random_feats_sfm, normalize=True)
            
    approx_softmax_pos_hyb = pos_att(x, y, random_feats_sfm, normalize=True)
    
    approx_softmax_ang = ang_hyb_lambda(x, y, random_feats_lambda, normalize=True)

    approx_softmax_hyb_ang = np.multiply((approx_softmax_ang), approx_softmax_pos_hyb) + \
                            np.multiply((1 - approx_softmax_ang), approx_softmax_trig_hyb)

    del approx_softmax_trig_hyb, approx_softmax_pos_hyb, approx_softmax_ang
    return approx_softmax_hyb_ang

def gau_hyb_att(x, y, random_feats_sfm, random_feats_lambda, normalize=True):

    approx_softmax_trig_hyb = trig_att(x, y, random_feats_sfm, normalize=True)
            
    approx_softmax_pos_hyb =pos_att(x, y, random_feats_sfm, normalize=True)

    approx_softmax_gau = gau_hyb_lambda(x, y, random_feats_lambda, normalize=True)
            
    approx_softmax_hyb_gau = np.multiply((1-approx_softmax_gau), approx_softmax_pos_hyb) + \
                            np.multiply(approx_softmax_gau, approx_softmax_trig_hyb)

    del approx_softmax_trig_hyb, approx_softmax_pos_hyb, approx_softmax_gau
    return approx_softmax_hyb_gau
