import math
import torch 
import torch.nn as nn
import torch.distributions as distributions

class RFF():
    def __init__(self, gamma = 1, D = 50):
        '''
        Random Fourier Features to approximate a Gaussian Kernel. 
        D = output dimension of the features
        gamma = temperature, where the kernel K(x-y) = exp(-gamma/2 ||x-y||^2)
        '''
        self.gamma = gamma
        self.D = D
        
    def generate(self, X):
        """ Generates random samples from isotropic Gaussian """
        d = X.shape[1]
       
        p = distributions.multivariate_normal.MultivariateNormal(torch.zeros(d),
  math.sqrt(self.gamma)*torch.eye(d)) 
       
        #Generate D iid samples from p(w) 
        
        self.w = p.sample(torch.Size([self.D]))
      
                
        #Generate D iid samples from Uniform(0,2*pi)
        uniform = distributions.uniform.Uniform(torch.tensor([0.0]), torch.tensor([2*math.pi]))
        self.u = uniform.sample(torch.Size([self.D]))
   
        return self.w, self.u
    
    def transform(self,X):
        """ Transforms the data X (n_samples, n_features) to the new map space Z(X) (n_samples, n_components)"""
        #Compute feature map Z(x):
        
        self.w, self.u = self.generate(X)
       
        Z = math.sqrt(2/self.D)*torch.cos(torch.matmul(X,(self.w).t()) + self.u.squeeze())
        return Z
    
    def compute_kernel(self, X):
        """ Computes the approximated kernel matrix K """
        
        Z = self.transform(X)
        K = torch.matmul(Z,Z.t())
        return K


def cosh(X):
    return (torch.exp(X)+torch.exp(-X))/2.0

class FavorPlus():
    def __init__(self, gamma = 1, D = 50):
        '''
        Random positive  Features to approximate a Gaussian Kernel. 
        D = output dimension of the features
        gamma = temperature, where the kernel K(x-y) = exp(-gamma/2 ||x-y||^2)
        '''
        self.gamma = gamma
        self.D = D
        
    

    def generate(self, X):
        """ Generates random samples from isotropic Gaussian """
        d = X.shape[1]
       
        p = distributions.multivariate_normal.MultivariateNormal(torch.zeros(d), math.sqrt(self.gamma)*torch.eye(d)) 
       
        #Generate D iid samples from p(w) 
        
        w = p.sample(torch.Size([self.D]))
        return w

        #TODO : Create orthogonal features
    
    def transform(self,X):
        """ Transforms the data X (n_samples, n_features) to the new map space Z(X) (n_samples, n_components)"""
        #Compute feature map Z(x):
        
        w = self.generate(X)
       
        Z = math.sqrt(2/self.D)*torch.exp(torch.matmul(X,w.t()))*(torch.exp(-torch.linalg.norm(X, dim=1)**2)).view(-1,1)
        return Z
    
    def compute_kernel(self, X):
        """ Computes the approximated kernel matrix K """
        
        Z = self.transform(X)
        K = torch.matmul(Z,Z.t())
        return K

    def approximate_softmax(self,X,Y):
        """ Computes the approximate softmax kernel"""
        #TODO Use orthogonal vectors instead of just random
        Z = X + Y
        X1, Y1 = self.transform(X), self.transform(Y)
        Lambda = torch.exp((torch.linalg.norm(X,dim=1)**2 + torch.linalg.norm(Y,dim=1)**2)/2.0) 
        approx = torch.mm(X1,Y1.t())*Lambda.view(-1,1)
        return approx

    #TODO: Bug somewhere since the computed matrices are far from stochastic



