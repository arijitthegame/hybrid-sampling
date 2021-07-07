import math
import torch 
import torch.nn as nn
import torch.distributions as distributions
from scipy.linalg import orth

#TODO : Possible bug since I am generating different \omega for X and Y.

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

    def orthogonalize(self, X):
        """ Generate orthogonal features"""
        
        w = self.generate(X)
        w_orth = orth(w.t().numpy())
        return torch.from_numpy(w_orth).t()
    
    def transform(self,X):
        """ Transforms the data X (n_samples, n_features) to the new map space Z(X) (n_samples, n_components)"""
        #Compute feature map Z(x):
        
        w = self.orthogonalize(X)
       
        Z = math.sqrt(2/self.D)*torch.exp(torch.matmul(X,w.t()))*(torch.exp(-torch.linalg.norm(X, dim=1)**2)).view(-1,1)
        return Z
    
    def compute_kernel(self, X):
        """ Computes the approximated kernel matrix K """
        
        Z = self.transform(X)
        K = torch.matmul(Z,Z.t())
        return K

    def approximate_softmax(self,X,Y):
        """ Computes the approximate softmax kernel"""
    
        Z = X + Y
        X1, Y1 = self.transform(X), self.transform(Y)
        Lambda = torch.exp((torch.linalg.norm(X,dim=1)**2 + torch.linalg.norm(Y,dim=1)**2)/2.0) 
        approx = torch.mm(X1,Y1.t())*Lambda.view(-1,1)
        return approx

 
class Hybrid():
    def __init__(self, gamma = 1, d_trig = 50, d_ang = 50):
        '''
        Random positive  Features to approximate a Gaussian Kernel. 
        D = output dimension of the features
        gamma = temperature, where the kernel K(x-y) = exp(-gamma/2 ||x-y||^2)
        '''
        self.gamma = gamma
        self.d_trig = d_trig
        self.d_ang = d_ang
    

    def generate(self, X):
        """ Generates random samples from isotropic Gaussian """
        d = X.shape[1]
       
        p = distributions.multivariate_normal.MultivariateNormal(torch.zeros(d), math.sqrt(self.gamma)*torch.eye(d)) 
       
        #Generate iid samples from p(w) 
        
        w_trig, w_ang = p.sample(torch.Size([self.d_trig])), p.sample(torch.Size([self.d_ang]))
        
        uniform = distributions.uniform.Uniform(torch.tensor([0.0]), torch.tensor([2*math.pi]))
        u = uniform.sample(torch.Size([self.d_trig]))
        return w_trig, w_ang, u

    
    def transform_trig(self,X):
        """ Transforms the data X (n_samples, n_features) to the new map space Z(X) (n_samples, n_components)
            Applying the trigonometric feature map
        """
     
        
        w_trig, _,  u = self.transform(X)
       
        Z_trig = math.sqrt(2/self.d_trig)*torch.cos(torch.matmul(X,w_trig.t()) + self.u.squeeze())
        return Z
    
    def transform_angular(self, X):
        """
        Compute the approximator for the angular kernel.
        """
        _, w_ang, _ = self.transform(X)
        Z_ang = math.sqrt(1/self.d_ang) torch.sign(torch.matmul(X, w_ang.t()))
        return Z_ang
    
    #def compute_hybrid(self, X) :
#     """
#     Computes the outer/tensor product of the above features. 
#     Tensors: (b,d_trig) \otimes (b,d_ang) -> (b, d_trig, d_ang)
#     """
       #Z_ang, Z_trig = self.transform_angular(X), self.transform_trig(X) 
    #   Z_hyb = torch.einsum('bi,bj->bij', (Z_ang, Z_trig))
        
  #TODO Add tensor product and compute the hybrid features. 
#np.einsum(“i, j -> ij”, vec, vec)



