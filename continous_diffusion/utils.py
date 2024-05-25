import einops
import torch
from torch.distributions import Transform, constraints
import numpy as np

def bmult(x,y):
    """element-wise mutiplication of two tensors along the batch dimention

    Args:
        x (torch.Tensor): shape=(b,...)
        y (torch.Tensor): shape=(b)

    Returns:
        torch.Tensor: shape=(b,...)
    """
    return einops.einsum(x, y, 'b ..., b -> b ...')
    

def cumulative_density_function(x, mu, sigma, height, offset):
    return (np.arctan((x - mu) / sigma) / np.pi + 0.5) * height + offset
    
def median(mu,sigma,height,offset):
    half_point=(height+offset)/2
    return np.tan(((half_point-offset)/height-0.5)*np.pi)*sigma+mu

def get_codomain(function,tmin,tmax,*args):
    
    imax=function(tmax,*args)
    imin=function(tmin,*args)

    return imin,imax
    
class TanTransform(Transform):
    bijective = True
    domain = constraints.real
    codomain = constraints.real
    sign = 1

    def _call(self, x):
        return torch.tan(x)

    def _inverse(self, y):
        return torch.atan(y)

    def log_abs_det_jacobian(self, x, y):
        return 2 * torch.log(torch.abs(torch.cos(x)))