import logging
import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt

from scipy.optimize import curve_fit
from torch.distributions import Uniform, TransformedDistribution
from torch.distributions.transforms import SigmoidTransform, AffineTransform, ComposeTransform

from .utils import TanTransform, get_codomain, cumulative_density_function, median


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdaptiveSchedule(nn.Module):
    def __init__(self, tmin, tmax, mu, sigma, height, offset):
        super().__init__()

        self.tmin, self.tmax = tmin, tmax
        self.parameters_history, self.times, self.entropy, self.medians = [], [], [], []
        self.set_parameters([mu, sigma, height, offset])

    def set_parameters(self, params):
        self.mu, self.sigma, self.height, self.offset = params
        self.medians.append(median(*params))
        self.optimal_parameters = nn.Parameter(torch.tensor(params, device='cpu'), requires_grad=False)
        self.parameters_history.append(params)
    
        self.transforms = ComposeTransform([
            AffineTransform(loc=-self.offset / self.height, scale=1 / self.height),
            AffineTransform(loc=-np.pi / 2, scale=np.pi),
            TanTransform(),
            AffineTransform(loc=self.mu, scale=self.sigma)
        ])

        imin, imax = get_codomain(cumulative_density_function, self.tmin, self.tmax, self.mu, self.sigma, self.height, self.offset)
        self.distribution = TransformedDistribution(Uniform(imin, imax), self.transforms)

    
    def update_optimal_parameters(self, history=500):
        times, entropy = self.times[-history:], self.entropy[-history:] 

        offset_lower_bound=-cumulative_density_function(0,self.mu, self.sigma, self.height, 0) #this is an approximation but it gets better and as the training progresses
        optimal_parameters, _ = curve_fit(cumulative_density_function, times, entropy, (self.mu,self.sigma,self.height, offset_lower_bound+1e-3), bounds=((0,0,0,offset_lower_bound),(np.inf,np.inf,np.inf,np.inf)))
        
        self.set_parameters(optimal_parameters.tolist())
        logging.info(f"Updated optimal parameters: mu={self.mu}, sigma={self.sigma}, height={self.height}, offset={self.offset}")    

    def make_timesteps(self, steps, tmin=None, tmax=None, device='cpu'):
        tmin=self.tmin if tmin is None else tmin
        tmax=self.tmax if tmax is None else tmax

        imin,imax=get_codomain(cumulative_density_function, tmin, tmax, self.mu, self.sigma, self.height, self.offset)
        indexes=torch.linspace(imax,imin,steps, device=device)

        timesteps = self.transforms(indexes)
        return torch.clamp(timesteps,self.tmin, self.tmax)

    def add_data(self, entropy, times):
        self.times += times.tolist()
        self.entropy += entropy.tolist()

    def sample(self, shape):
        return self.distribution.sample(shape)










    # just some plotting functions
    def plot_entropy_time_curve(self, filename='et.png',title='CrossEntropy-Sigma Curve'):
        history_cutoff=int(1e5)
        if len(self.times)>=history_cutoff:
            self.times=self.times[-history_cutoff:]
            self.entropy=self.entropy[-history_cutoff:]

        plt.close()
        plt.figure(figsize=(20, 4)) 
        # Calculate logarithmic indices for coloring
        indices = np.arange(1, len(self.times) + 1)
        log_indices = np.log(indices)[::-1]  # Reverse to give more weight to recent points
        log_indices = (log_indices - np.min(log_indices)) / (np.max(log_indices) - np.min(log_indices))

        # Scatter plot of entropy vs. time
        plt.scatter(self.times, self.entropy, c=log_indices, cmap='viridis', label='datapoints')


        # Plot the best fit function
        t = np.logspace(np.log10(self.tmin), np.log10(self.tmax), 500, base=10.)
        s = cumulative_density_function(t, *self.optimal_parameters.detach().cpu().tolist())

        plt.plot(t, s, color='purple', label='Learnt unnormalized CFD')
        plt.title(title)
        plt.xlabel('Sigma')
        plt.ylabel('CrossEntropy')
        plt.xscale('log')
        plt.ylim(-0.2,7.2)
        plt.xlim(0.7,200)
        plt.legend()

        # Save the plot to a file
        plt.savefig(filename)
        plt.show()

    def plot_training_curves(self,title='CrossEntropy-Sigma Curve',filename='curves.png'):
        plt.close()
        plt.figure(figsize=(20, 4)) 

        
        # Calculate logarithmic indices for coloring
        indices = np.arange(1, len(self.parameters_history))
        log_indices = np.log(indices)[::-1]  # Reverse to give more weight to recent points
        log_indices = (log_indices - np.min(log_indices)) / (np.max(log_indices) - np.min(log_indices))

        cmap=plt.get_cmap('viridis')
        colors=cmap(log_indices)
        # Plot the best fit function
        for p,c in zip(self.parameters_history[1:],colors):
            t = np.logspace(np.log10(self.tmin), np.log10(self.tmax), 500, base=10.)
            s = cumulative_density_function(t, *p)
            plt.plot(t, s, color=c)

        plt.title(title)
        plt.xlabel('Sigma')
        plt.ylabel('CrossEntropy')
        plt.xscale('log')
        plt.ylim(-0.2,7.2)
        plt.xlim(0.7,200)
        plt.legend()

        # Save the plot to a file
        plt.savefig(filename)
        plt.show()





