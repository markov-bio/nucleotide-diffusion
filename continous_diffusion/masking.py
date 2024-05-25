import torch
from torch import nn, Tensor

import random



class MaskMaker(nn.Module):
    def __init__(self,masking_range,max_prefix):
        super().__init__()

        self.masking_range=masking_range
        self.max_prefix=max_prefix

    def forward(self,shape,attn_mask,device='cpu'):
        batch_size,sequence_length=shape
        masking_ratio=torch.rand((batch_size,),device=device)
        masking_ratio=self.masking_range[0] + masking_ratio*(self.masking_range[1]-self.masking_range[0]) #to make uniform in masking_range

        if self.max_prefix>0:
            prefixes=torch.randint(low=0,high=self.max_prefix,size=(batch_size,),device=device)
        else:
            prefixes=torch.zeros((batch_size,),device=device)

        mask=create_fraction_mask(masking_ratio,prefixes,shape[1], attn_mask, device)
        return mask




def create_fraction_mask(fractions:Tensor, prefixes:Tensor, seq_len:int, attn_mask, device):
    batch_size = fractions.shape[0]
    # Create a random matrix of shape (batch_size, seq_len)
    rand_matrix = torch.rand(batch_size, seq_len, device=device)
    
    # mask the prefixes 
    indices = torch.arange(seq_len,device=device).expand(batch_size, seq_len)
    prefix_mask = indices < prefixes.unsqueeze(1)

    # make sure that the prefixes are not longer than the non-padded part of the sequence
    total_mask=torch.logical_or(~attn_mask,prefix_mask)
    broken_positions=torch.all(total_mask,dim=-1)
    total_mask[broken_positions]=False
    rand_matrix[total_mask] = float('inf')

    # Compute the number of True values needed for each row
    len_to_noise=(rand_matrix!=float('inf')).sum(dim=-1)
    num_true = (fractions * len_to_noise).clamp(min=1).long()
    
    # Sort each row to get the thresholds
    sorted_rand_matrix, _ = torch.sort(rand_matrix, dim=1)
    
    # Create a threshold matrix by selecting the appropriate value from each row
    threshold = sorted_rand_matrix[torch.arange(batch_size, device=device), num_true]
    
    # Create the mask by comparing the random matrix with the threshold
    mask = rand_matrix < threshold.unsqueeze(1)
    
    return mask
