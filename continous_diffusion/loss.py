import torch
from torch import nn, Tensor
import einops

from .embedding import Embedder
from .scheduling import AdaptiveSchedule

class Loss(nn.Module):
    def __init__(self, noise_schedule: AdaptiveSchedule):
        super().__init__()
        self.noise_schedule = noise_schedule
        
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, target_tokens: Tensor, logits: Tensor, sigma: Tensor, attn_mask: Tensor) -> Tensor:
        # Transform the output embeddings back to logits
        logits = einops.rearrange(logits, 'b ... c -> b c (...)')

        # Flatten target tokens 
        target_tokens = target_tokens.flatten(start_dim=1)
        attn_mask = attn_mask.flatten(start_dim=1)

        # Compute cross-entropy loss
        loss = self.cross_entropy_loss(logits, target_tokens)
        
        #averaging over the non-padding tokens
        padding_mask = torch.logical_not(attn_mask)
        loss[padding_mask] = 0
        loss = loss.sum(dim=-1) / (padding_mask.shape[-1] - padding_mask.float().sum(dim=-1))

        # Update the adaptive schedule with the current loss and sigma (useful for plotting)
        self.noise_schedule.add_data(loss, sigma)

        return loss.mean()
