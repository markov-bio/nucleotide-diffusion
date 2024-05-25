import einops
import torch
from torch import nn, Tensor

from .conditioning import TimeConditioning
from .DiT_block import DiTBlock
from .RoPe import RotaryEmbedding
from .utils import bmult
class DiffusionTransformer(nn.Module):
    def __init__(self, embed_dim, num_embeddings, hidden_dim, qkv_dim, num_heads, cond_dim, n_blocks, max_len=5000):
        super().__init__()
        # ensure the embedding dimension is compatible with the number of heads
        assert qkv_dim % num_heads == 0 and qkv_dim != num_heads, "embedding dimension must be divisible by number of heads and not equal to it."

        self.rope = RotaryEmbedding(qkv_dim // num_heads)

        self.time_conditioning = TimeConditioning(cond_dim, cond_dim)

        self.embedding_to_hidden = nn.Linear(4*embed_dim,hidden_dim)
        self.hidden_to_logits = nn.Linear(hidden_dim,num_embeddings)

        self.dit_blocks = nn.Sequential(
            *[DiTBlock(hidden_dim, qkv_dim, num_heads, cond_dim, self.rope, max_len) for _ in range(n_blocks)]
        )

    def forward(self, noised_embeddings,clean_embeddings,self_cond,m, sigma: Tensor, attn_mask: Tensor = None, masking: Tensor = None) -> Tensor:

        x=torch.cat((noised_embeddings,clean_embeddings,self_cond,m),dim=-1)
        # define the preconditioning
        c_noise=torch.log(sigma)/4
        c_skip= 1/(1+sigma**2)
        c_in=torch.sqrt(1/(1+sigma**2))
        c_out = sigma/torch.sqrt(1+sigma**2)
        
        x = bmult(x,c_in)
        conditioning = self.time_conditioning(c_noise)

        attn_mask = transform_attn_mask(attn_mask)

        x=self.embedding_to_hidden(x)
        # skip = x.clone()
        # apply the sequence of dit blocks
        for block in self.dit_blocks:
            x = block(x, conditioning, attn_mask) 

        # x = bmult(skip,c_skip) + bmult(x,c_out)
        x=self.hidden_to_logits(x)
        return x

def transform_attn_mask(attn_mask):
    """Transform the attention mask for broadcasting."""
    if attn_mask==None: return None
    ones=torch.full((attn_mask.shape[-1],),True,device=attn_mask.device)
    return einops.einsum(ones,attn_mask, 'l, b m -> b l m').unsqueeze(1)

def apply_masking(computed_tensor:Tensor, original_tensor:Tensor, masking:Tensor)->Tensor:
    computed_tensor[~masking]=original_tensor[~masking].to(computed_tensor.dtype)
    return computed_tensor