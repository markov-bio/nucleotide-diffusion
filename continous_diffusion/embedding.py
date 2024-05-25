import einops
import torch
from torch import nn
from torch.nn import functional as F


class Embedder(nn.Module):
    def __init__(self, tokenizer, embed_dim):
        super().__init__()
        self.tokenizer=tokenizer
        self.pad_token=tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        self.mask_token=tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        self.num_embeddings = tokenizer.vocab_size

        self.embedding = nn.Embedding(self.num_embeddings, embed_dim)
        self.embed_dim=embed_dim
    
    def forward(self, x):
        embeddings = self.embedding(x)
        return F.normalize(embeddings, p=2, dim=-1)

    def expected_embedding(self, logits):
        prob=F.softmax(logits,dim=-1)
        out = F.linear(prob, self.embedding.weight.t())
        return out

        