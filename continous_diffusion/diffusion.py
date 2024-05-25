import torch
from torch import Tensor
from torch.distributions.categorical import Categorical

from composer.models import ComposerModel

from tqdm import tqdm
import math
import random


from .model import DiffusionTransformer
from .loss import Loss
from .model import DiffusionTransformer
from .utils import bmult
from .embedding import Embedder
from .masking import MaskMaker
from .scheduling import AdaptiveSchedule


class Diffusion(ComposerModel):
    def __init__(self, model:DiffusionTransformer, embedder:Embedder, Loss:Loss, mask_maker:MaskMaker, p_self_cond=0.2, p_mask_cond=0.1):

        super().__init__()

        self.model=model

        self.Loss=Loss
        self.embedder= embedder
        self.noise_schedule=Loss.noise_schedule
        self.tokenizer=self.embedder.tokenizer
        self.mask_maker = mask_maker

        self.p_self_cond=p_self_cond
        self.p_mask_cond=p_mask_cond

        self.n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def make_sample(self,tokens:Tensor):
        attn_mask=tokens!=self.embedder.pad_token

        t = self.noise_schedule.sample(shape=(tokens.shape[0],))
        sigma = t.to(tokens.device)  #Index on cpu then send resulting tensor to cuda
        masking=self.mask_maker(tokens.shape,attn_mask,tokens.device)

        x =  self.embedder(tokens) * math.sqrt(self.embedder.embed_dim)
        noised_embeddings = masking.unsqueeze(-1) *(x.clone() + bmult(torch.randn_like(x), sigma))
        clean_embeddings = ~masking.unsqueeze(-1) * x.clone() 
        if random.random()<self.p_mask_cond:
            clean_embeddings = clean_embeddings + ~masking.unsqueeze(-1)*bmult(torch.randn_like(x), sigma)

        self_cond=random.random()<self.p_self_cond

        return noised_embeddings, clean_embeddings, sigma, attn_mask, masking, self_cond

    def specialized_forward(self,noised_embeddings,clean_embeddings,sigma,attn_mask=None,masking:Tensor=None,self_cond=False): #think of a better name for this funciton
        m = torch.ones_like(noised_embeddings) if masking is None else masking.float().unsqueeze(-1).expand(noised_embeddings.shape)

        #self_cond is either a bool (usually used for training) or a tensor of the pre-computed self-conditioning
        s=torch.zeros_like(noised_embeddings) if isinstance(self_cond,bool) else self_cond

        if self_cond is True:
            with torch.no_grad():
                x=self.model(noised_embeddings,clean_embeddings,s,m,sigma,attn_mask,masking)
                s=self.embedder.expected_embedding(x)
        
        return self.model(noised_embeddings,clean_embeddings,s,m,sigma,attn_mask,masking)

    def forward(self,batch):
        noised_embeddings, clean_embeddings, sigma, attn_mask, masking, self_cond= self.make_sample(batch)
        out=self.specialized_forward(noised_embeddings,clean_embeddings,sigma,attn_mask,masking,self_cond)
        return out,sigma,attn_mask,masking

    def loss(self,outputs,batch):
        output_embeddings,sigma,attn_mask,masking=outputs
        masking=torch.logical_and(attn_mask,masking)
        return self.Loss(batch,output_embeddings,sigma,masking) 

    @torch.no_grad()
    def infill(self,tokens,n_steps):
        masking=tokens==self.embedder.mask_token
        # attn_mask=tokens!=self.embedder.pad_token # not 100% sure what to do with it
        device=next(self.parameters()).device

        x = self.embedder(tokens) * math.sqrt(self.embedder.embed_dim)
        noised_embeddings = masking.unsqueeze(-1) *(torch.randn(x.shape, device=device) * self.noise_schedule.tmax)
        clean_embeddings = ~masking.unsqueeze(-1) * x
        
        return self.denoise(noised_embeddings,clean_embeddings,self.noise_schedule.tmax,n_steps,masking)

    @torch.no_grad()
    def generate(self, batch_size, sequence_lenght, n_steps):
        """
        It denoises the embedded input x for n_steps starting from t_max to t_min
        """
        shape=(batch_size,sequence_lenght,self.embedder.embed_dim) 
        device=next(self.parameters()).device

        noised_embeddings = torch.randn(shape, device=device) * self.noise_schedule.tmax
        clean_embeddings = torch.zeros(shape,device=device)
        
        return self.denoise(noised_embeddings,clean_embeddings,self.noise_schedule.tmax,n_steps)
    
    @torch.no_grad()
    def denoise(self, noised_embeddings, clean_embeddings, noise_level, n_steps, guidance=1., masking=None):
        device=next(self.parameters()).device
        timesteps=self.noise_schedule.make_timesteps(n_steps,tmax=noise_level,device=device).unsqueeze(1)
        x_i=torch.zeros_like(noised_embeddings) 

        for i in tqdm(range(n_steps-1)):        
            if masking is not None:
                uncond_emb=clean_embeddings + ~masking.unsqueeze(-1)*bmult(torch.randn_like(clean_embeddings),timesteps[i])
                uncond_logits=self.specialized_forward(noised_embeddings,uncond_emb,timesteps[i],masking,self_cond=x_i)

            logits=self.specialized_forward(noised_embeddings,clean_embeddings,timesteps[i],masking,self_cond=x_i)

            # x_i will be used as self-conditioning in the next step
            x_i=self.embedder.expected_embedding(logits)
            if masking is not None and guidance<1.:
                x_i = guidance*x_i + (1-guidance)*self.embedder.expected_embedding(uncond_logits)
                

            derivative=(noised_embeddings-x_i)/timesteps[i]
            
            delta_t=timesteps[i+1]-timesteps[i]
            noised_embeddings = noised_embeddings + derivative * delta_t
        
        return self.specialized_forward(noised_embeddings,clean_embeddings,timesteps[-1],self_cond=x_i) 

    
    def generate_text(self,batch_size,text_lenght,n_steps=1000,file=None):
        logits=self.generate(batch_size,text_lenght,n_steps)

        distrubution=Categorical(logits=logits)
        sample=distrubution.sample()

        generated_text=self.tokenizer.batch_decode(sample)
        if file is None:
            print(generated_text)
            return generated_text

        with open(file,'w') as file:
            file.write("\n".join(generated_text))
        return generated_text




class DiffusionModel(Diffusion):
    def __init__(self,embed_dim,hidden_dim,qkv_dim,num_heads,cond_dim,n_blocks,tokenizer,p_self_cond,p_mask_cond,masking_range,max_prefix):
        num_embeddings=tokenizer.vocab_size
        dit=DiffusionTransformer(embed_dim,num_embeddings,hidden_dim,qkv_dim,num_heads,cond_dim,n_blocks)
        embedder=Embedder(tokenizer,embed_dim)
        schedule=AdaptiveSchedule(tmin=0.01,tmax=200, mu=3., sigma=2., height=6., offset=-1.)
        loss=Loss(schedule)
        mask_maker=MaskMaker(masking_range,max_prefix) 
        super().__init__(dit, embedder, loss, mask_maker, p_self_cond, p_mask_cond)