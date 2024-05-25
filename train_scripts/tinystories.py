# %%
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR,SequentialLR,ExponentialLR

from continous_diffusion.diffusion import DiffusionModel
from continous_diffusion.callbacks import SchedulerUpdater, PlottingData, WriteText, FindUnused

from datasets import load_dataset
from transformers import AutoTokenizer
import composer
from composer.loggers import WandBLogger
import os

if __name__ == "__main__":
    dataset = load_dataset("roneneldan/TinyStories")['train']

    # dataset_path = os.path.expanduser("~/.cache/huggingface/datasets/roneneldan___parquet/roneneldan--TinyStories-a62fc98e062666ca")
    # dataset = load_dataset(dataset_path)['train']

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")  # or any suitable tokenizer
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print(f"vocab_size: {tokenizer.vocab_size}")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format("torch")  

    # device="cuda" if torch.cuda.is_available() else "cpu"
    # %%
    # embed_dim, hidden_dim, qkv_dim, num_heads, cond_dim, n_blocks = 256, 1024, 1024, 8, 128, 8 #paper parameters (not sure about qkv_dim)  
    # embed_dim, hidden_dim, qkv_dim, num_heads, cond_dim, n_blocks = 64, 256, 2048, 16, 128, 20  
    embed_dim, hidden_dim, qkv_dim, num_heads, cond_dim, n_blocks = 64, 256, 1024, 16, 64, 8 
    # embed_dim, hidden_dim, qkv_dim, num_heads, cond_dim, n_blocks = 64, 128, 512, 8, 64, 2  
    model=DiffusionModel(embed_dim,hidden_dim,qkv_dim,num_heads,cond_dim,n_blocks,tokenizer,p_self_cond=0.4,p_mask_cond=0.8,masking_range=(.3,1.),max_prefix=0)

    print(f"n parameters:{model.n_parameters/1e6}M")
    # model.load_state_dict(torch.load('checkpoints/ep1_0.961538M'))
    # model=torch.compile(model)
    #%%
    sampler=composer.utils.dist.get_sampler(tokenized_datasets['input_ids'])
    train_loader = DataLoader(tokenized_datasets['input_ids'], batch_size=256, sampler=sampler)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)

    n_epochs=1
    warmup_duration = 0.2
    decay_duration = 1-warmup_duration
    n_batches=n_epochs*len(train_loader)
    gamma = 0.05 ** (1 / (n_batches*decay_duration)) 

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=int(n_batches * warmup_duration))
    decay_scheduler = ExponentialLR(optimizer, gamma=gamma)
    lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[int(n_batches * warmup_duration)])

    callbacks=[PlottingData(200,model),SchedulerUpdater(200,model),WriteText(1000,model)]


    #%%
    trainer=composer.Trainer(
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=None,
        max_duration=f'{n_epochs}ep',
        device='gpu',
        callbacks=callbacks,
        loggers=WandBLogger(project='cdcd-natural-language', group='markovbio'),
        optimizers=optimizer,
        schedulers=lr_scheduler,
        step_schedulers_every_batch=True,
        save_folder="./checkpoints",
        save_filename="ep{epoch}_"+f"{model.n_parameters/1e6}M",
        save_latest_filename="latest",
        save_overwrite=True,
        save_interval='1ep',
        algorithms=FindUnused() #necessary for self-conditioning when training with multi-gpu
    )

    trainer.fit()
                
# %%
