import torch
import torch.distributed as dist
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
import transformers
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import torch.optim as optim
from functools import partial

import os
import sys
import math
import contextlib
import signal

import multiprocessing

from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoModelForMaskedLM, AutoModel, AutoConfig, AutoTokenizer, DataCollatorForLanguageModeling


from timm.utils import ModelEmaV3

use_ddp=False
use_hpu=False

# via https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/koleo_loss.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the Apache License, Version 2.0
class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=1)  # noqa: E741
        return I

    def forward(self, student_output, eps=1e-8):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        with torch.cuda.amp.autocast(enabled=False):
            student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
            I = self.pairwise_NNs_inner(student_output)  # noqa: E741
            distances = self.pdist(student_output, student_output[I])  # BxD, BxD -> B
            loss = -torch.log(distances + eps).mean()
        return loss

# Gemini-3.1-Pro
# functional version of code above
def ko_leo_loss(student_output, eps=1e-8):
    """
    Kozachenko-Leonenko entropic loss regularizer.
    Args:
        student_output (Tensor): BxD tensor of student embeddings.
        eps (float): Small value for numerical stability.
    """
    # Ensure calculation is in float32 for stability
    with torch.cuda.amp.autocast(enabled=False):
        # 1. L2 Normalize vectors (crucial for dot-prod to represent distance)
        x = F.normalize(student_output.float(), p=2, dim=-1, eps=eps)
        n, d = x.shape

        # 2. Find nearest neighbors using dot product
        # (x * x.T) gives the cosine similarity matrix
        dots = torch.mm(x, x.t())
        
        # Fill diagonal with -1 to ensure a vector doesn't pick itself as its NN
        dots.view(-1)[:: (n + 1)].fill_(-1.0)
        
        # Find indices of max similarity (minimum distance)
        nn_indices = torch.max(dots, dim=1)[1]

        # 3. Calculate L2 distances to the nearest neighbors
        # We index the original x with the discovered NN indices
        diff = x - x[nn_indices]
        distances = torch.norm(diff, p=2, dim=-1)

        # 4. Final Loss: Negative log of the distances
        loss = -torch.log(distances + eps).mean()
        
    return loss

#if use_hpu:
#    from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

def getDataLoader(dataset, batch_size, epoch, collate_fn=None):
    num_workers = 10
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size = batch_size, 
        #shuffle=True, 
        num_workers=num_workers, 
        persistent_workers = False, 
        prefetch_factor=2, 
        generator=torch.Generator().manual_seed(41),
        collate_fn=collate_fn,
        drop_last=True
    )

def filter_state_dict(state_dict):
    out_dict={}
    for k, v in state_dict.items():
        k = k.replace('_orig_mod.', '')
        k = k.replace('module.', '')
        out_dict[k] = v
    return out_dict

def train(device):
    is_head_proc = not use_ddp or dist.get_rank() == 0

    #config = AutoConfig.from_pretrained("answerdotai/ModernBERT-base")
    config = AutoConfig.from_pretrained("facebook/esm2_t30_150M_UR50D")
    model = AutoModelForMaskedLM.from_config(
        config,
        #attn_implementation='sdpa'
        attn_implementation="kernels-community/flash-attn3"
    ).to(device)

    model_ema = ModelEmaV3(model, decay=0.9998)

    if (use_ddp):
        if use_hpu:
            model = DDP(model)
        else:
            model = DDP(model, device_ids=[device], gradient_as_bucket_view=True, find_unused_parameters=False)
    
    do_compile=True

    if do_compile:
        if use_hpu:
            model = torch.compile(model, backend='hpu_backend')
            model_ema = torch.compile(model_ema, backend='hpu_backend')
        else:
            model = torch.compile(model)
            model_ema = torch.compile(model_ema)

    #tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')
    tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t30_150M_UR50D')
    
    #ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    ds = load_dataset("bloyal/uniref50", split="train", streaming=True)

    if (use_ddp):
        ds = split_dataset_by_node(ds, rank=dist.get_rank(), world_size=dist.get_world_size())

    ds = ds.map(lambda x: tokenizer(x['text'], truncation=True, max_length=512, padding=False, add_special_tokens=False), batched=True, remove_columns=ds.column_names)

    print(ds.info)

    ds_len = ds.info.splits['train'].num_examples
    if use_ddp:
        ds_len = ds_len // dist.get_world_size()


    full_data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors='pt'
    )

    mlm_data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
        mask_replace_prob=0.7,
        random_replace_prob=0.3,
        return_tensors='pt'
    )

    def joint_collate(examples):
        return full_data_collator(examples), mlm_data_collator(examples)


    num_epochs = 1
    batch_size = 192
    grad_accum_iters = 8
    learning_rate = 1e-4

    optimizer = optim.AdamW(
        [
            *model.parameters(),
        ],
        lr=learning_rate,
        weight_decay=1e-2
    )


    scaler = torch.amp.GradScaler(device.type)
    model.train()

    for epoch in range(num_epochs):
        train_dataloader = getDataLoader(ds, batch_size, epoch, collate_fn=joint_collate)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=learning_rate, 
            #steps_per_epoch=len(train_dataloader),
            steps_per_epoch = ds_len // batch_size,
            epochs=num_epochs,
            pct_start=0.1
        )
        lr_scheduler.last_epoch = (ds_len // batch_size) * epoch
        for i, (full_batch, mlm_batch) in enumerate(train_dataloader):
            full_batch = {k: v.to(device, non_blocking=True) for k, v in full_batch.items()}
            mlm_batch = {k: v.to(device, non_blocking=True) for k, v in mlm_batch.items()}

            is_optim_step_iter = (i+1) % grad_accum_iters == 0
            ddp_sync_context = contextlib.nullcontext() if is_optim_step_iter or not use_ddp else model.no_sync()
            with torch.amp.autocast(device.type, enabled=True, dtype=torch.bfloat16), ddp_sync_context:
                outputs = model(**mlm_batch, output_hidden_states=True)
                #student_repr = outputs.hidden_states[-1][mlm_batch['labels'] != -100].flatten(end_dim=-2)

                #with torch.no_grad():
                #    outputs_ema = model_ema(**full_batch, output_hidden_states=True)
                #    teacher_repr = outputs_ema.hidden_states[-1][mlm_batch['labels'] != -100].flatten(end_dim=-2)


                loss_mlm = outputs.loss
                #loss_mse = F.mse_loss(student_repr, teacher_repr)
                #loss_ko_leo = ko_leo_loss(student_repr)

                loss = loss_mlm #+ 1.0 * loss_mse + 0.3 * loss_ko_leo

                #loss = 1.0 * loss_mlm + 1.0 * loss_distogram + 2.0 * structure_loss + 0.5 * loss_distogram_self_distill * int(epoch > 0)

                scaler.scale(loss).backward()
                perplexity = math.exp(loss_mlm.detach().item())

                if is_optim_step_iter:
                    if not use_hpu and use_ddp:
                        torch.cuda.synchronize()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                    #if do_compile:
                    #    model_ema._orig_mod.update(model)
                    #else:
                    #    model_ema.update(model)

                lr_scheduler.step()
            if (i+1) % grad_accum_iters == 0 and is_head_proc:
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {i + 1}/{ds_len // batch_size} | Loss (MLM): {loss_mlm.item():.4f} | PPLX: {perplexity:.4f} | Loss (MSE): {loss_mse.item():.4f} | Loss (KoLeo): {loss_ko_leo.item():.4f}")
                sys.stdout.flush()
            #torch.cuda.empty_cache()
            #p.step()
        torch.cuda.synchronize()
        if is_head_proc:
            checkpoint_name = f"ESM2_150M_UR50_1epoch_MSE0.0_KoLeo0.0/epoch_{epoch}"
            if use_ddp:
                model.module.save_pretrained(f"./outputs/{checkpoint_name}")
            else:
                model.save_pretrained(f"./outputs/{checkpoint_name}")
            torch.save(optimizer.state_dict(), f"./outputs/{checkpoint_name}/optimizer.pth")


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!, terminating child processes')

    active = multiprocessing.active_children()
    for child in active:
        child.terminate()

    if use_ddp:
        dist.destroy_process_group()
    
    sys.exit(0)

def main():
    #signal.signal(signal.SIGINT, signal_handler)

    if use_ddp:
        torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
        acc = torch.accelerator.current_accelerator()
        backend = torch.distributed.get_default_backend_for_device(acc)
        dist.init_process_group(backend)
        rank = dist.get_rank()
        
        if use_hpu:
            device = torch.device('hpu')
        else:
            device = acc
    else:
        if use_hpu:
            device = torch.device('hpu')
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    '''
    try:
        train(device)
    except Exception as e:
        print("An error occurred:", e)

        active = multiprocessing.active_children()
        for child in active:
            child.terminate()

        if use_ddp:
            dist.destroy_process_group()
    '''
    train(device)

    if use_ddp:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()