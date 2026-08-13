import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
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
import time

import pandas as pd

import multiprocessing

from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoModelForMaskedLM, AutoModel, AutoConfig, AutoTokenizer, DataCollatorForLanguageModeling


from timm.utils import ModelEmaV3

use_ddp=True
use_hpu=False

# via https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/koleo_loss.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the Apache License, Version 2.0
# modified from a module to a functional version of code above by Gemini-3.1-Pro
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
    num_workers = 40
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

def tokenize_function_a_b_gene(examples, tokenizer=None):
    sequences_with_spaces = []
    
    # Use zip to iterate through all rows in the current batch simultaneously
    for i in range(len(examples['v_call_alpha'])):
        # Extract values for the current row
        v_a = str(examples['v_call_alpha'][i])
        j_aa_a = str(examples['junction_aa_alpha'][i])
        j_call_a = str(examples['j_call_alpha'][i])
        
        v_b = str(examples['v_call_beta'][i])
        j_aa_b = str(examples['junction_aa_beta'][i])
        j_call_b = str(examples['j_call_beta'][i])
        
        # Format the amino acids: "C A S S" (space between each letter)
        # We treat Gene IDs as single "words" (tokens)
        aa_alpha_spaced = " ".join(list(j_aa_a))
        aa_beta_spaced = " ".join(list(j_aa_b))
        
        # Construct the full string:
        # [GENE_V_A] [CDR3_A_SPACED] [GENE_J_A] [SEP] [GENE_V_B] [CDR3_B_SPACED] [GENE_J_B]
        full_seq = (
            f"{v_a} {aa_alpha_spaced} {j_call_a} "
            f"{tokenizer.sep_token} "
            f"{v_b} {aa_beta_spaced} {j_call_b}"
        )
        
        sequences_with_spaces.append(full_seq)

    tokenizer_outputs = tokenizer(sequences_with_spaces, truncation=True, max_length=512, padding=False, add_special_tokens=False)
    
    return tokenizer_outputs

def tokenize_mhc_epi(examples, tokenizer=None):
    sequences_with_spaces = []
    
    # Use zip to iterate through all rows in the current batch simultaneously
    for i in range(len(examples['gene'])):
        # Extract values for the current row
        gene = str(examples['gene'][i])
        epitope = str(examples['peptide'][i])
        epitope_aa_spaced = " ".join(list(epitope))
        
        # Construct the full string:
        # [MHC_gene] [epitope]
        full_seq = f"{gene} {epitope_aa_spaced}"
        
        sequences_with_spaces.append(full_seq)

    tokenizer_outputs = tokenizer(sequences_with_spaces, truncation=True, max_length=512, padding=False, add_special_tokens=False)
    
    return tokenizer_outputs

def train(device):
    is_head_proc = not use_ddp or dist.get_rank() == 0

    

    #tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')
    tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t30_150M_UR50D')
    
    #ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    #ds = load_dataset("bloyal/uniref50", split="train", streaming=False)

    # TCR alpha + beta with genes
    
    df = pd.read_csv('../alpha_beta_gene_mlm/ab_gene_processed.csv')
    ds = Dataset.from_pandas(df)
    gene_cols = ['v_call_beta', 'j_call_beta', 'v_call_alpha', 'j_call_alpha']

    all_genes = []
    for col in gene_cols:
        print(col)
        for gene in df[col].value_counts().keys():
            all_genes.append(gene)
    
    num_added_tokens = tokenizer.add_tokens(all_genes)
    print(f"Added {num_added_tokens} tokens")
    tokenizer.add_special_tokens({"sep_token": "<sep>"})
    assert "<sep>" in tokenizer.get_vocab(), "Failed to add <sep> token"


    # pMHC
    '''
    df = pd.read_csv('./data/pMHC/SysteMHC_2.0_deduped.csv')
    ds = Dataset.from_pandas(df)
    gene_col = 'gene'
    all_genes = []
    for gene in df[gene_col].value_counts().keys():
        all_genes.append(gene)

    num_added_tokens = tokenizer.add_tokens(all_genes)
    print(f"Added {num_added_tokens} tokens")
    '''

    #config = AutoConfig.from_pretrained("answerdotai/ModernBERT-base")
    config = AutoConfig.from_pretrained("facebook/esm2_t30_150M_UR50D")
    config.vocab_size = len(tokenizer)
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
            model = DDP(model, device_ids=[device], gradient_as_bucket_view=True, find_unused_parameters=True)
    
    do_compile=True

    if do_compile:
        if use_hpu:
            model = torch.compile(model, backend='hpu_backend')
            model_ema = torch.compile(model_ema, backend='hpu_backend')
        else:
            model = torch.compile(model)
            model_ema = torch.compile(model_ema)


    ds = ds.shuffle()
    if (use_ddp):
        ds = split_dataset_by_node(ds, rank=dist.get_rank(), world_size=dist.get_world_size())

    print(ds.info)

    #ds_len = ds.info.splits['train'].num_examples
    ds_len = len(ds)

    #ds = ds.to_iterable_dataset(num_shards=40)
    #ds = ds.map(lambda x: tokenizer(x['text'], truncation=True, max_length=512, padding=False, add_special_tokens=False), batched=True, remove_columns=ds.column_names)
    #ds = ds.map(lambda x: tokenizer(x['text'], truncation=True, max_length=512, padding=False, add_special_tokens=False), batched=True, num_proc=40, remove_columns=ds.column_names)

    ds = ds.map(partial(tokenize_function_a_b_gene, tokenizer=tokenizer), batched=True, num_proc=128, remove_columns=ds.column_names)
    #ds = ds.map(partial(tokenize_mhc_epi, tokenizer=tokenizer), batched=True, num_proc=128, remove_columns=ds.column_names)
    
    
    #if use_ddp:
    #    ds_len = ds_len // dist.get_world_size()


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


    num_epochs = 10
    batch_size = 768
    grad_accum_iters = 1
    learning_rate = 3e-4

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
        train_dataloader = getDataLoader(ds, batch_size, epoch, collate_fn=joint_collate, prefetch_factor=3)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=learning_rate, 
            #steps_per_epoch=len(train_dataloader),
            steps_per_epoch = ds_len // batch_size,
            epochs=num_epochs,
            pct_start=0.1
        )
        lr_scheduler.last_epoch = (ds_len // batch_size) * epoch
        start_time = time.time()
        toks_elapsed = 0
        for i, (full_batch, mlm_batch) in enumerate(train_dataloader):
            full_batch = {k: v.to(device, non_blocking=True) for k, v in full_batch.items()}
            mlm_batch = {k: v.to(device, non_blocking=True) for k, v in mlm_batch.items()}

            is_optim_step_iter = (i+1) % grad_accum_iters == 0
            ddp_sync_context = contextlib.nullcontext() if is_optim_step_iter or not use_ddp else model.no_sync()
            with torch.amp.autocast(device.type, enabled=True, dtype=torch.bfloat16), ddp_sync_context:
                outputs = model(**mlm_batch, output_hidden_states=True)
                student_repr = outputs.hidden_states[-1][mlm_batch['labels'] != -100].flatten(end_dim=-2)

                with torch.no_grad():
                    outputs_ema = model_ema(**full_batch, output_hidden_states=True)
                    teacher_repr = outputs_ema.hidden_states[-1][mlm_batch['labels'] != -100].flatten(end_dim=-2)


                loss_mlm = outputs.loss
                loss_mse = F.mse_loss(student_repr, teacher_repr)
                loss_ko_leo = ko_leo_loss(student_repr)

                loss = loss_mlm + 1.0 * loss_mse + 0.1 * loss_ko_leo


                scaler.scale(loss).backward()
                perplexity = math.exp(loss_mlm.detach().item())

                toks_elapsed = toks_elapsed + full_batch['attention_mask'].sum().item()

                if is_optim_step_iter:
                    if not use_hpu and use_ddp:
                        torch.cuda.synchronize()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                    if do_compile:
                        model_ema._orig_mod.update(model)
                    else:
                        model_ema.update(model)

                lr_scheduler.step()
            if (i+1) % grad_accum_iters == 0 and is_head_proc:
                toks_per_sec = toks_elapsed / (time.time() - start_time)
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {i + 1}/{ds_len // batch_size} | Loss (MLM): {loss_mlm.item():.4f} | PPLX: {perplexity:.4f} | Loss (MSE): {loss_mse.item():.4f} | Loss (KoLeo): {loss_ko_leo.item():.4f} | toks/s: {toks_per_sec:.2f}")
                toks_elapsed = 0
                start_time = time.time()
                sys.stdout.flush()
            #torch.cuda.empty_cache()
            #p.step()
        torch.cuda.synchronize()
        if is_head_proc:
            checkpoint_name = f"ESM2_150M_TCR_a_b_gene_10epoch_MSE1.0_KoLeo0.1_SepToken/epoch_{epoch}"
            if use_ddp:
                model.module.save_pretrained(f"./outputs/{checkpoint_name}")
            else:
                model.save_pretrained(f"./outputs/{checkpoint_name}")
            tokenizer.save_pretrained(f"./outputs/{checkpoint_name}")
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