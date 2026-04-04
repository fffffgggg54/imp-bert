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

from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoModelForMaskedLM, AutoModel, AutoConfig, AutoTokenizer, DataCollatorForLanguageModeling


use_ddp=False
use_hpu=True

if use_hpu:
    from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

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

    config = AutoConfig.from_pretrained("answerdotai/ModernBERT-base")
    model = AutoModelForMaskedLM.from_config(
        config,
        attn_implementation='sdpa'
        #attn_implementation="kernels-community/flash-attn3"
    ).to(device)

    if (use_ddp):
        if use_hpu:
            model = DDP(model)
        else:
            model = DDP(model, device_ids=[device], gradient_as_bucket_view=True, find_unused_parameters=False)

    if use_hpu:
        model = torch.compile(model, backend='hpu_backend')
    else:
        model = torch.compile(model)

    tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')
    
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)

    if (use_ddp):
        ds = split_dataset_by_node(ds, rank=dist.get_rank(), world_size=dist.get_world_size())

    ds = ds.map(lambda x: tokenizer(x['text'], truncation=True, max_length=512, padding=False, add_special_tokens=False), batched=True, remove_columns=ds.column_names)

    ds_len = ds.info.splits['train'].num_examples
    if use_ddp:
        ds_len = ds_len // dist.get_world_size()


    mlm_data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
        mask_replace_prob=0.7,
        random_replace_prob=0.3,
        return_tensors='pt'
    )
    num_epochs = 5
    batch_size = 1
    grad_accum_iters = 16
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
        train_dataloader = getDataLoader(ds, batch_size, epoch, collate_fn=mlm_data_collator)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=learning_rate, 
            #steps_per_epoch=len(train_dataloader),
            steps_per_epoch = ds_len // batch_size,
            epochs=num_epochs,
            pct_start=0.1
        )
        lr_scheduler.last_epoch = (ds_len // batch_size) * epoch
        for i, batch in enumerate(train_dataloader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            is_optim_step_iter = (i+1) % grad_accum_iters == 0
            ddp_sync_context = contextlib.nullcontext() if is_optim_step_iter or not use_ddp else model.no_sync()
            with torch.amp.autocast(device.type, enabled=True, dtype=torch.bfloat16), ddp_sync_context:
                outputs = model(**batch, output_hidden_states=True)
                loss_mlm = outputs.loss

                #loss = 1.0 * loss_mlm + 1.0 * loss_distogram + 2.0 * structure_loss + 0.5 * loss_distogram_self_distill * int(epoch > 0)

                scaler.scale(loss_mlm).backward()
                perplexity = math.exp(loss_mlm.detach().item())

                if is_optim_step_iter:
                    torch.cuda.synchronize()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                lr_scheduler.step()
            if (i+1) % grad_accum_iters == 0 and is_head_proc:
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {i + 1}/{ds_len // batch_size} | Loss (MLM): {loss_mlm.item():.4f} | PPLX: {perplexity:.4f}")
                sys.stdout.flush()
            #torch.cuda.empty_cache()
            #p.step()
        torch.cuda.synchronize()
        if is_head_proc:
            checkpoint_name = f"scratch/epoch_{epoch}"
            if use_ddp:
                model.module.save_pretrained(f"./outputs/{checkpoint_name}")
            else:
                model.save_pretrained(f"./outputs/{checkpoint_name}")
            torch.save(optimizer.state_dict(), f"./outputs/{checkpoint_name}/optimizer.pth")




def main():
    if use_ddp:
        torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
        acc = torch.accelerator.current_accelerator()
        backend = torch.distributed.get_default_backend_for_device(acc)
        dist.init_process_group(backend)
        rank = dist.get_rank()
        
        if use_hpu:
            device = torch.device('hpu')
        else:
            device = rank % torch.accelerator.device_count()
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train(device)

    if use_ddp:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()