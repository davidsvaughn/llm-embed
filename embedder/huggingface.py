"""
HuggingFace embedder and utility functions
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from .base import Embedder
from utils import to_adict
from model_utils import load_model, tokenize_data_batched
from ddp_utils import is_main

def _get_hf_embedding(last_hidden,
                     attention_mask,
                     pooling_mode="mean",
                     padding_side="left",
                     last_token_offset=0):
    # embedding_mask is attention_mask with the last 4 tokens zeroed out (when score included)
    last_token_index = -1 - last_token_offset

    if pooling_mode == "lasttoken":
        if padding_side == "left":
            output = last_hidden[:, last_token_index]
        elif padding_side == "right":
            batch_size = last_hidden.size(0)
            batch_indices = torch.arange(batch_size, device=last_hidden.device)
            last_token_positions = attention_mask.sum(dim=1) + last_token_index
            output = last_hidden[batch_indices, last_token_positions]
        else:
            raise ValueError(f"Padding side '{padding_side}' not recognized.")
            
    elif pooling_mode == "mean":
        embedding_mask = attention_mask.clone()
        if padding_side == "left":
            if last_token_offset>0:
                embedding_mask[:, -last_token_offset:] = 0
        elif padding_side == "right":
            if last_token_offset>0:
                last_token_positions = embedding_mask.sum(dim=1) + last_token_index
                for i, pos in enumerate(last_token_positions):
                    embedding_mask[i, pos+1:] = 0
        else:
            raise ValueError(f"Padding side '{padding_side}' not recognized.")

        weights = embedding_mask / embedding_mask.sum(dim=1).unsqueeze(-1)
        output = torch.sum(last_hidden * weights.unsqueeze(-1), dim=1)
        
    else:
        raise ValueError(f"pooling_mode '{pooling_mode}' not recognized.")
    
    return output.to(dtype=last_hidden.dtype)


def get_hf_embedding(model, 
                     input_ids, 
                     attention_mask,
                     labels=None,
                     pooling_mode="mean",
                     padding_side="left",
                     hidden_layer=-1, 
                     last_token_offset=0,
                     **kwargs):
    """
    Works with both left and right padding.
    """
    if labels is not None:
        if isinstance(labels, bool):
            if labels: # if labels is True, set labels to input_ids where attention_mask is 1
                labels = input_ids.masked_fill(attention_mask == 0, -100)
            else: # if labels is False, set labels to None
                labels = None

    # Get model outputs
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        output_hidden_states=True,
        return_dict=True
    )
    attention_mask = attention_mask.to(dtype=outputs.hidden_states[-1].dtype)
    
    loss = outputs.loss if labels is not None else None

    # Get the last hidden state - default is -1, i.e. the last (top) layer
    last_hidden = outputs.hidden_states[hidden_layer]  # [batch_size, seq_len, hidden_size]
    
    output = [_get_hf_embedding(last_hidden, attention_mask, ps, padding_side, last_token_offset) for ps in pooling_mode.split(",")]
    output = torch.cat(output, dim=-1)

    if labels is None:
        return output
    return output, loss


def compute_hf_embeddings_ddp(model, data, hidden_layer=-1, verbose=False, **kwargs):
    """
    Computes embeddings for the provided data using a Hugging Face model, supporting both single GPU and distributed data parallel (DDP) configurations.
    This function tokenizes the input data (if not already tokenized) using the model's tokenizer or a provided tokenizer, processes the data in batches, and extracts embeddings from a specified hidden layer. In DDP mode, the batches are split across GPUs and the results are gathered on the main process (rank 0), ensuring the embeddings are ordered correctly. 
    Parameters:
        model (torch.nn.Module): The model used for embedding extraction. It should have a callable method for embedding extraction (via get_hf_embedding) and ideally a 'tokenizer' attribute.
        data (dict): A dictionary where each key maps to a list of records. Each record is expected to have at least the following attributes:
                     - batch_index: An integer used to determine GPU assignment in DDP.
                     - tokenized_text: A dictionary containing pre-tokenized inputs (e.g., 'input_ids', 'attention_mask').
                     - score: A numerical score associated with the record.
                     - index: A unique identifier for the record.
        hidden_layer (int, optional): The index of the hidden layer from which to extract embeddings. Defaults to -1 (typically the last layer).
        verbose (bool, optional): If True, prints debug and status information during processing. Defaults to False.
        **kwargs: Additional keyword arguments. May include:
                  - tokenizer: An alternative tokenizer in case the model does not have a built-in tokenizer.
                  - other parameters needed for tokenization and embedding extraction.
    Returns:
        dict: A dictionary mapping each item from the input data to a dictionary with the following keys:
              - 'x': A NumPy array containing the embeddings for all records, ordered according to their original indices.
              - 'y': A NumPy array with the scores corresponding to each record.
              - 'idx': A NumPy array with the unique identifiers of the records.
    Notes:
        - In distributed mode, this function communicates between GPUs using torch.distributed to gather embeddings on the main process.
        - A barrier synchronization is performed at the end of processing for each item.
        - It is assumed that auxiliary functions (e.g., tokenize_data_batched, get_hf_embedding) and necessary modules (e.g., torch, numpy, torch.distributed as dist, tqdm) are properly imported and configured outside of this function.
    """
    emb_data = {}
    is_distributed = dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    world_size = dist.get_world_size() if is_distributed else 1

    if verbose and rank == 0:
        print("Running DDP version of compute_embeddings")
    
    #---------------------------------------------------------------------------------------
    ''' Tokenize data here (if not already done...) '''
    
    if hasattr(model, 'tokenizer'):
        tokenizer = model.tokenizer
    else:
        tokenizer = kwargs.get('tokenizer', None)
    assert tokenizer is not None, "Tokenizer must be provided!"
    
    # -- first applies tokenizer's chat_template to messages...
    tokenize_data_batched(data, tokenizer, **kwargs)
    
    #---------------------------------------------------------------------------------------
    disable = len(data) < 2 or not is_main()
    for item, records in tqdm(data.items(), desc="Getting embeddings", disable=disable):
        if verbose and rank == 0:
            print(f"\nProcessing item {item} with {len(records)} records")
        
        emb_data[item] = {'x': [], 'y': [], 'idx': []}
        
        # Split batches across GPUs if using DDP
        batches = {}
        batch_to_global_idx = {}
        for global_idx, rec in enumerate(records):
            batch_idx = rec.batch_index
            if batch_idx % world_size == rank:
                if batch_idx not in batches:
                    batches[batch_idx] = []
                    batch_to_global_idx[batch_idx] = []
                batches[batch_idx].append(rec)
                batch_to_global_idx[batch_idx].append(global_idx)
        
        if verbose and rank == 0:
            print(f"Rank {rank}: Processing {len(batches)} batches")
        
        # Process batches on each GPU and track ordering
        local_results = []
        
        for batch_idx, batch_recs in batches.items():
            input_ids = torch.stack([rec.tokenized_text['input_ids'] for rec in batch_recs]).to(f"cuda:{rank}")
            attention_mask = torch.stack([rec.tokenized_text['attention_mask'] for rec in batch_recs]).to(f"cuda:{rank}")
            
            with torch.no_grad():
                emb = get_hf_embedding(model, input_ids, attention_mask, hidden_layer=hidden_layer, **kwargs)
                emb = emb.to(torch.float32)
                
                global_indices = torch.tensor(batch_to_global_idx[batch_idx], device=emb.device)
                scores = torch.tensor([rec.score for rec in batch_recs], device=emb.device)
                uids = torch.tensor([rec.index for rec in batch_recs], device=emb.device)
                local_results.append((global_indices, emb, scores, uids))
        
        if verbose and rank == 0:
            print(f"Rank {rank}: Generated {len(local_results)} local results")
            
        if is_distributed:
            # Gather results from all processes
            if rank == 0:
                all_global_indices = []
                all_embeddings = []
                all_scores = []
                all_uids = []
                
                # Add local results first
                for global_indices, emb, scores, uids in local_results:
                    all_global_indices.extend(global_indices.cpu().numpy())
                    all_embeddings.append(emb.cpu().numpy())
                    all_scores.extend(scores.cpu().numpy())
                    all_uids.extend(uids.cpu().numpy())
                
                if verbose:
                    print(f"Rank 0: After local processing - {len(all_global_indices)} indices, {len(all_embeddings)} embedding batches")
                
                # Receive from other processes
                for src_rank in range(1, world_size):
                    # Get number of batches from this rank
                    num_batches = torch.zeros(1, dtype=torch.long, device=f"cuda:{rank}")
                    dist.recv(num_batches, src=src_rank)
                    num_batches = num_batches.item()
                    if verbose: print(f"Rank 0: Expecting {num_batches} batches from rank {src_rank}")
                    
                    for _ in range(num_batches):
                        # Receive batch size first
                        batch_size = torch.zeros(1, dtype=torch.long, device=f"cuda:{rank}")
                        dist.recv(batch_size, src=src_rank)
                        batch_size = batch_size.item()
                        
                        # Receive batch data
                        indices = torch.zeros(batch_size, dtype=torch.long, device=f"cuda:{rank}")
                        embeddings = torch.zeros((batch_size, emb.shape[1]), dtype=torch.float32, device=f"cuda:{rank}")
                        scores = torch.zeros(batch_size, dtype=torch.long, device=f"cuda:{rank}")
                        uids = torch.zeros(batch_size, dtype=torch.long, device=f"cuda:{rank}")
                        
                        dist.recv(indices, src=src_rank)
                        dist.recv(embeddings, src=src_rank)
                        dist.recv(scores, src=src_rank)
                        dist.recv(uids, src=src_rank)
                        
                        all_global_indices.extend(indices.cpu().numpy())
                        all_embeddings.append(embeddings.cpu().numpy())
                        all_scores.extend(scores.cpu().numpy())
                        all_uids.extend(uids.cpu().numpy())
                
                if verbose: print(f"Rank 0: After gathering - {len(all_global_indices)} indices, {len(all_embeddings)} embedding batches")
                
                if len(all_global_indices) > 0:
                    # Create properly ordered arrays
                    ordering = np.argsort(all_global_indices)
                    emb_data[item]['x'] = np.concatenate(all_embeddings)[ordering]
                    emb_data[item]['y'] = np.array(all_scores)[ordering]
                    emb_data[item]['idx'] = np.array(all_uids)[ordering]
                    if verbose: print(f"Rank 0: Final arrays - x shape: {emb_data[item]['x'].shape}, y shape: {emb_data[item]['y'].shape}, idx shape: {emb_data[item]['idx'].shape}")
                else:
                    if verbose: print("WARNING: No data collected!")
                    emb_data[item]['x'] = np.array([])
                    emb_data[item]['y'] = np.array([])
                    emb_data[item]['idx'] = np.array([])
                
            else:
                # Send number of batches first
                num_batches = torch.tensor([len(local_results)], device=f"cuda:{rank}")
                dist.send(num_batches, dst=0)
                if verbose: print(f"Rank {rank}: Sending {len(local_results)} batches to rank 0")
                
                # Send each batch
                for global_indices, emb, scores, uids in local_results:
                    batch_size = torch.tensor([len(global_indices)], device=f"cuda:{rank}")
                    dist.send(batch_size, dst=0)
                    dist.send(global_indices, dst=0)
                    dist.send(emb, dst=0)
                    dist.send(scores, dst=0)
                    dist.send(uids, dst=0)
            
        else:
            # Single GPU mode
            all_embeddings = []
            all_global_indices = []
            all_scores = []
            all_uids = []
            
            for global_indices, emb, scores, uids in local_results:
                all_global_indices.extend(global_indices.cpu().numpy())
                all_embeddings.append(emb.cpu().numpy())
                all_scores.extend(scores.cpu().numpy())
                all_uids.extend(uids.cpu().numpy())
                
            if len(all_global_indices) > 0:
                ordering = np.argsort(all_global_indices)
                emb_data[item]['x'] = np.concatenate(all_embeddings)[ordering]
                emb_data[item]['y'] = np.array(all_scores)[ordering]
                emb_data[item]['idx'] = np.array(all_uids)[ordering]
            else:
                emb_data[item]['x'] = np.array([])
                emb_data[item]['y'] = np.array([])
                emb_data[item]['idx'] = np.array([])
            
        # Synchronize processes
        if is_distributed:
            dist.barrier(device_ids=[torch.cuda.current_device()])
    
    if verbose and rank == 0:
        # Final check of data
        for item in emb_data:
            print(f"\nFinal check - Item {item}:")
            print(f"X shape: {emb_data[item]['x'].shape if isinstance(emb_data[item]['x'], np.ndarray) else 'empty list'}")
            print(f"Y shape: {emb_data[item]['y'].shape if isinstance(emb_data[item]['y'], np.ndarray) else 'empty list'}")
            print(f"IDX shape: {emb_data[item]['idx'].shape if isinstance(emb_data[item]['idx'], np.ndarray) else 'empty list'}")
    
    return emb_data


class HuggingfaceEmbedder(Embedder):
    """
    HuggingfaceEmbedder class for Huggingface LLM models: Can be used for Single-GPU or DDP/Multi-GPU inference 
    - especially useful during fine-tuning, where multi-GPU inference speeds up inner validation loop
    - can also be run outside of fine-tuning
    - currently only returns a single hidden layer, either last-token or mean-pooled embeddings
    """
    def __init__(self, cfg=None, model=None, tokenizer=None, **kwargs):
        if cfg is not None:
            super().__init__(cfg)
            self.model = load_model(cfg)
        else:
            self.model = model
            if self.model is not None and tokenizer is not None:
                self.model.tokenizer = tokenizer
            self.cfg = to_adict(kwargs)
        
    def get_embedding(self, input_ids, attention_mask, **kwargs):
        return get_hf_embedding(self.model, input_ids, attention_mask, **{**self.cfg, **kwargs})
        
    # data is a dictionary of { item : record list }
    def compute_embeddings(self, data, **kwargs):
        return compute_hf_embeddings_ddp(self.model, data, **{**self.cfg, **kwargs}) # calls get_hf_embedding internally...
