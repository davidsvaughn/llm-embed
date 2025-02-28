import os, sys
import numpy as np
from tqdm import tqdm
import copy
import traceback
import argparse

import torch
import torch.distributed as dist

try:
    from exllamav2 import model_init
    from exllamav2.generator import ExLlamaV2BaseGenerator
except ImportError:
    print('WARNING: ExLlamaV2 not installed')

from util import to_adict, fix_repeats
from model_utils import load_model, load_chat_tokenizer, tokenize_data_batched, apply_chat_template_batched
from ddp_utils import is_main
import torch.nn.functional as F

#===================================================================================================
''' Embedding Layer Aggregation Functions : currently only supports ExLlamaV2 embeddings'''

def select_layer_indices(M, N, method=1, seed=1234):
    """
    Embedding Aggregation Helper Function
    - Implements multiple algorithms for randomly selecting layer indices
    - Based on random seed, so every call with same seed will return same layer indices selections
    - M is the number of layers, N is the number of embeddings dimensions in each layer
    
    Args:
        M (int): Number of layers.
        N (int): Number of embedding dimensions in each layer.
        method (int): Method for selecting layer indices. Possible values are:
            0 - Linear probability distribution.
            1 - Uniform random selection.
            2 - Fixed number of random indices per layer.
            3 - Combination of uniform random selection and fixed number of random indices.
        seed (int): Random seed for reproducibility.
    
    Returns:
        list: List of selected layer indices.
    """
    np.random.seed(seed)
    
    if method == 0:
        p = np.linspace(1, min(5, M), M)
        p /= np.sum(p)
        r = np.random.choice(M, size=N, p=p)
        I = [r == i for i in range(M)]
    
    elif method == 1:
        r = np.random.randint(0, M, size=N)
        I = [r == i for i in range(M)]
    
    elif method == 2:
        Q = 1.5
        I, s = [], int(Q * N // M)
        for i in range(M):
            p = np.random.permutation(N)
            I.append(p[:s])

    elif method == 3:
        r = np.random.RandomState(seed).randint(0, M, size=N)
        I, s = [], int(1 * N//M)
        for i in range(M):
            v = np.where(r==i)[0]
            p = np.random.RandomState(seed+i).permutation(N)[:s]
            p = np.unique(np.concatenate((v,p)))
            I.append(p)
            
    return I

def aggregate_layers(L, **kwargs):
    """
    Embedding Aggregation Functions
    - Takes a dictionary of layer embeddings and aggregates them
    - Outputs a single numpy array of embeddings
    """
    X = list(L.values())
    M, N = len(X), X[0].shape[1]
    I = select_layer_indices(M, N, **kwargs)
    return np.concatenate([x[:, i] for x, i in zip(X, I)], axis=1)

            
#===================================================================================================

''' ExLlamaV2 Embedding Extractor '''

'''
some changes had to be made to: /home/azureuser/llm-embed/exllamav2/exllamav2/model.py#L850
might need to mokey patch...  only if we want multiple layers of embeddings... last layers already works

        def forward(self,
        .....
            if "extract_state_indices" in kwargs:
                # return result.get("logits"), result["states"]
                return result.get("logits"), {k: result["states"][k] for k in kwargs["extract_state_indices"] if k in result["states"]}
            elif "last_state" in result:
                return result.get("logits"), result["last_state"]
            else:
                return result.get("logits")
'''

class ExLlamaV2EmbeddingGenerator(ExLlamaV2BaseGenerator):
    """Generator class for extracting embeddings from ExLlamaV2 models"""
    
    def __init__(self, model, tokenizer, cfg, chat_tokenizer, batch_size=16):
        super().__init__(model, None, tokenizer)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.chat_tokenizer = chat_tokenizer
        self.cfg = cfg
        self.batch_size = batch_size

    def _compute_embedding(self, ids, hidden, position_offsets, mode='last', debug=False):
        """
        Compute embedding from hidden state using specified pooling mode.
        
        Args:
            ids (torch.Tensor): Token IDs tensor of shape (batch_size, seq_len)
            hidden (torch.Tensor): Hidden state tensor of shape (batch_size, seq_len, hidden_dim)
            position_offsets: Position offsets for batched inputs
            mode: Pooling mode ('last' or 'mean')
                debug: Whether to print debug information
            inputs (list or str): List of strings to embed or a single string.
            
        Returns:
            Tensor containing embeddings
        """
        if mode == 'last':
            # Last Token Embedding
            return hidden[:, -1, :]
        
        # Mean Pooled - weighted sum of embeddings
        pad_mask = (ids != self.tokenizer.pad_token_id).long().to(hidden.device)
        weights = pad_mask / pad_mask.sum(dim=1).unsqueeze(-1)
        return torch.sum(torch.nan_to_num(hidden, nan=0.0) * weights.unsqueeze(-1), dim=1)

    def _compute_embeddings(self, 
                           inputs, 
                           modes=['last', 'mean'],
                           return_token=False, 
                           debug=False, 
                           **kwargs):
        """
        Compute embeddings for a batch of inputs.
        
        Args:
            inputs: List of strings to embed
            modes: List of pooling modes to use
            return_token: Whether to return next predicted token
            debug: Whether to print debug information
            
        Returns:
            Numpy array of embeddings or dictionary of layer embeddings
        """
        batch_size = 1 if isinstance(inputs, str) else len(inputs)
        prompts_identical = batch_size == 1 or all(s == inputs[0] for s in inputs)

        # Tokenize inputs - ExLlamaV2 uses its own tokenizer
        ids, position_offsets = self.tokenizer.encode(inputs,
                                                     encode_special_tokens=False,
                                                     return_offsets=True,
                                                     add_bos=False)
        if prompts_identical:
            position_offsets = None
        mask = self.tokenizer.padding_mask(ids) if batch_size > 1 else None

        # Forward pass with error handling
        try:
            logits, hidden = self.model.forward(ids,
                                               input_mask=mask,
                                               position_offsets=position_offsets,
                                               return_last_state=True,
                                               **kwargs)
        except Exception as e1:
            print(f"ERROR: {e1}")
            traceback.print_exc()
            print(f"\t====> Error in forward pass: ids.shape={ids.shape}")
            
            # Handle degenerating inputs by fixing repeats and potentially truncating
            try:
                # Find the longest input
                idx = torch.where(position_offsets == 0)[0].item()
                len1 = len(inputs[idx])
                
                # Fix repeating patterns in all inputs
                inputs = [fix_repeats(p) for p in inputs]
                len2 = len(inputs[idx])
                
                # If fixing didn't reduce size significantly, truncate to median length
                if len2/len1 > 0.8:
                    mlen = sorted([len(p) for p in inputs])[len(inputs)//2]
                    inputs[idx] = inputs[idx][:mlen]

                # Re-encode the fixed/truncated inputs
                ids, position_offsets = self.tokenizer.encode(inputs,
                                                            encode_special_tokens=False,
                                                            return_offsets=True,
                                                            add_bos=False)
                print(f"\t====> Now ids.shape={ids.shape}")
                mask = self.tokenizer.padding_mask(ids) if batch_size > 1 else None
                
                # Try forward pass again
                logits, hidden = self.model.forward(ids,
                                                  input_mask=mask,
                                                  position_offsets=position_offsets,
                                                  return_last_state=True,
                                                  **kwargs)
            except Exception as e2:
                print(f"ERROR: {e2}")
                traceback.print_exc()
                
                # More detailed error information
                idx = torch.where(position_offsets == 0)[0].item()
                print(f'longest input len = {len(inputs[idx])}')
                print('-----------------------------------------------------------------')
                print(inputs[idx])
                print('-----------------------------------------------------------------')
                print(f"Prompt: {self.tokenizer.decode(ids[idx])}")
                sys.exit(1)

        # Process embeddings based on hidden state format
        if not isinstance(hidden, dict):
            # Single hidden state - process according to requested modes
            xx = []
            for mode in modes:
                xx.append(self._compute_embedding(ids, hidden, position_offsets, mode=mode, debug=debug))
            x = torch.cat(xx, dim=-1).cpu().numpy()
        else:
            # Multiple hidden states (layer outputs)
            x = {}
            for key, value in hidden.items():
                xx = []
                for mode in modes:
                    xx.append(self._compute_embedding(ids, value, position_offsets, mode=mode, debug=debug))
                x[key] = torch.cat(xx, dim=-1).cpu().numpy()

        # Optionally include predicted next token
        if return_token:
            tok_ids = logits[:, -1, :].argmax(-1)
            x = (x, tok_ids)

            if debug:
                next_tokens = self.tokenizer.decode(tok_ids[:,None])
                for i, token in enumerate(next_tokens):
                    print(f"Prompt {i+1}: {inputs[i]}")
                    print(f"Next Token: {token}\n")
        
        return x
    
    def compute_embeddings(self, data, **kwargs):
        """
        Compute embeddings for a dataset.
        
        Args:
            data: Dictionary of items to records
            
        Returns:
            Dictionary containing embeddings, scores and indices
        """
        emb_data = {}
        disable_outer = len(data) < 2
        
        for item, records in tqdm(data.items(), desc="Computing embeddings", disable=disable_outer):
            # Apply chat template
            inputs = apply_chat_template_batched(self.chat_tokenizer, records)
            
            # Compute embeddings in batches
            x = []
            batch_range = range(0, len(inputs), self.batch_size)
            for i in tqdm(batch_range, desc="Processing batches", disable=not disable_outer):
                batch_inputs = inputs[i:i+self.batch_size]
                x.append(self._compute_embeddings(batch_inputs, **kwargs))
            
            # Process results based on whether we have multi-layer embeddings
            if isinstance(x[0], dict):
                # Merge dictionaries for multi-layer embeddings
                x = {k: np.concatenate([xx[k] for xx in x], axis=0) for k in x[0]}
            else:
                x = np.concatenate(x, axis=0)
                
            # Verify output dimensions match input records
            assert len(next(iter(x.values())) if isinstance(x, dict) else x) == len(records)
            
            # Store embeddings and metadata
            emb_data[item] = {
                'x': x, 
                'y': np.array([rec.score for rec in records]),
                'idx': np.array([rec.index for rec in records]),
            }
            
        return emb_data
        

''' HuggingFace Embedding Extractor '''
def _get_hf_embedding(last_hidden,
                      attention_mask,
                      pooling_strategy="mean",
                      padding_side="left",
                      last_token_offset=0):
    # embedding_mask is attention_mask with the last 4 tokens zeroed out (when score included)
    last_token_index = -1 - last_token_offset

    if pooling_strategy == "last":
        if padding_side == "left":
            output = last_hidden[:, last_token_index]
        elif padding_side == "right":
            batch_size = last_hidden.size(0)
            batch_indices = torch.arange(batch_size, device=last_hidden.device)
            last_token_positions = attention_mask.sum(dim=1) + last_token_index
            output = last_hidden[batch_indices, last_token_positions]
        else:
            raise ValueError(f"Padding side '{padding_side}' not recognized.")
            
    elif pooling_strategy == "mean":
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
        raise ValueError(f"Pooling strategy '{pooling_strategy}' not recognized.")
    
    return output.to(dtype=last_hidden.dtype)


def get_hf_embedding(model, 
                     input_ids, 
                     attention_mask,
                     labels=None,
                     pooling_strategy="mean",
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
    
    output = [_get_hf_embedding(last_hidden, attention_mask, ps, padding_side, last_token_offset) for ps in pooling_strategy.split(",")]
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
            # dist.barrier() 
            dist.barrier(device_ids=[torch.cuda.current_device()])
    
    if verbose and rank == 0:
        # Final check of data
        for item in emb_data:
            print(f"\nFinal check - Item {item}:")
            print(f"X shape: {emb_data[item]['x'].shape if isinstance(emb_data[item]['x'], np.ndarray) else 'empty list'}")
            print(f"Y shape: {emb_data[item]['y'].shape if isinstance(emb_data[item]['y'], np.ndarray) else 'empty list'}")
            print(f"IDX shape: {emb_data[item]['idx'].shape if isinstance(emb_data[item]['idx'], np.ndarray) else 'empty list'}")
    
    return emb_data

#===================================================================================================

def compute_st_embeddings_ddp(model, data, verbose=False, **kwargs):
    """
    Compute sentence-transformer embeddings for the provided dataset using Distributed Data Parallel (DDP).
    This function processes input data by tokenizing it and computing embeddings using the provided model. It supports both distributed and single-GPU modes. In DDP mode, the records are split across multiple GPUs, where each GPU processes its local batches and the results are gathered and ordered by the process with rank 0. In single-GPU mode, all processing is performed locally.
    Parameters:
        model (torch.nn.Module): The model used for encoding text into embeddings. This model should offer an 'encode' method
                                 and may optionally include a 'tokenizer' attribute.
        data (dict): A dictionary mapping item identifiers to collections of records. Each record is expected to contain
                     attributes like 'batch_index', 'chat_text', 'score', and 'index'. Tokenization is applied to these records.
        verbose (bool, optional): If True, prints detailed logging information during processing. Defaults to False.
        **kwargs: Additional keyword arguments. Notably, this includes:
            - 'tokenizer': A tokenizer to use if the model does not already have one.
            - Other parameters that may be used by the 'tokenize_data_batched' function during tokenization.
    Returns:
        dict: A dictionary (emb_data) where each key corresponds to an input item and maps to another dictionary with:
            - 'x' (numpy.ndarray): The computed embeddings ordered based on the original record indexing.
            - 'y' (numpy.ndarray): The scores associated with each record.
            - 'idx' (numpy.ndarray): The original indices of the records.
    Notes:
        - The function first checks for and applies tokenization using either the model's tokenizer or the one provided in kwargs.
        - In DDP mode, batches are distributed across GPUs based on the 'batch_index' modulo the world size.
        - The rank 0 process gathers embeddings from all other processes, orders the results, and constructs the final arrays.
        - Synchronization is performed at the end of processing each item to ensure all distributed processes are aligned.
    """
    emb_data = {}
    is_distributed = dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    world_size = dist.get_world_size() if is_distributed else 1
    
    #---------------------------------------------------------------------------------------
    ''' Tokenize data here (if not already done...) '''
    
    if hasattr(model, 'tokenizer'):
        tokenizer = model.tokenizer
    else:
        tokenizer = kwargs.get('tokenizer', None)
    assert tokenizer is not None, "Tokenizer must be provided!"
    
    # -- first applies tokenizer's chat_template to messages...
    tokenize_data_batched(data, tokenizer=tokenizer, **kwargs)
    
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
            # input_ids = torch.stack([rec.tokenized_text['input_ids'] for rec in batch_recs]).to(f"cuda:{rank}")
            # attention_mask = torch.stack([rec.tokenized_text['attention_mask'] for rec in batch_recs]).to(f"cuda:{rank}")
            texts = [rec.chat_text for rec in batch_recs]
            # texts = torch.stack(texts).to(f"cuda:{rank}")
            
            with torch.no_grad():
                emb = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
                # emb = emb.to(torch.float32)
                
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
            # dist.barrier() 
            dist.barrier(device_ids=[torch.cuda.current_device()])
    
    if verbose and rank == 0:
        # Final check of data
        for item in emb_data:
            print(f"\nFinal check - Item {item}:")
            print(f"X shape: {emb_data[item]['x'].shape if isinstance(emb_data[item]['x'], np.ndarray) else 'empty list'}")
            print(f"Y shape: {emb_data[item]['y'].shape if isinstance(emb_data[item]['y'], np.ndarray) else 'empty list'}")
            print(f"IDX shape: {emb_data[item]['idx'].shape if isinstance(emb_data[item]['idx'], np.ndarray) else 'empty list'}")
    
    return emb_data

#===================================================================================================
''' Embedder Base Class '''

class Embedder:
    def __init__(self, cfg=None, **kwargs):
        if cfg is not None:
            self.cfg = cfg
            self.model_id = cfg.model_id
            # self.norm = cfg.normalize if 'normalize' in cfg else False
            
    def get_embedding(self, input_ids, attention_mask, **kwargs):
        pass
    
    def compute_embeddings(self, data, **kwargs):
        pass

#---------------------------------------------------------------------------------------
from sentence_transformers import SentenceTransformer 

class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_path=None, **kwargs):
        if model_path is not None:
            self.model = SentenceTransformer(model_path)
        else:
            self.model = kwargs.get('model', None)
    
    def compute_embeddings(self, data, **kwargs):
        
        return compute_st_embeddings_ddp(self.model, data, **kwargs)

#---------------------------------------------------------------------------------------
''' HuggingfaceEmbedder class for Huggingface LLM models : Can be used for Single-GPU or DDP/Multi-GPU inference 
    - especially useful during fine-tuning, where multi-GPU inference speeds up inner validation loop
    - can also be run outside of fine-tuning
    - currently only returns a single hidden layer, either last-token or mean-pooled embeddings
'''
class HuggingfaceEmbedder(Embedder):
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

#---------------------------------------------------------------------------------------
''' ExLlamaV2Embedder class for ExLlamaV2 LLM models : Only single-GPU mode 
    - targets production setting: single-GPU inference using quantized ExLlamaV2 models
    - can return embeddings from multiple layers, aggregated using different randomization methods
    - supports both last-token and mean-pooled embeddings, as well as concatenation of both
'''
class ExLlamaV2Embedder(Embedder):
    # some of this copied from: exllamav2/test_inference.py 
    def __init__(self, max_seq_len=8192, batch_size=16, **cfg):
        super().__init__(to_adict(cfg))
        parser = argparse.ArgumentParser(description = "Test inference on ExLlamaV2 model")
        parser.add_argument("-er", "--eval_rows", type = int, default = 128, help = "Number of rows to apply from dataset")
        parser.add_argument("-el", "--eval_length", type = int, default = 4096, help = "Max no. tokens per sample")
        parser.add_argument("-pnb", "--prompt_no_bos", action = "store_true", help = "Don't add BOS token to prompt")
        parser.add_argument("-t", "--tokens", type = int, default = 128, help = "Max no. tokens")
        parser.add_argument("-nwu", "--no_warmup", action = "store_true", help = "Skip warmup before testing model")
        model_init.add_args(parser)
        self.args = args = parser.parse_args()
        args.model_dir = self.model_id
        model_init.check_args(args)
        # model_init.print_options(args)

        # initialize exl2 model
        model, tokenizer = model_init.init(
            args,
            max_input_len = max_seq_len * batch_size,
        )
        
        # get huggingface chat tokenizer
        chat_tokenizer = load_chat_tokenizer(self.model_id)

        # initialize embedder (subclass of ExLlamaV2BaseGenerator)
        self.embedder = ExLlamaV2EmbeddingGenerator(model, tokenizer, cfg, chat_tokenizer, batch_size=batch_size)
        self.embedder.warmup()
        
        # multi-layer embeddings...
        # requires 'layers' or 'extract_state_indices' key in cfg, where value is a list of layer indices
        # for example: cfg.layers = [41, 48, 57]
        if 'layers' in self.cfg:
            self.cfg.extract_state_indices = self.cfg.layers

    # data is a dictionary of item to record list
    def compute_embeddings(self, data, **kwargs):
        emb_data = self.embedder.compute_embeddings(data, **{**self.cfg, **kwargs})
        
        # possibly aggregate embeddings from multiple layers, aggregated using different randomization methods
        # cfg.method = 0, 1, 2, or 3 (default is 1)
        if 'extract_state_indices' in self.cfg:
            for item, data in emb_data.items():
                data['x'] = aggregate_layers(data['x'], **self.cfg)
                
        return emb_data
        
#===================================================================================================

''' Embedder Factory : returns either HuggingfaceEmbedder or ExLlamaV2Embedder based on model_id '''
def EmbedderFactory(cfg=None, model_id=None, model_type='hf'): # model_type = 'hf' or 'st' or 'exl2'
    if model_type == 'st':
        return SentenceTransformerEmbedder(model_id)
    
    if cfg is None:
        cfg = to_adict({ 'model_id': model_id })
    else:
        if model_id is None:
            model_id = cfg.get('model_id', None)
        else:
            # neither is None, so override cfg model_id with model_id
            cfg = to_adict(copy.deepcopy(cfg))
            cfg.model_id = model_id
            
    if model_id is None:
        raise ValueError("model_id must be provided")
    
    if 'exl2' in model_id or model_type == 'exl2':
        # ExLlamaV2 LLM models
        return ExLlamaV2Embedder(**cfg)

    # Huggingface LLM models
    return HuggingfaceEmbedder(cfg)
