"""
SentenceTransformer embedder and utility functions
"""

import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from .base import Embedder
from model_utils import tokenize_data_batched
from ddp_utils import is_main

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
            texts = [rec.chat_text for rec in batch_recs]
            
            with torch.no_grad():
                emb = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
                
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


class SentenceTransformerEmbedder(Embedder):
    """
    SentenceTransformerEmbedder class for SentenceTransformer models
    - Provides embeddings from pre-trained sentence transformer models
    """
    def __init__(self, model_path=None, **kwargs):
        if model_path is not None:
            self.model = SentenceTransformer(model_path)
        else:
            self.model = kwargs.get('model', None)
    
    def compute_embeddings(self, data, **kwargs):
        return compute_st_embeddings_ddp(self.model, data, **kwargs)