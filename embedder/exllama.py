"""
ExLlamaV2 embedder and utility functions
"""

import os
import sys
import traceback
import argparse
import numpy as np
import torch
from tqdm import tqdm

try:
    from exllamav2 import model_init
    from exllamav2.generator import ExLlamaV2BaseGenerator
except ImportError:
    print('WARNING: ExLlamaV2 not installed')

from .base import Embedder
from .aggregation import aggregate_layers
from utils import to_adict, fix_repeats
from model_utils import load_chat_tokenizer, apply_chat_template_batched

class ExLlamaV2EmbeddingGenerator(ExLlamaV2BaseGenerator):
    """Generator class for extracting embeddings from ExLlamaV2 models"""
    
    def __init__(self, model, tokenizer, cfg, chat_tokenizer, batch_size=16):
        super().__init__(model, None, tokenizer)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.chat_tokenizer = chat_tokenizer
        self.cfg = cfg
        self.batch_size = batch_size

    def _compute_embedding(self, ids, hidden, position_offsets, mode='lasttoken', debug=False):
        """
        Compute embedding from hidden state using specified pooling mode.
        
        Args:
            ids (torch.Tensor): Token IDs tensor of shape (batch_size, seq_len)
            hidden (torch.Tensor): Hidden state tensor of shape (batch_size, seq_len, hidden_dim)
            position_offsets: Position offsets for batched inputs
            mode: Pooling mode ('lasttoken' or 'mean')
                debug: Whether to print debug information
            inputs (list or str): List of strings to embed or a single string.
            
        Returns:
            Tensor containing embeddings
        """
        if mode == 'lasttoken':
            # Last Token Embedding
            return hidden[:, -1, :]
        
        # Mean Pooled - weighted sum of embeddings
        pad_mask = (ids != self.tokenizer.pad_token_id).long().to(hidden.device)
        weights = pad_mask / pad_mask.sum(dim=1).unsqueeze(-1)
        return torch.sum(torch.nan_to_num(hidden, nan=0.0) * weights.unsqueeze(-1), dim=1)

    def _compute_embeddings(self, 
                           inputs, 
                           modes=['lasttoken', 'mean'],
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


class ExLlamaV2Embedder(Embedder):
    """
    ExLlamaV2Embedder class for ExLlamaV2 LLM models: Only single-GPU mode 
    - targets production setting: single-GPU inference using quantized ExLlamaV2 models
    - can return embeddings from multiple layers, aggregated using different randomization methods
    - supports both last-token and mean-pooled embeddings, as well as concatenation of both
    """
    # some of this copied from: exllamav2/test_inference.py 
    def __init__(self, max_seq_len=8192, batch_size=16, **cfg):
        super().__init__(to_adict(cfg))
        
        args = argparse.Namespace()
        args.eval_rows = 128
        args.eval_length = 4096
        args.prompt_no_bos = False
        args.no_warmup = False
        args.tokens = 128

        # Add default model_init arguments
        parser = argparse.ArgumentParser()
        model_init.add_args(parser)
        default_args = parser.parse_args([])  # Empty list to avoid parsing actual command line args
        for key, value in vars(default_args).items():
            setattr(args, key, value)

        # Override with any model-specific config from self.cfg
        if hasattr(self.cfg, 'model_args'):
            for key, value in self.cfg.model_args.items():
                setattr(args, key, value)
        #------------------------------------------------------------------------------
        
        args.model_dir = self.model_id
        model_init.check_args(args)

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
