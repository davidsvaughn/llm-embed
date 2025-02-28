"""
Factory function for creating embedders
"""

import copy
from util import to_adict
from .huggingface import HuggingfaceEmbedder
from .exllama import ExLlamaV2Embedder
from .sent_trans import SentenceTransformerEmbedder

def EmbedderFactory(cfg=None, model_id=None, model_type='hf'):
    """
    Factory function for creating embedders based on model type
    
    Args:
        cfg: Configuration object or dictionary
        model_id: Model identifier (overrides cfg.model_id if both are provided)
        model_type: Model type ('hf', 'st', 'exl2')
        
    Returns:
        Appropriate embedder instance based on model type
    """
    if model_type == 'st':
        return SentenceTransformerEmbedder(model_id)
    
    if cfg is None:
        cfg = to_adict({'model_id': model_id})
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