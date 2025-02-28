"""
Embeddings package for different model types (HuggingFace, ExLlamaV2, SentenceTransformer)
"""

from .base import Embedder
from .huggingface import HuggingfaceEmbedder, get_hf_embedding, compute_hf_embeddings_ddp
from .exllama import ExLlamaV2Embedder, ExLlamaV2EmbeddingGenerator
from .sent_trans import SentenceTransformerEmbedder, compute_st_embeddings_ddp
from .aggregation import select_layer_indices, aggregate_layers
from .factory import EmbedderFactory

__all__ = [
    'Embedder',
    'HuggingfaceEmbedder',
    'ExLlamaV2Embedder',
    'SentenceTransformerEmbedder',
    'ExLlamaV2EmbeddingGenerator',
    'get_hf_embedding',
    'compute_hf_embeddings_ddp',
    'compute_st_embeddings_ddp',
    'select_layer_indices',
    'aggregate_layers',
    'EmbedderFactory',
]
