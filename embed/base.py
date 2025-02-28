"""
Base class for all embedders
"""

class Embedder:
    def __init__(self, cfg=None, **kwargs):
        if cfg is not None:
            self.cfg = cfg
            self.model_id = cfg.model_id
            
    def get_embedding(self, input_ids, attention_mask, **kwargs):
        pass
    
    def compute_embeddings(self, data, **kwargs):
        pass
