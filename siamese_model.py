import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Union, Literal, Optional
from transformers import PreTrainedTokenizerBase
from embedder.huggingface import HuggingfaceEmbedder

class AngularLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(AngularLoss, self).__init__()
        self.margin = margin

    def forward(self, emb1, emb2, labels):
        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)
        cos_theta = F.cosine_similarity(emb1, emb2)
        theta = F.acos(cos_theta)
        loss = torch.mean(labels * theta + (1 - labels) * torch.clamp(self.margin - theta, min=0.0))
        logits = torch.stack([-cos_theta, cos_theta], dim=1)
        return {"logits": logits, "loss": loss}

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss with flexibility for both cosine and euclidean distances.
    Args:
        margin: float, the margin for negative pairs (default: 1.0)
        metric: str, either 'cosine' or 'euclidean' (default: 'cosine')
        reduction: str, 'mean' or 'sum' (default: 'mean')
    """
    def __init__(self, metric='cosine', temperature=1.0, margin=None, margin_mult=0, use_diff_norm=False, **kwargs):
        super().__init__()
        self.metric = metric
        self.temperature = temperature
        self.reduction = 'mean'
        self.use_diff_norm = use_diff_norm
        self.margin_mult = margin_mult
        if margin is None:
            self.margin = 1.0 if metric == 'cosine' else 2.0
        else:
            assert margin > 0, "margin must be positive"
            self.margin = margin

    def forward(self, emb1, emb2, labels):

        # https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html
        if self.metric == 'cosine':
            cos = F.cosine_similarity(emb1, emb2)
            logits = torch.stack([-cos, cos], dim=1)
            d = 1-cos # range == [0..2]
            
            # labels == diff_norm
            margin = 1 + labels 
            losses = (labels==0).long()*d  + (1 - (labels==0).long()) * F.relu(margin - d)

        else:  # euclidean
            d = F.pairwise_distance(emb1, emb2, p=2)
            logits = torch.stack([d-1, 1-d], dim=1)
            
            if self.use_diff_norm:
                margin = self.margin + labels * self.margin_mult
            else:
                margin = self.margin + (labels-1) * self.margin_mult
            
            losses = (labels==0).long() * d.pow(2) + (1 - (labels==0).long()) * F.relu(margin - d).pow(2)

        # Logits only used for binary classification (in validation metrics)
        output_dict = {"logits": logits, "loss": losses.mean()}

        return output_dict


class DataCollatorForSiameseNetwork:
    def __init__(self, 
                tokenizer: PreTrainedTokenizerBase,
                padding: Union[bool, str] = True,
                max_length: int = None,
                pad_to_multiple_of: int = None,
                return_tensors: str = "pt",
                padding_side: str = "left",
                use_diff_norm: bool = False):
        
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
        self.padding_side = padding_side
        self.use_diff_norm = use_diff_norm
        # Store original padding side
        self.original_padding_side = self.tokenizer.padding_side
    
    def __call__(self, features: List[Dict[str, List[int]]], debug=False) -> Dict[str, torch.Tensor]:
        # Set padding side to what we want
        self.tokenizer.padding_side = self.padding_side

        # Separate the paired sequences
        batch_1 = [{
            "input_ids": feature["input_ids_1"],
            "attention_mask": feature["attention_mask_1"],
        } for feature in features]
        
        batch_2 = [{
            "input_ids": feature["input_ids_2"],
            "attention_mask": feature["attention_mask_2"],
        } for feature in features]

        # Pad sequences in each batch separately
        batch_1_padded = self.tokenizer.pad(
            batch_1,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors
        )
        
        batch_2_padded = self.tokenizer.pad(
            batch_2,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors
        )

        # Combine into final batch
        batch = {
            "input_ids_1": batch_1_padded["input_ids"],
            "attention_mask_1": batch_1_padded["attention_mask"],
            "input_ids_2": batch_2_padded["input_ids"],
            "attention_mask_2": batch_2_padded["attention_mask"],
        }

        # set 'labels' to either 'diff_norm' or 'diff_abs'
        label_name = "diff_norm" if self.use_diff_norm else "diff_abs"
        batch["labels"] = torch.tensor([f[label_name] for f in features], dtype=torch.float32)
            
        # Restore original padding side
        self.tokenizer.padding_side = self.original_padding_side

        return batch


class SiameseCausalLM(nn.Module):
    def __init__(self, model,
                similarity_type: Literal["cosine", "euclidean"] = "cosine",
                pooling_mode: Literal["mean", "lasttoken"] = "mean",
                padding_side: Literal["left", "right"] = "left",
                projection_dim: float = 0.0,
                lm_loss_weight: float = 0.0,
                hidden_layer: int = -1,
                use_diff_norm: bool = False,
                **kwargs
        ):
        super().__init__()

        self.config = model.config
        self.pooling_mode = pooling_mode
        self.padding_side = padding_side
        self.lm_loss_weight = lm_loss_weight
        self.hidden_layer = hidden_layer
        self.use_diff_norm = use_diff_norm
        self.projection = None
        
        self.encoder = model
        self.embedder = HuggingfaceEmbedder(model=model)
        
        if similarity_type in ["cosine", "euclidean"]:
            self.loss_function = ContrastiveLoss(metric=similarity_type, use_diff_norm=use_diff_norm, **kwargs)
        else:
            self.loss_function = AngularLoss()
        
    def get_embedding(self, input_ids, attention_mask, last_token_offset=0, **kwargs):
        output, lm_loss = self.embedder.get_embedding(input_ids,
                                                      attention_mask,
                                                      labels=True,
                                                      pooling_mode=self.pooling_mode,
                                                      padding_side=self.padding_side,
                                                      hidden_layer=self.hidden_layer,
                                                      last_token_offset=last_token_offset,
                                                      **kwargs)
        return output, lm_loss
    
    def forward(
        self,
        input_ids_1: torch.Tensor,
        attention_mask_1: torch.Tensor,
        input_ids_2: torch.Tensor,
        attention_mask_2: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        
        # Get embeddings for both sequences
        emb1, lm_loss_1 = self.get_embedding(input_ids_1, attention_mask_1)
        emb2, lm_loss_2 = self.get_embedding(input_ids_2, attention_mask_2)

        # Compute similarity
        output_dict = self.loss_function(emb1, emb2, labels)

        if self.lm_loss_weight > 0:
            pair_loss = output_dict["loss"]
            lm_loss = (lm_loss_1 + lm_loss_2)/2  # average lm loss
            total_loss = pair_loss + self.lm_loss_weight * lm_loss

            output_dict["loss"] = total_loss
            output_dict["pair_loss"] = pair_loss
            output_dict["lm_loss"] = lm_loss

        return output_dict


def tokenize_batch(batch, tokenizer):
    # tokenize and encode txt1
    tokenized_text = tokenizer(batch["txt1"],
                               return_offsets_mapping=True,
                               add_special_tokens=False)
    data = {f'{k}_1': tokenized_text[k] for k in tokenized_text.keys()}
    
    # tokenize and encode txt2
    tokenized_text = tokenizer(batch["txt2"],
                               return_offsets_mapping=True,
                               add_special_tokens=False)
    data = {**data, **{f'{k}_2': tokenized_text[k] for k in tokenized_text.keys()}}
    
    # add label
    try:
        data['diff_abs'] = batch['diff_abs']
    except:
        data['diff_abs'] = batch['diff']
    
    try:
        data['diff_norm'] = batch['diff_norm']
    except:
        pass
    
    # remove offset_mapping
    del data['offset_mapping_1']
    del data['offset_mapping_2']

    return data


def tokenize_dataset(dataset, tokenizer, args):
    """Tokenize and prepare a dataset for Siamese network training"""
    from functools import partial
    import numpy as np
    
    # tokenize dataset
    print(f"\nTokenizing dataset...")
    tokenized_dataset = dataset.map(partial(tokenize_batch, tokenizer=tokenizer),
                                   batched=True, 
                                   remove_columns=list(dataset.features))

    # filter out rows of tokenized_dataset that are too long
    print(f"\nFiltering out responses that are too long...(max_seq_length: {args.max_seq_length})")
    tokenized_dataset = tokenized_dataset.filter(
        lambda x: len(x["input_ids_1"]) <= args.max_seq_length and len(x["input_ids_2"]) <= args.max_seq_length
    )

    # make histogram of input lengths
    input_lengths = np.concatenate([np.array([len(x) for x in tokenized_dataset[f"input_ids_{i}"]]) for i in range(1,3)])
    
    # Draw simple ascii histogram
    def ascii_hist(x, nb=20, maxlen=100, nb0=None):
        w = np.ptp(x)/nb  # get bin width from num bins
        min_val, max_val = np.min(x), np.max(x)     # get min/max vals
        bins = np.arange(min_val, max_val + 1, w)   # create bins
        hist, _ = np.histogram(x, bins)     # get histogram sizes
        scale = maxlen/hist.max()
        # get index of last nonzero value of hist
        h = (scale*hist).astype(int)
        last_nz = np.nonzero(h)[0][-1]
        if nb0 is None:
            if last_nz < 0.8*nb:
                nb = int(nb*nb/(last_nz+1))
                ascii_hist(x, nb=nb, maxlen=maxlen, nb0=nb)
                return
        # draw histogram
        for i in range(len(hist)):
            print(f"{bins[i]:0.0f} - {bins[i]+w:0.0f}\t{'#' * int(scale*hist[i])}")
            if i == last_nz: break
    
    ascii_hist(input_lengths, nb=20, maxlen=100)
    
    # print # samples
    print(f"\nNumber of samples: {len(tokenized_dataset)}")
    return tokenized_dataset
