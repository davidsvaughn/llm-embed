import os, sys
import re
import copy
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel, PeftConfig
import torch
from jinja2 import Template
import hashlib
from tqdm import tqdm
from typing import Any, Dict, List, Union
from utils import to_adict
from ddp_utils import is_main, printmain

def get_tokenizer_uid(tokenizer):
    if not hasattr(tokenizer, 'tok_uid'):
        vocab_dict = tokenizer.get_vocab()
        vocab_str = str(sorted(vocab_dict.items()))  # Sort for consistency
        tokenizer.tok_uid = hashlib.md5(vocab_str.encode()).hexdigest()
    return tokenizer.tok_uid

def get_chat_template_uid(tokenizer):
    if not hasattr(tokenizer, 'chat_uid'):
        tokenizer.chat_uid = hashlib.md5(tokenizer.chat_template.encode()).hexdigest()
    return tokenizer.chat_uid

# load checkpoint tokenizer
def load_checkpoint_tokenizer(checkpoint_dir, padding_side='left'):
    config = PeftConfig.from_pretrained(checkpoint_dir)
    printmain(f"Loading tokenizer for: {config.base_model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer

def load_hf_tokenizer(tokenizer_id, padding_side='left'):
    printmain(f"Loading tokenizer for: {tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer

# version of load_tokenizer that only takes cfg as argument
def load_tokenizer(cfg=None, **kwargs):
    if cfg is not None:
        cfg = to_adict(copy.deepcopy(cfg))
        cfg.update(kwargs)
    else:
        cfg = to_adict(kwargs)
    return load_hf_tokenizer(cfg.get('tokenizer_id', cfg.model_id), cfg.get('padding_side', 'left'))

# load a tokenizer with chat_template
def load_chat_tokenizer(model_id, depth=0, verbose=1):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
            if verbose>0:
                print(f"Loaded Chat Tokenizer: {model_id}")
            if verbose>1:
                print(f"Chat Template:\n{tokenizer.chat_template}")
            return tokenizer
    except Exception as e:
        if depth > 0:
            print(f"ERROR in get_chat_tokenizer: {e}")
            traceback.print_exc()
    # recurse with base_model_name_or_path
    config = AutoConfig.from_pretrained(model_id)
    return load_chat_tokenizer(config.base_model_name_or_path, depth+1, verbose+1)


def load_checkpoint_model(checkpoint_dir, tokenizer=True, padding_side='left'):
    device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if torch.cuda.is_available() else "cpu"
    config = PeftConfig.from_pretrained(checkpoint_dir)

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation='flash_attention_2',
        device_map={"": device},  # Map to specific GPU instead of "auto"
    )
    model = PeftModel.from_pretrained(model, checkpoint_dir)
    model = model.merge_and_unload()
    model.eval()
    
    if tokenizer:
        model.tokenizer = load_checkpoint_tokenizer(checkpoint_dir, padding_side=padding_side)
    return model

def load_hf_model(model_id, tokenizer=True, padding_side='left'):
    device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if torch.cuda.is_available() else "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
        device_map={"": device},  # Map to specific GPU instead of "auto"
    )
    model.eval()
    
    if tokenizer:
        model.tokenizer = load_hf_tokenizer(model_id, padding_side=padding_side)
    return model

def load_model(cfg=None, model_id=None, tokenizer=True):
    if cfg is not None and model_id is None:
        model_id = cfg.get('model_id', None)
        
    if model_id is None:
        raise ValueError("model_id must be provided")
        
    printmain(f"Loading model: {model_id}")
    
    # get tokenizer padding side
    padding_side = 'left' if cfg is None else cfg.get('padding_side', 'left')
    
    # check if model_id is checkpoint directory
    if 'checkpoint' in model_id:
        model = load_checkpoint_model(model_id, tokenizer=tokenizer, padding_side=padding_side)
    else:
        model = load_hf_model(model_id, tokenizer=tokenizer, padding_side=padding_side)
    model.model_id = model_id
    return model
    
#--------------------------------------------------------------------------------

def apply_chat_template(tokenizer, payload, add_generation_prompt=True):
    chat_text = tokenizer.apply_chat_template(payload.messages, 
                                              tokenize=False,
                                              add_special_tokens=False,
                                              add_generation_prompt=add_generation_prompt and payload.messages[-1]['role'] == 'user',
                                              )
    if 'target' in payload:
        # truncate chat_text to just before rightmost occurrence of target value
        chat_text = chat_text.rsplit(str(payload.target[-1]), 1)[0]
        
    return chat_text


def apply_chat_template_batched(tokenizer, records, add_generation_prompt=True):
    return [apply_chat_template(tokenizer, rec.payload, add_generation_prompt=add_generation_prompt) for rec in records]


#--------------------------------------------------------------------------------
def tokenize_data_batched(data: Union[List[str], Dict[str, List[str]]],
                          tokenizer: Any,
                          batch_size: int = 8,
                          max_length: int = None,
                          **kwargs):
    """
    Batch tokenize data with progress bar.
    
    Args:
        data: List of strings or Dict of lists of strings
        tokenizer: HuggingFace tokenizer
        batch_size: Number of items to process at once
        max_length: Optional max sequence length
        
    Returns:
        None
        
    Side effects:
        Applies tokenizer's chat_template to each record.payload.messages
        Adds tokenized_text (input_ids, attention_mask) and batch_index to each record
    """
  
    # if data is a dict, flatten the nested structure into a single list of records
    records = [rec for item, recs in data.items() for rec in recs] if isinstance(data, dict) else data
        
    # if every record in records already contains tokenized_text, return
    if tokenizer is not None and all([hasattr(rec, 'tokenized_text') for rec in records]):
        return

    # Process in batches
    batch_index = 0
    
    disable=not is_main()
    disable = True

    for i in tqdm(range(0, len(records), batch_size), desc="Tokenizing", disable=not is_main()):
        batch = records[i:i + batch_size]
        
        if tokenizer is not None:
            # Prepare all texts for the batch
            batch_inputs = apply_chat_template_batched(tokenizer, batch)
            
            # Tokenize the batch at once
            tokenized = tokenizer(
                batch_inputs,
                add_special_tokens=False,
                padding=True,
                truncation=True if max_length else False,
                max_length=max_length,
                return_tensors="pt"
            )
        
        # Store tokenized results back in the records
        for idx, rec in enumerate(batch):
            '''
            --> Each record has 2 fields added:
                - tokenized_text: { 'input_ids': tensor, 'attention_mask': tensor }
                - batch_index: int [ used for batch inference, not training...]
            '''
            if tokenizer is not None:
                # Extract this record's tokens from the batch
                rec.tokenized_text = {
                    'input_ids': tokenized['input_ids'][idx],
                    'attention_mask': tokenized['attention_mask'][idx]
                }
                
                # store pre-tokenized text, after chat template has been applied
                rec.chat_text = batch_inputs[idx]
            
            # each record stores its own batch index - used for batch inference
            rec.batch_index = batch_index
            
        batch_index += 1


#--------------------------------------------------------------------------------

class PromptBuilder:
    def __init__(self,
                 prompt_dir,
                 **kwargs
    ):
        self.kwargs = kwargs
        self.tokenizer = kwargs.get('tokenizer', None)
        self.null_target = 'NULL'

        # compile regex to match 10 or more consecutive whitespace characters
        self.whitespace_pattern = re.compile(r'\s{10,}')
        
        # load prompt templates...
        self.user_template, self.asst_template = None, None
        try:
            user_template_file = os.path.join(prompt_dir, 'user.j2')
            if os.path.exists(user_template_file):
                with open(user_template_file, 'r') as f:
                    self.user_template = Template(f.read())
                    
            asst_template_file = os.path.join(prompt_dir, 'asst.j2')
            if os.path.exists(asst_template_file):
                with open(asst_template_file, 'r') as f:
                    temp = f.read()
                self.asst_template = Template(temp)
                
                # set target_name = last variable in asst_template
                matches = re.findall(r'\{\{\s*(.*?)\s*\}\}', temp)
                if len(matches) > 0:
                    self.target_name = matches[-1]
            
        except Exception as e:
            print(f'ERROR: {e}')
            traceback.print_exc()
            sys.exit()

    def reduce_whitespace(self, text, replacement='\n\n'):
        return self.whitespace_pattern.sub(replacement, text.strip())

    # Receives a string and returns the string with Unicode escape sequences replaced by their corresponding characters.
    def fix_unicode_escapes(self, text):
        try:
            if ('\\u' in text) or ('\\U' in text):
                return text.encode('latin1', errors='ignore').decode('unicode-escape')
            return text
        except:
            pass
        return text
    
    def apply(self, template, **kwargs):
        try:
            # merge kwargs without altering the originals (2nd kwargs overrides 1st)...
            text = template.render(**{**self.kwargs, **kwargs}).strip()
            text = self.reduce_whitespace(text)
            # text = self.fix_unicode_escapes(text) # no longer needed...?
            return text
        except Exception as e:
            print(f'ERROR: {e}')
            print(f'args: {kwargs}')
            traceback.print_exc()
            sys.exit()
            
    def get_messages(self, **kwargs):
        user_text = self.apply(self.user_template, **kwargs)
        messages = [{"role": "user", "content": user_text}]
        
        asst_text = self.apply(self.asst_template, **kwargs)
        messages.append({"role": "assistant", "content": asst_text})
            
        return messages
    
    def get_payload(self, **kwargs):
        
        # check if self.target_name is in kwargs
        if self.target_name in kwargs:
            # if so, use it as target_value
            target_value = kwargs[self.target_name]
        if self.target_name not in kwargs:
            # otherwise, use self.null_target as target_value
            target_value = self.null_target
            kwargs = { **kwargs, self.target_name: target_value }
            
        messages = self.get_messages(**kwargs)
        
        return to_adict({
            "messages": messages,
            "target": (self.target_name, target_value),
        })
    
    # def render(self, **kwargs):
    #     tokenizer = kwargs.get('tokenizer', self.tokenizer)
    #     assert tokenizer is not None, "tokenizer must be provided to apply chat template"
        
    #     #--------------------------------------------------------------
    #     # TRUNCATION LOGIC: useful for embedding models...
    #     # set truncate variable to unique string
    #     if self.truncate_to is not None:
    #         # store original value of truncate_to, if it exists
    #         orig_str = str(kwargs.get(self.truncate_to, None))
    #         # replace truncate_to with unique string
    #         kwargs = { **kwargs, self.truncate_to: self.uniq_str }
    #     #--------------------------------------------------------------
        
    #     # get messages
    #     messages = self.get_messages(**kwargs)
    #     # apply chat template
    #     chat_text = tokenizer.apply_chat_template(messages, 
    #                                               tokenize=False,
    #                                               add_special_tokens=False,
    #                                               add_generation_prompt= messages[-1]['role'] == 'user',
    #                                               )
        
    #     #--------------------------------------------------------------
    #     # truncate chat_text to just before rightmost occurrence of unique string
    #     if self.truncate_to is not None:
    #         chat_text = chat_text.rsplit(self.uniq_str, 1)[0]
    #         # restore original value of truncate_to, if it exists
    #         if orig_str is not None:
    #             chat_text = chat_text.replace(self.uniq_str, orig_str)
    #     #--------------------------------------------------------------
        
    #     return chat_text
    
    # def render_chat_text(self, record, tokenizer=None):
        
    #     if tokenizer is None:
    #         tokenizer = self.tokenizer
    #     assert tokenizer is not None, "tokenizer must be provided to apply chat template"
        
    #     chid = get_chat_template_uid(tokenizer)
    #     if 'chat_text' not in record:
    #         record['chat_text'] = {}
    #     else:
    #         if chid in record['chat_text']:
    #             return record['chat_text'][chid]
            
    #     chat_text = self.render(**{**record, 'tokenizer': tokenizer})
    #     record['chat_text'][chid] = chat_text
    #     return chat_text
        
        
