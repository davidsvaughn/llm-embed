#--------------------------------------------------------------------------------------------------
# 1. GPU Setup and Utilities - housekeeping related to single/multi-GPU training

import os,sys
from ddp_utils import is_main, main, printmain, get_num_processes
import torch
import torch.distributed as dist

#--------------------------------------------------------------------------------------------------
# 2. Argument Parsing
# - cmd line arguments take precedence over these defaults
# - can be run with no cmd line arguments, or with any of the following
# - all arguments are logged to wandb

from transformers import TrainingArguments, HfArgumentParser
from dataclasses import dataclass, asdict, field
from typing import List, Optional, cast
from jinja2 import Template
from util import clear_cuda_tensors, to_adict

cur_dir = os.path.dirname(os.path.abspath(__file__))

# PACKING: https://huggingface.co/blog/sirluk/llm-sequence-packing

@dataclass
class ScriptArguments:

    model_id:       str             = field(default="meta-llama/Llama-3.2-3B-Instruct", metadata={"help": "The HuggingFace model id"})
    # model_id:       str             = field(default="microsoft/Phi-3-mini-128k-instruct", metadata={"help": "The HuggingFace model id"})
    # model_id:       str             = field(default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", metadata={"help": "The HuggingFace model id"})
    
    # dataset_id:     List[str]       = field(default_factory=lambda: ["davidsvaughn/math_pairs_1159"], metadata={"help": "The HuggingFace dataset id"}) # *****
    # dataset_id:     List[str]       = field(default_factory=lambda: ["davidsvaughn/math_pairs_460"], metadata={"help": "The HuggingFace dataset id"})
    # dataset_id:     List[str]       = field(default_factory=lambda: ["davidsvaughn/math_pairs_7186"], metadata={"help": "The HuggingFace dataset id"})
    dataset_id:     List[str]       = field(default_factory=lambda: ["davidsvaughn/math_pairs_1426"], metadata={"help": "The HuggingFace dataset id"})
    
    # dataset_id:     List[str]       = field(default_factory=lambda: ["davidsvaughn/math_pairs_279"], metadata={"help": "The HuggingFace dataset id"}) # *****
    # dataset_id:     List[str]       = field(default_factory=lambda: ["davidsvaughn/bw_pairs_722"], metadata={"help": "The HuggingFace dataset id"})
    # dataset_id:     List[str]       = field(default_factory=lambda: ["davidsvaughn/bw_fw-dev_pairs_834"], metadata={"help": "The HuggingFace dataset id"})
    

    # dataset_id:     List[str]       = field(default_factory=lambda: ["davidsvaughn/mixed_pairs_910"], metadata={"help": "The HuggingFace dataset id"})
    # dataset_id:     List[str]       = field(default_factory=lambda: ["davidsvaughn/mixed_pairs_475"], metadata={"help": "The HuggingFace dataset id"})
    # dataset_id:     List[str]       = field(default_factory=lambda: ["davidsvaughn/math_pairs_870"], metadata={"help": "The HuggingFace dataset id"})
    # dataset_id:     List[str]       = field(default_factory=lambda: ["davidsvaughn/mixed_pairs_962"], metadata={"help": "The HuggingFace dataset id"})
    
    # dataset_id:     List[str]       = field(default_factory=lambda: ["davidsvaughn/math_pairs_460"], metadata={"help": "The HuggingFace dataset id"})
    # dataset_id:     List[str]       = field(default_factory=lambda: ["davidsvaughn/math_pairs_263"], metadata={"help": "The HuggingFace dataset id"})
    
    
    prompt_template:str             = field(default="prompts/math/user.j2", metadata={"help": "The prompt template to use"})
    
    use_jinja2:     Optional[bool]  = field(default=True, metadata={"help": "Whether to use jinja2 templates"})
    data_dir:       Optional[str]   = field(default='~/data', metadata={"help": "The directory to store the data"})
    rand_seed:      Optional[int]   = field(default=1357, metadata={"help": "The random seed to use"})
    max_seq_length: Optional[int]   = field(default=2048, metadata={"help": "The maximum sequence length"})
    subsample_train:Optional[float] = field(default=1000000, metadata={"help": "The number of training samples to use"})
    subsample_eval: Optional[float] = field(default=20000, metadata={"help": "The number of evaluation samples to use"})
    max_samples:    Optional[int]   = field(default=1000000, metadata={"help": "The maximum number of samples to load"})
    lora_alpha:     Optional[int]   = field(default=32, metadata={"help": "The LoRA alpha parameter"})
    lora_r:         Optional[int]   = field(default=32, metadata={"help": "The LoRA r parameter"})
    lora_dropout:   Optional[float] = field(default=0.1, metadata={"help": "The LoRA dropout rate"})
    attn_implementation: Optional[str] = field(default="flash_attention_2", metadata={"help": "The attention implementation to use"})
    use_4bit:       Optional[bool]  = field(default=False, metadata={"help": "Whether to use 4-bit quantization"})
    use_double_quant:Optional[bool] = field(default=False, metadata={"help": "Whether to use double quantization"})
    shuffle:        Optional[bool]  = field(default=True, metadata={"help": "Whether to shuffle the training data"})
    deepspeed_template:      Optional[str]   = field(default="zero2_decay_template.j2", metadata={"help": "The deepspeed configuration file"})
    prompt_loss_weight: Optional[float] = field(default=1.0, metadata={"help": "The prompt loss weight"})

    #### Siamese Network Arguments ####
    padding_side:   Optional[str]   = field(default="left", metadata={"help": "The padding side to use"})       # left | right
    projection_dim: Optional[float] = field(default=0.0, metadata={"help": "The projection dimension to use"})

    hidden_layer:   Optional[int]   = field(default=-1, metadata={"help": "The hidden layer to use"})
    # use_diff:       Optional[bool]  = field(default=True, metadata={"help": "use score diffs in loss"})
    margin:         Optional[float] = field(default=2.0, metadata={"help": "The margin to use in contrastive loss"})
    use_xgb:        Optional[bool]  = field(default=True, metadata={"help": "Whether to run XGBoost"})
    
    bucket_size_multiplier: Optional[int] = field(default=-1, metadata={"help": "The bucket size multiplier to use"})
    max_batch_area: Optional[int] = field(default=-1, metadata={"help": "The bucket size multiplier to use"})
    
    #----------------
    similarity_type:Optional[str]   = field(default="euclidean", metadata={"help": "The similarity type to use"})  # cosine | euclidean | angular
    # similarity_type:Optional[str]   = field(default="cosine", metadata={"help": "The similarity type to use"})  # cosine | euclidean | angular
    
    use_diff_norm:  Optional[bool]  = field(default=False, metadata={"help": "use normalized score diffs in loss"})
    # use_diff_norm:  Optional[bool]  = field(default=True, metadata={"help": "use normalized score diffs in loss"})
    
    margin_mult:    Optional[float] = field(default=1.0, metadata={"help": "The margin multiplier to use in contrastive loss"})
    # margin_mult:    Optional[float] = field(default=2.0, metadata={"help": "The margin multiplier to use in contrastive loss"})
    
    lm_loss_weight: Optional[float] = field(default=0.001, metadata={"help": "The language model loss weight"})
    # lm_loss_weight: Optional[float] = field(default=0.0, metadata={"help": "The language model loss weight"})
    
    # pooling_strategy:Optional[str]  = field(default="mean", metadata={"help": "The pooling strategy to use"})   # mean | last
    pooling_strategy:Optional[str]  = field(default="last", metadata={"help": "The pooling strategy to use"})   # mean | last  
    # pooling_strategy:Optional[str]  = field(default="mean,last", metadata={"help": "The pooling strategy to use"})   # mean | last
    
    #----------------

# instantiate default training arguments
training_args = TrainingArguments(
    num_train_epochs                = 6,        # ** 6 ** number of training epochs
    
    per_device_train_batch_size     = 4,    # 4       # batch size per device during training
    per_device_eval_batch_size      = 4,    # 4       # batch size for evaluation
    gradient_accumulation_steps     = 8,   # 8        # number of steps before performing a backward/update pass
    
    # gradient_checkpointing          = True,     # use gradient checkpointing to save memory
    remove_unused_columns           = False,    # False makes custom fields (like 'completion_mask') available inside compute_loss function
    logging_strategy                = "steps", 
    eval_strategy                   = "steps",
    output_dir                      = "/home/azureuser/embed/output", # directory to save model checkpoints
    
    logging_steps                   = 10,       # log train set metrics
    
    # eval_steps                      = 100,       # log eval set metrics
    eval_steps                      = 50,
    
    save_steps                      = 50,
    # save_steps                      = 1000000,       # for debugging (high == don't save)
    
    bf16                            = True,     # use bfloat16 precision
    tf32                            = True,     # use tf32 precision
    max_grad_norm                   = 0.3,      # max gradient norm based on QLoRA paper
    warmup_ratio                    = 0.05,     # warmup ratio based on QLoRA paper
    weight_decay                    = 0.001,    # weight decay
    lr_scheduler_type               = "constant_with_warmup",   # use constant learning rate scheduler
    gradient_checkpointing_kwargs   = {"use_reentrant": True},
    report_to                       = "wandb",  # report metrics to wandb

    # only use deepspeed for multi-GPU training: torchrun --nproc_per_node 4 run_plw.py
    learning_rate                   = 1e-4,     # learning rate, based on QLoRA paper
    deepspeed = cur_dir+"/zero2_decay_lr1.json" if 'LOCAL_RANK' in os.environ else None,
    
    # learning_rate                   = 2e-4,     # learning rate, based on QLoRA paper
    # deepspeed = cur_dir+"/zero2_decay_lr2.json" if 'LOCAL_RANK' in os.environ else None,
    # deepspeed = cur_dir+"/zero3_decay.json" if 'LOCAL_RANK' in os.environ else None,
)

# if '--help' is in command line arguments, print usage and exit
if '--help' in sys.argv:
    HfArgumentParser((ScriptArguments, TrainingArguments)).parse_args_into_dataclasses()

# parse into ScriptArguments and remaining args
parser = HfArgumentParser(ScriptArguments)
script_args, cmd_train_args = parser.parse_known_args()

# evaluate string literal to intrinsic type
def totype(s):
    try: return eval(s)
    except: return s

# overwrite default training_args with command line training arguments
for k,v in zip(cmd_train_args[::2], cmd_train_args[1::2]):
    setattr(training_args, k.lstrip('-'), totype(v))
    
for k in ['prompt_template']:
    if getattr(script_args, k).endswith('.j2'):
        script_args.use_jinja2 = True
        with open(cur_dir+'/'+getattr(script_args, k), 'r') as f:
            setattr(script_args, k, f.read())

if is_main():
    if 'deepspeed_template' in script_args and script_args.deepspeed_template and training_args.deepspeed:
        with open(cur_dir+'/'+getattr(script_args, 'deepspeed_template'), 'r') as f:
            ds_template = Template(f.read())
        lr = training_args.learning_rate
        ds_config = ds_template.render(lr=lr)
        with open(training_args.deepspeed, 'w') as f:
            f.write(ds_config)

script_args = ScriptArguments(**vars(script_args)) # cast namespace to dataclass

#---------------------------------------------------------------------------------------

EVAL_MULT = 1
USE_SCORE = False

#---------------------------------------------------------------------------------------
# for external validation loop
from data_utils import get_config, load_items

root_data_dir = '/home/azureuser/embed/data'
root_prompt_dir = '/mnt/llm-train/embed/simple/prompts'

# validation items
math_items = [123362, 33082, 13272, 27218, 29632, 31600, 52414, 78382]
bw_items = [33234, 63166, 96340, 125665] # 96182
fw_items = [56356, 57236, 61342, 108694] # 107550

# debugging....
math_items = [123362, 29632]
bw_items = [33234]
fw_items = [57236]


test_items = {}
if 'math' in script_args.dataset_id[0]:
    test_items['math'] = math_items
if 'bw' in script_args.dataset_id[0]:
    test_items['bw'] = bw_items
if 'fw' in script_args.dataset_id[0]:
    test_items['fw'] = fw_items

# if 'mixed_pairs' in script_args.dataset_id[0]:
#     test_items['bw'] = bw_items
#     test_items['fw-con'] = fw_items
    # training_args.gradient_accumulation_steps = 16
    # training_args.per_device_train_batch_size = training_args.per_device_eval_batch_size = 2
# elif 'math_pairs' in script_args.dataset_id[0]: # just math
#     training_args.gradient_accumulation_steps = 8
#     training_args.per_device_train_batch_size = training_args.per_device_eval_batch_size = 4
    

#---------------------------------------------------------------------------------------
DEBUG = 0
# script_args.subsample_eval = 1000

if DEBUG:
    training_args.logging_steps = 10

    if DEBUG == 1:
        script_args.subsample_eval = 5000
        training_args.eval_steps = 100
        training_args.save_steps = training_args.eval_steps

    elif DEBUG == 2:
        training_args.gradient_accumulation_steps = 2
        script_args.max_samples = 10000
        script_args.subsample_eval = 1000
        training_args.eval_steps = 50
        training_args.save_steps = training_args.eval_steps
    
    elif DEBUG == 3:
        training_args.gradient_accumulation_steps = 2
        script_args.max_samples = 5000
        script_args.subsample_train = 10000
        script_args.subsample_eval = 500
        # training_args.eval_steps = 20
        
        # training_args.eval_steps = 300
        # training_args.save_steps = training_args.eval_steps
        EVAL_MULT = 1
        
        # test_items = {'math': [13272, 27218, 123362],
        #               'bw': [33234, 96182, 63166],
        #               'fw-con': [56356, 57236, 61342]}
                      
        test_items = {'math': [123362],
                    #   'bw': [33234],
                    #   'fw-con': [56356],
                      }
    elif DEBUG == 4:
        training_args.gradient_accumulation_steps = 2
        script_args.max_samples = 5000
        script_args.subsample_train = 500
        script_args.subsample_eval = 100
        training_args.eval_steps = 300
        EVAL_MULT = 1           
        test_items = {'math': [123362],
                      }
                      


#---------------------------------------------------------------------------------------

# Print all arguments for verification
if is_main():
    print('-'*100)
    print("\n-------- Training Arguments ---------")
    for key, value in asdict(training_args).items():
        print(f"{key}: {value}")

    print("\n-------- Script Arguments -----------")
    for key, value in asdict(script_args).items():
        print(f"{key}: {value}")
    print('-'*100)

    # check if output directory exists and is non-empty, if not create it, if so, increment the directory name
    i, prefix = 0, training_args.output_dir
    while os.path.exists(training_args.output_dir) and os.listdir(training_args.output_dir):
        i+=1
        training_args.output_dir = f"{prefix}{i}"
        # if i>2: break #dsv
    os.makedirs(training_args.output_dir, exist_ok=True)
    print(f"OUTPUT DIRECTORY: {training_args.output_dir}")

#--------------------------------------------------------------------------------------------------
# 3. Logging setup and random seed initialization

import wandb, json

# this allows logging of all arguments to wandb, without throwing JSON serialization errors
def make_json_serializable(d):
    def is_json_serializable(value):
        try:
            json.dumps(value)
            return True
        except (TypeError, OverflowError):
            return False
    return {k: v if is_json_serializable(v) else str(v) for k, v in d.items()}

# initialize wandb and log all arguments
if is_main():
    wandb.init(project= f"Embedding-LLM--{script_args.dataset_id[0].replace('/','_')}")
    wandb.config.update(make_json_serializable(asdict(training_args)))
    wandb.config.update(make_json_serializable(asdict(script_args)))
    wandb_run_name = wandb.run.name
    wandb_run_id = wandb.run.id
    print(f"Wandb run name: {wandb_run_name}, id: {wandb_run_id}")
    print()

# set random seed for reproducibility
import random
import numpy as np

# set seed
torch.manual_seed(script_args.rand_seed)
np.random.seed(script_args.rand_seed)
random.seed(script_args.rand_seed)

#--------------------------------------------------------------------------------------------------
# 5. Dataset Loading and Preprocessing

from model_utils import apply_chat_template
from datasets import DatasetDict, load_dataset, Sequence, Value, Dataset
from transformers import AutoTokenizer
from functools import partial
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

# draw simple ascii histogram
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

#--------------------------------------------------------------------------------------------------
# Function to tokenize and encode a batch of samples, and creates prompt/completion masks.
# Note: This function assumes a single user/asst chat exchange (i.e. prompt + completion).
# For arbitrary length user/asst chat dialogues, a more general user-masking solution was proposed 
# here: https://github.com/huggingface/trl/issues/632#issuecomment-1972630547

def tokenize_batch(batch, tokenizer):
    # tokenize and encode txt1
    tokenized_text = tokenizer( batch["txt1"],
                                return_offsets_mapping=True,
                                add_special_tokens=False)
    data = { f'{k}_1' : tokenized_text[k] for k in tokenized_text.keys()}
    
    
    # tokenize and encode txt2
    tokenized_text = tokenizer( batch["txt2"],
                                return_offsets_mapping=True,
                                add_special_tokens=False)
    data = {**data, **{ f'{k}_2' : tokenized_text[k] for k in tokenized_text.keys()}}
    
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

#--------------------------------------------------------------------------------------------------

# tokenize dataset
def tokenize_dataset(dataset, tokenizer, args):
    # tokenize dataset
    print(f"\nTokenizing dataset...")
    tokenized_dataset = dataset.map(partial(tokenize_batch, tokenizer=tokenizer),
                                    batched=True, 
                                    remove_columns=list(dataset.features))

    # filter out rows of tokenized_dataset that are too long
    print(f"\nFiltering out responses that are too long...(max_seq_length: {args.max_seq_length})")
    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids_1"]) <= args.max_seq_length and len(x["input_ids_2"]) <= args.max_seq_length)

    # make histogram of input lengths
    input_lengths = np.concatenate([np.array([len(x) for x in tokenized_dataset[f"input_ids_{i}"]]) for i in range(1,3)])
    ascii_hist(input_lengths, nb=20, maxlen=100)
    
    # print # samples
    print(f"\nNumber of samples: {len(tokenized_dataset)}")
    return tokenized_dataset

#--------------------------------------------------------------------------------------------------

def check_dataset_exists(dataset_id):
    api = HfApi()
    try:
        # Try to fetch the dataset info
        api.dataset_info(dataset_id)
        return True
    except RepositoryNotFoundError:
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
    
@main
def prepare_dataset(tokenizer, args):
    print(f"\nBuilding dataset...")

    # Load dataset from HuggingFace hub
    dataset = load_dataset(*args.dataset_id)

    # shuffle dataset
    dataset = dataset.shuffle(seed=args.rand_seed)
    
    # print splits and number of samples
    dataset_keys = list(dataset.keys())
    for k in dataset_keys:
        print(f"Number of {k} samples: {len(dataset[k])}")

        # if number of samples is more than max_samples, randomly select max_samples
        if len(dataset[k]) > args.max_samples:
            dataset[k] = dataset[k].shuffle(seed=args.rand_seed).select(range(args.max_samples))
            print(f"Randomly selected {args.max_samples} samples from {k} split")

    # if there is no validation dataset, split the training dataset to create a validation set
    if 'validation' not in dataset_keys:
        dataset = dataset['train'].train_test_split(test_size=0.1, seed=args.rand_seed)
        dataset['validation'] = dataset.pop('test') # rename 'test' to 'validation'
        dataset_keys = list(dataset.keys())
        
    #----------------------------------------------------------------------------------------------
    #dsv - old method
    def format_prompt(inputs, template_text, tokenizer):
        template = Template(template_text)
        # if hasattr(format_prompt, 'template'):
        #     template = format_prompt.template
        # else:
        #     format_prompt.template = template = Template(template_text)
        user_text = template.render(**inputs).strip()
        msgs = [{"role": "user", "content": user_text}]
        txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return txt + "{'score':'"
    #----------------------------------------------------------------------------------------------
    
    # apply instruction template and chat template to each sample
    def format_sample(sample):
        try:
            payload1 = to_adict(json.loads(sample['payload1']))
            payload2 = to_adict(json.loads(sample['payload2']))
            sample["txt1"] = apply_chat_template(tokenizer, payload1)
            sample["txt2"] = apply_chat_template(tokenizer, payload2)
        except:
            #dsv - old method
            sample['txt1'] = format_prompt({**sample, 'text': sample['text1'] }, args.prompt_template, tokenizer)
            sample['txt2'] = format_prompt({**sample, 'text': sample['text2'] }, args.prompt_template, tokenizer)

        return sample

    # keep these fields
    remove_columns = list(dataset[k].features)
    for name in ['diff', 'diff_norm', 'diff_abs']:
        try: remove_columns.remove(name) # i.e. keep these (remove from remove_columns)
        except: pass
    
    # format each sample
    print(f"\nFormatting samples (applying chat template...)")
    dataset = DatasetDict( { k : dataset[k].map(format_sample, 
                                                remove_columns=remove_columns) for k in dataset_keys })

    # tokenize
    dataset = DatasetDict({ k : tokenize_dataset(dataset[k], tokenizer, args) for k in dataset_keys })

    # print sizes of each dataset split
    for k in dataset_keys:
        print(f"Total count of {k} tokenized sequences: {len(dataset[k])}")
        
    return dataset

# NEW version - save/load dataset to/from HF hub
def load_or_prepare_dataset(tokenizer, args):
    dataset_tag = args.dataset_id[0]
    dataset_tag = dataset_tag.replace("davidsvaughn/", "").replace("/", "_")
    model_tag = args.model_id.split('/')[-1]
    data_tag = f'{dataset_tag}_{model_tag}_tokenized'
    hf_dataset_id = f"davidsvaughn/{data_tag}"
    
    # check if dataset is already on HF hub
    if check_dataset_exists(hf_dataset_id):
        return hf_dataset_id
    
    # prepare tokenized dataset
    dataset = prepare_dataset(tokenizer, args)
    
    # push to HF hub
    if dataset is not None:
        print(f"\nPushing tokenized dataset to HuggingFace hub: {hf_dataset_id}")
        dataset.push_to_hub(hf_dataset_id, private=True)
        if get_num_processes() == 1:
            sys.exit()
        else:
            # barrier for multi-GPU training
            dist.barrier()
    return hf_dataset_id

# load tokenizer for model
tokenizer = AutoTokenizer.from_pretrained(script_args.model_id)
tokenizer.padding_side = script_args.padding_side
tokenizer.pad_token = tokenizer.eos_token
# tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

# load or prepare (+ load) dataset
hf_dataset_id = load_or_prepare_dataset(tokenizer, script_args)
print(f"\nLoading tokenized dataset from HuggingFace hub: {hf_dataset_id}")
llm_dataset = load_dataset(hf_dataset_id)

# random sample from dataset:
#     0<n<1: return n fraction of samples
#     n>1:   return n samples
#     else:  return all samples
def random_subset(dataset, n):
    m = len(dataset)
    if n<=0 or n>=m: return dataset
    n = int(m*n) if n<1 else int(n)
    idx = np.random.permutation(m)
    return dataset.select(idx[:n])

# subsample train & validation sets for faster training
for k,n in zip(['train', 'validation'], [script_args.subsample_train, script_args.subsample_eval]):
    m = len(llm_dataset[k])
    llm_dataset[k] = random_subset(llm_dataset[k], n)
    printmain(f"Using {len(llm_dataset[k])} of {m} {k} sequences")

#--------------------------------------------------------------------------------------------------
# 6. Model Initialization

from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=script_args.use_double_quant,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
) if script_args.use_4bit else None

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation=script_args.attn_implementation,
    # attn_implementation='eager',
    use_cache=not training_args.gradient_checkpointing,
    quantization_config=bnb_config,
    trust_remote_code=True,
)

#---------------------------------------------------------------------------------------
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase, AutoModelForCausalLM, PreTrainedTokenizerBase, PreTrainedModel
from typing import Dict, List, Union, Literal
from embed_utils import HuggingfaceEmbedder

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
            # margin = 1.0
            # emb1 = F.normalize(emb1, p=2, dim=1)
            # emb2 = F.normalize(emb2, p=2, dim=1)
            cos = F.cosine_similarity(emb1, emb2)
            logits = torch.stack([-cos, cos], dim=1)
            d = 1-cos # range == [0..2]
            
            # original
            # losses = labels*d + (1-labels)*F.relu(self.margin - d)
            
            # labels == diff_norm
            margin = 1 + labels 
            losses = (labels==0).long()*d  + (1 - (labels==0).long()) * F.relu(margin - d)

        else:  # euclidean
            # get norm of each embedding
            # emb1_norm_a = emb1.norm(p=2, dim=-1)
            # emb2_norm_a = emb2.norm(p=2, dim=-1)
            
            #normalize embeddings...
            # emb1 = F.normalize(emb1, p=2, dim=-1)
            # emb2 = F.normalize(emb2, p=2, dim=-1)
            # dnorm = F.pairwise_distance(F.normalize(emb1, p=2, dim=-1), F.normalize(emb2, p=2, dim=-1), p=2)
            
            # get norm now
            # emb1_norm_b = emb1.norm(p=2, dim=-1)
            # emb2_norm_b = emb2.norm(p=2, dim=-1)
            
            #-------------------------------------------
            d = F.pairwise_distance(emb1, emb2, p=2)
            logits = torch.stack([d-1, 1-d], dim=1)
            
            # logits = torch.stack([d-1, dnorm], dim=1)

            # labels = abs(score diffs)
            # diffs = labels
            if self.use_diff_norm:
                margin = self.margin + labels * self.margin_mult
            else:
                margin = self.margin + (labels-1) * self.margin_mult
            
            losses = (labels==0).long() * d.pow(2) + (1 - (labels==0).long()) * F.relu(margin - d).pow(2)

        # Logits for binary classification
        output_dict = {"logits": logits, "loss": losses.mean()}

        return output_dict
    
#---------------------------------------------------------------------------------------
T,S = 0,0
      
@dataclass
class DataCollatorForSiameseNetwork:#(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: int = None
    pad_to_multiple_of: int = None
    return_tensors: str = "pt"
    padding_side: str = "left"  # right left
    use_diff_norm: bool = False

    def __post_init__(self):
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
            # max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors
        )
        
        batch_2_padded = self.tokenizer.pad(
            batch_2,
            padding=self.padding,
            # max_length=self.max_length,
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
        
#---------------------------------------------------------------------------------------
VERBOSE = 0

class SiameseCausalLM(nn.Module): # PreTrainedModel
    def __init__(self, model,
                similarity_type: Literal["cosine", "euclidean"] = "cosine",
                pooling_strategy: Literal["mean", "last"] = "mean",
                padding_side: Literal["left", "right"] = "left",
                projection_dim: float = 0.0,
                lm_loss_weight: float = 0.0,
                hidden_layer: int = -1,
                # use_diff: bool = True,
                use_diff_norm: bool = False,
                **kwargs
        ):
        super().__init__()
        # super().__init__(model.config)

        self.config = model.config
        self.pooling_strategy = pooling_strategy
        self.padding_side = padding_side
        self.lm_loss_weight = lm_loss_weight
        self.hidden_layer = hidden_layer
        # self.use_diff = use_diff
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
                                                      pooling_strategy=self.pooling_strategy,
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
        # last_token_offset = 4 if USE_SCORE else 0
        
        emb1, lm_loss_1 = self.get_embedding(input_ids_1, attention_mask_1)
        emb2, lm_loss_2 = self.get_embedding(input_ids_2, attention_mask_2)

        # Compute similarity
        output_dict = self.loss_function(emb1, emb2, labels)
        
        # if main, print shape of emb1 and emb2 and labels
        if is_main() and VERBOSE>1:
            print(f"emb1: {emb1.shape}")
            print(f"emb2: {emb2.shape}")
            print(f"labels: {labels.shape}")

        if self.lm_loss_weight>0:
            pair_loss = output_dict["loss"]
            lm_loss = (lm_loss_1 + lm_loss_2)/2 # average lm loss
            total_loss = pair_loss + self.lm_loss_weight * lm_loss

            output_dict["loss"] = total_loss
            output_dict["pair_loss"] = pair_loss
            output_dict["lm_loss"] = lm_loss
            
            if is_main() and VERBOSE>0:
                print(f"pair_loss: {pair_loss}")
                print(f"lm_loss: {lm_loss}")
                print(f"total_loss: {total_loss}")
                print('-'*40)

        return output_dict

#---------------------------------------------------------------------------------------

if script_args.use_4bit:
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)


# find all linear modules for LoRA
def find_all_linear_names(model, verbose=True):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    lora_module_names = list(lora_module_names)
    if verbose:
        printmain(f'\nLoRA target modules: {lora_module_names}\n')
    return lora_module_names
target_modules = find_all_linear_names(model)

# create lora config
peft_config = LoraConfig(
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    r=script_args.lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=target_modules,
)

# initialize peft model
printmain("Initializing peft model...")
model = get_peft_model(model, peft_config)

#--------------------------------------------------------------------------------------------------

from transformers import Trainer, TrainerCallback
from sklearn.metrics import precision_recall_curve, accuracy_score, precision_recall_fscore_support

# model = SiameseCausalLM(model, **asdict(script_args))
model = SiameseCausalLM(model, 
                        similarity_type=script_args.similarity_type, 
                        pooling_strategy=script_args.pooling_strategy,
                        padding_side=script_args.padding_side,
                        projection_dim=script_args.projection_dim,
                        lm_loss_weight=script_args.lm_loss_weight,
                        hidden_layer=script_args.hidden_layer,
                        # use_diff=script_args.use_diff,
                        use_diff_norm=script_args.use_diff_norm,
                        margin=script_args.margin,
                        margin_mult=script_args.margin_mult,
                        )

collator = DataCollatorForSiameseNetwork(tokenizer, 
                                         padding=True, 
                                         pad_to_multiple_of=4,
                                         padding_side=script_args.padding_side,
                                         use_diff_norm=script_args.use_diff_norm,
                                         )

from kappa import fkappa, ikappa
from sklearn.metrics import average_precision_score

def compute_metrics(pred):
    labels = pred.label_ids
    diffs = labels
    labels = (labels == 0)
    try:
        logits = pred.predictions
        preds = logits.argmax(-1)
    except:
        logits = pred.predictions[0]
        preds = logits.argmax(-1)

    # cosine:    [-1, 1] - scores are on [-1,1] for cosine similarity
    # euclidean: [?,  1] - euclidean distance, only bounded above by 1
    scores = logits[:, -1]

    # compute average precision (threshold-agnostic metric)
    ap = average_precision_score(labels, scores)

    dists = logits[:,0] + 1 # logits = [d-1, 1-d] or [-cos, cos] (dists = d or 1-cos)
    corr = np.corrcoef(dists, diffs)[0,1]
    # compute the mean & std of distance at each value of diff
    diff_vals = np.unique(diffs)
    diff_vals.sort()
    dist_means = [dists[diffs==d].mean() for d in diff_vals]
    dist_stds = [dists[diffs==d].std() for d in diff_vals]
    
    # normalize distances
    # dists_norm = logits[:,1] # logits = [d-1, dnorm]
    # distnorm_means = [dists_norm[diffs==d].mean() for d in diff_vals]
    # distnorm_stds = [dists_norm[diffs==d].std() for d in diff_vals]
    
    #-------------------------------------------------------------------------
    if is_main():
        print("\nDIFF:\tMEAN\t+/- STD")
        for v,m,s in zip(diff_vals, dist_means, dist_stds):
            print(f"{v:.4g}:\t{m:.4g}\t+/- {s:.4g}")
        print()
        # print("\nDIFF:\tMEAN\t+/- STD")
        # for v,m,s in zip(diff_vals, distnorm_means, distnorm_stds):
        #     print(f"{v:.4g}:\t{m:.4g}\t+/- {s:.4g}")
        # print()
    #-------------------------------------------------------------------------

    # precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')

    # find optimal threshold for f1 score
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2*recall*precision/(recall+precision)
    idx = np.argmax(f1_scores)
    prec, rec, thresh = precision[idx], recall[idx], thresholds[idx]
    f1 = np.max(f1_scores)

    # redefine preds using optimal threshold
    preds = (scores > thresh).astype(int)

    # quadratic weighted kappa
    kappa = ikappa(labels, preds)
    # k2 = fkappa(labels, scores * 0.5 + 0.5)

    metrics = {
        'AP': round(ap, 4),
        'QWK': round(kappa, 4),
        'F1': round(f1, 4),
        'Prec': round(prec, 4),
        'Recall': round(rec, 4),
        'Corr': round(corr, 4),
        # 'acc': round(acc, 4),
    }

    # return metrics
    return metrics

#---------------------------------------------------------------------------------------
# for custom test items

from glob import glob
from xgb_utils import run_xgb_on_items

def run_xgb(model, data, **kwargs):
    # if is_main(): start = time.time()
    
    embedder = HuggingfaceEmbedder(model=model.encoder, tokenizer=tokenizer)#, **kwargs)
    
    mean_qwks = {}
    for item_type, item_data in data.items():
        qwks = run_xgb_on_items(embedder, item_data, **kwargs) # **kwargs here instead of in HuggingfaceEmbedder initialization ??? 
        if qwks is not None:
            mean_qwk = round(np.mean(qwks), 4)
            item_name = f'QWK_{item_type.upper()}'
            mean_qwks[item_name] = mean_qwk
            print(f"\n{item_name} : {mean_qwk}\t{qwks}\n" + '-'*50)
    
    # print(f"\nTime to run xgb: {time.time()-start:.2f} seconds")
    
    # if dist.is_initialized():
    #     dist.barrier()
        
    if is_main():
        return mean_qwks
    else:
        return None

#---------------------------------------------------------------------------------------
from transformers import TrainerCallback, TrainerState, TrainerControl

# DDP VERSION
class XGBEvalCallbackDDP(TrainerCallback):
    def __init__(self, test_items, eval_function, eval_steps=20):
        self.eval_function = eval_function
        self.eval_steps = eval_steps
        # Initialize test data on all processes since we need it for distributed embedding computation

        self.test_data = None
        test_data = {}
        for item_type, item_list in test_items.items():
            cfg = get_config(item_type, items = item_list,
                             root_data_dir=root_data_dir,
                             root_prompt_dir=root_prompt_dir,
                            )
            test_data[item_type] = load_items(cfg)
        self.test_data = test_data
        
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called after each training step."""
        if state.global_step % self.eval_steps == 0:
            model = kwargs['model']
            model.eval()  # Set to evaluation mode
            
            # Run evaluation on all processes
            with torch.no_grad():
                metrics = self.eval_function(model, self.test_data, step=state.global_step)

            # Only log metrics on the main process
            if args.local_rank == 0:
                # Log metrics
                for metric_name, value in metrics.items():
                    state.log_history.append({
                        f"{metric_name}": value,
                        "step": state.global_step
                    })

                wandb.log(
                    {f"eval/{k}": v for k, v in metrics.items()},
                    step=state.global_step
                )
            
            # Make sure all processes are synchronized before continuing
            # if dist.is_initialized(): dist.barrier(device_ids=[torch.cuda.current_device()])
                
            # Make sure to return to training mode
            model.train()
            
if script_args.use_xgb:
    xgb_eval_callback = XGBEvalCallbackDDP(test_items, 
                                           partial(run_xgb,
                                                   padding_side=script_args.padding_side,
                                                   pooling_strategy=script_args.pooling_strategy,
                                                   hidden_layer=script_args.hidden_layer,
                                                   ),
                                           eval_steps = int(EVAL_MULT*training_args.eval_steps))

#---------------------------------------------------------------------------------------

from torch.utils.data import DataLoader, Sampler, DistributedSampler     
from tqdm import tqdm
from tqdm.auto import tqdm

#---------------------------------------------------------------------------------------

def siamese_argsort(lengths_1, lengths_2, batch_size, 
                    chunk_size=None,
                    bucket_size=None,
                    multiplier=4,
                    seed=-1):
    if chunk_size is None:
        chunk_size = batch_size * multiplier
    if bucket_size is None:
        bucket_size = chunk_size * multiplier
    if seed<0:
        indices = list(range(len(lengths_1)))
    else:
        indices = np.random.default_rng(seed=seed).permutation(len(lengths_1))
    
    batches = []
    for i in tqdm(range(0, len(indices), bucket_size), desc="Creating buckets", disable=not is_main()):
        bucket = indices[i:i + bucket_size]
        
        # Sort bucket by lengths_1
        left_lengths = [lengths_1[idx] for idx in bucket]
        bucket = [x for _, x in sorted(zip(left_lengths, bucket))]
        
        # for each chunk of bucket, re-sort by lengths_2
        for j in range(0, len(bucket), chunk_size):
            chunk = bucket[j:j + chunk_size]
            right_lengths = [lengths_2[idx] for idx in chunk]
            chunk = [x for _, x in sorted(zip(right_lengths, chunk))]
            
            # Create batches from sorted chunk
            for k in range(0, len(chunk), batch_size):
                batch = chunk[k:k + batch_size]
                if len(batch) == batch_size:
                    batches.append(batch)
            
    # flatten batches
    indices = [idx for batch in batches for idx in batch]
    return indices

def siamese_argsort_dynamic(
    lengths_1,
    lengths_2,
    max_size, 
    chunk_size=None,
    bucket_size=None,
    multiplier=4,
    seed=-1
):
    """
    Sort indices (for a Siamese/dual-input network) such that each batch
    respects dynamic constraints on the total left- and right-side lengths.

    Args:
        lengths_1 (List[int]): Lengths of left sequences.
        lengths_2 (List[int]): Lengths of right sequences.
        max_size (int): The max total area allowed for either the left or right sequences in a batch.
        chunk_size (int, optional): Size of each chunk after bucketing. 
            Defaults to `batch_size * multiplier` from the old version. 
            You can set something meaningful or let it default.
        bucket_size (int, optional): Size of each bucket. 
            Defaults to `chunk_size * multiplier` from the old version.
        multiplier (int, optional): Factor to determine default chunk and bucket sizes. Defaults to 4.
        seed (int, optional): For reproducible shuffle. If < 0, no fixed seed. Defaults to -1.

    Returns:
        List[int]: A permutation of indices that respects dynamic batch constraints.
    """
    num_replicas = dist.get_world_size() if dist.is_initialized() else 1

                
    # Default fallback for chunk_size/bucket_size if not provided
    if chunk_size is None:
        chunk_size = 4 * multiplier
    if bucket_size is None:
        bucket_size = chunk_size * multiplier
        
    indices = siamese_argsort(lengths_1, lengths_2, 1, chunk_size, bucket_size, seed=seed)
    batches, current_batch = [], []
    current_left_max, current_right_max = 0,0
    for idx in tqdm(indices, desc="Creating dynamic batches", disable=not is_main()):
        l_len, r_len = lengths_1[idx], lengths_2[idx]
        pos_left_max = max(current_left_max, l_len)
        pos_right_max = max(current_right_max, r_len)
        pos_left_size = pos_left_max * (1 + len(current_batch))
        pos_right_size = pos_right_max * (1 + len(current_batch))
        
        if (pos_left_size <= max_size) and (pos_right_size <= max_size):
            current_batch.append(int(idx))
            current_left_max = pos_left_max
            current_right_max = pos_right_max
        else:
            if current_batch:
                batches.append(current_batch)
            current_batch = [idx]
            current_left_max = l_len
            current_right_max = r_len
        
    if current_batch:
        batches.append(current_batch)
        
    if len(batches) % num_replicas != 0:
        padding_size = num_replicas - (len(batches) % num_replicas)
        batches += batches[:padding_size]
    
    return batches

#--------------------------------------------------------------------------------------- 


class LengthBasedSampler(Sampler):
    def __init__(self, 
                 lengths_1: List[int], 
                 lengths_2: List[int], 
                 batch_size: int, 
                 shuffle: bool = True, 
                 bucket_size_multiplier: int = script_args.bucket_size_multiplier, 
                 num_replicas: Optional[int] = None, 
                 rank: Optional[int] = None, 
                 seed: int = 0):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1   
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        
        self.lengths_1 = lengths_1
        self.lengths_2 = lengths_2
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bucket_size = self.num_replicas * batch_size * bucket_size_multiplier**2
        self.chunk_size = self.num_replicas * batch_size * bucket_size_multiplier
        
        self.num_samples = len(self.lengths_1)
        self.total_size = self.num_samples
        if self.total_size % (self.batch_size * self.num_replicas) != 0:
            self.total_size = (
                (self.total_size // (self.batch_size * self.num_replicas) + 1)
                * self.batch_size
                * self.num_replicas
            )
            
    def set_epoch(self, epoch: int):
        if epoch==0:
            self.epoch = int(epoch)
        else:
            self.epoch += 1
        print(f"EPOCH == {self.epoch}")
            
    def __len__(self):
        l = (self.total_size // self.num_replicas) #// self.batch_size
        return l
        
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        print(f"SEED == {self.seed + self.epoch}")
        
        indices = siamese_argsort(self.lengths_1, self.lengths_2, self.batch_size, self.chunk_size, self.bucket_size, seed=self.seed + self.epoch)
        
        # Add padding if needed (train only)
        if len(indices) < self.total_size:
            padding_size = self.total_size - len(indices)
            indices += indices[:padding_size]
            
        # Subset for this rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        
        return iter(indices)

 
class ListOfBatchesSampler(LengthBasedSampler):
    def __init__(self, 
                 lengths_1: List[int], 
                 lengths_2: List[int], 
                 max_size: int = 1000000,
                 batch_size: int = 4, 
                 shuffle: bool = True, 
                 bucket_size_multiplier: int = script_args.bucket_size_multiplier, 
                 num_replicas: Optional[int] = None, 
                 rank: Optional[int] = None, 
                 seed: int = 0):
        super().__init__(lengths_1, lengths_2, batch_size, shuffle, bucket_size_multiplier, num_replicas, rank, seed)
        self.max_size = max_size
        self.batches = {}
        
    def get_epoch_batches(self):
        if self.epoch not in self.batches:
            self.batches[self.epoch] = siamese_argsort_dynamic(self.lengths_1, self.lengths_2, self.max_size,
                                                            #    self.chunk_size, self.bucket_size, 
                                                               multiplier=script_args.bucket_size_multiplier, 
                                                               seed=self.seed + self.epoch)
        return self.batches[self.epoch]
    
    # def __len__(self):
    #     batches = self.get_epoch_batches()
    #     l = len(batches) // self.num_replicas
    #     return l
    
    def __len__(self):
        l = (self.total_size // self.num_replicas) // self.batch_size
        return l

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        print(f"SEED == {self.seed + self.epoch}")
        
        batches = self.get_epoch_batches()
        
        # Subset for this rank
        batches = batches[self.rank:len(batches):self.num_replicas]
        
        # Yield the entire mini-batch at once, so the DataLoader sees it as a "batch" directly
        for batch in batches:
            yield batch


from typing import Sized, Iterator
class DynamicBatchDataLoader(DataLoader):
    """
    Custom DataLoader that works with samplers that yield batches of indices
    instead of single indices. This allows for dynamic batch sizes.
    """
    def __init__(
        self,
        dataset: Sized,
        sampler: Iterator[List[int]],
        num_workers: int = 0,
        collate_fn = None,
        drop_last: bool = False,
        pin_memory: bool = False,
        timeout: float = 0,
        worker_init_fn = None,
        multiprocessing_context=None,
        generator=None
    ):
        # We pass batch_size=1 and batch_sampler=None to parent class
        # since we'll handle batching ourselves
        super().__init__(
            dataset=dataset,
            batch_size=1,  # Not used but required
            sampler=None,  # We'll handle batching
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=drop_last,
            pin_memory=pin_memory,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator
        )
        self.custom_batch_sampler = sampler

    def __iter__(self):
        # Create an iterator that yields batches based on the indices
        # provided by our custom batch sampler
        for batch_indices in self.custom_batch_sampler:
            # Get samples for current batch
            batch = [self.dataset[int(idx)] for idx in batch_indices]
            
            # Apply collate function if provided
            if self.collate_fn is not None:
                batch = self.collate_fn(batch)
                
            yield batch

    def __len__(self) -> int:
        # If the batch_sampler has a __len__, use it
        if hasattr(self.custom_batch_sampler, '__len__'):
            return len(self.custom_batch_sampler)
        # Otherwise, we can't know the length
        raise NotImplementedError("Dataset length cannot be determined")
              
#---------------------------------------------------------------------------------------

class LengthAwareTrainer(Trainer):
    # override initialization
    def __init__(self, *args, data_dir=None, hf_dataset_id=None, max_batch_area=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.hf_dataset_id = hf_dataset_id
        self.max_batch_area = max_batch_area
        self.train_sampler = None
        
        
    def check_lengths(self, dataset, lengths):
        # check sizes
        if len(dataset) != len(lengths):
            print(f"Length mismatch: dataset={len(dataset)}, lengths={len(lengths)}")
            return False
        # randomly check a few lengths against dataset
        indices = np.random.choice(len(dataset), 10)
        for i in indices:
            example = dataset[int(i)]
            len1, len2 = len(example['input_ids_1']), len(example['input_ids_2'])
            if len1 != lengths[i,0] or len2 != lengths[i,1]:
                print(f"Length mismatch: {len1} != {lengths[i,0]} or {len2} != {lengths[i,1]}")
                return False
        return True
           
    def compute_lengths(self, dataset):
        def _compute_lengths(examples):
            length_1, length_2 = [], []
            for ids1, ids2 in zip(examples['input_ids_1'], examples['input_ids_2']):
                length_1.append(len(ids1))
                length_2.append(len(ids2))
            return { 'length_1': length_1, 'length_2': length_2 }
        lengths = dataset.map(_compute_lengths, batched=True, desc="Computing lengths")
        lengths_1, lengths_2 = lengths['length_1'], lengths['length_2']
        # create 2 column numpy array of lengths
        lengths = np.stack([lengths_1, lengths_2], axis=1)
        return lengths
    
    def get_lengths(self, data_split):
        if data_split not in ['train', 'eval']:
            raise ValueError(f"Invalid data split: {data_split}")
        dataset = self.train_dataset if data_split == 'train' else self.eval_dataset
        data_split_id = f'{self.hf_dataset_id}_{data_split}'.replace("/", "_")
        
        # check if lengths are already cached as 2 column numpy array
        lengths_dir = os.path.expanduser(f"{self.data_dir}/lengths")
        lengths_file = f"{lengths_dir}/{data_split_id}_lengths.npy"
        if os.path.exists(lengths_file):
            lengths = np.load(lengths_file)
            if not self.check_lengths(dataset, lengths):
                print("Lengths do not match dataset. Recomputing...")
                lengths = self.compute_lengths(dataset)
        else:
            if is_main():
                print("Computing lengths...")
                lengths = self.compute_lengths(dataset)
                os.makedirs(lengths_dir, exist_ok=True)
                np.save(lengths_file, lengths)
            dist.barrier()
            lengths = np.load(lengths_file)
                
        return lengths[:,0], lengths[:,1]
    
     
    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        lengths_1, lengths_2 = self.get_lengths('eval')
        
        # get sorted indices
        indices = siamese_argsort(lengths_1, lengths_2, self.args.eval_batch_size)
        
        # sort eval_dataset by indices
        eval_dataset = eval_dataset.select(indices)
        
        return super().get_eval_dataloader(eval_dataset)
        
    
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        lengths_1, lengths_2 = self.get_lengths('train')
        # print(f"\nLengths: {len(lengths_1)}")
        
        #---------------------------------------------------------
        # use DynamicBatchDataLoader
        
        if self.max_batch_area>0:
            
            self.train_sampler = ListOfBatchesSampler(
                lengths_1=lengths_1,
                lengths_2=lengths_2,
                max_size=self.max_batch_area,
                batch_size=self.args.train_batch_size,
                shuffle=True,
                # seed=self.args.seed,
                seed=script_args.rand_seed
            )
            
            return DynamicBatchDataLoader(
                self.train_dataset,
                sampler=self.train_sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
            
        #---------------------------------------------------------
        # default to Padding Reducing Sampler
        
        self.train_sampler = LengthBasedSampler(
            lengths_1=lengths_1,
            lengths_2=lengths_2,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            # seed=self.args.seed,
            seed=script_args.rand_seed
        )
            
        # Create DataLoader
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=self.train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


#---------------------------------------------------------------------------------------
# Create Trainer

if script_args.bucket_size_multiplier>0 or script_args.max_batch_area>0:
    # use padding-reducing sampler
    trainer = LengthAwareTrainer(
        model=model,
        args=training_args,
        data_dir=script_args.data_dir,
        hf_dataset_id=hf_dataset_id,
        max_batch_area=script_args.max_batch_area,
        train_dataset=llm_dataset['train'],
        eval_dataset=llm_dataset['validation'],
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[xgb_eval_callback] if script_args.use_xgb else None,
        
        ### preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        ### callbacks=[LossLoggingCallback()],
    )
else:
    # use default sampler
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=llm_dataset['train'],
        eval_dataset=llm_dataset['validation'],
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[xgb_eval_callback] if script_args.use_xgb else None,
    )

#--------------------------------------------------------------------------------------------------
# Add callbacks for saving model and deleting global steps

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from safetensors.torch import save_file
import shutil

def remove_dir(dir_path):
    try:
        shutil.rmtree(dir_path)
    except Exception as e:
        print(f'Failed to delete "{dir_path}". Reason: {e}')

def remove_dirs(dir_paths):
    for dir_path in dir_paths:
        remove_dir(dir_path)

def remove_files(file_paths):
    for file_path in file_paths:
        try:
            os.remove(file_path)
        except Exception as e:
            print(f'Failed to delete "{file_path}". Reason: {e}')

#---------------------------------------------------------------------------------------
 
class SaveDeepSpeedPeftModelCallback(TrainerCallback):
    def __init__(self, trainer, save_steps=500):
        self.trainer = trainer
        self.save_steps = save_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # if (state.global_step + 1) % self.save_steps == 0:
        if (state.global_step) % self.save_steps == 0:
            self.trainer.accelerator.wait_for_everyone()
            
            model = self.trainer.model_wrapped if hasattr(self.trainer, "model_wrapped") else self.trainer.model
            unwrapped_model = self.trainer.accelerator.unwrap_model(model)
            
            if self.trainer.accelerator.is_main_process:
                ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
                unwrapped_model.encoder.save_pretrained(ckpt_dir, **kwargs)
                
            self.trainer.accelerator.wait_for_everyone()
        return control

uploaded_checkpoints = set()
upload_attempts = {}
max_attempts = 3
def push_checkpoint_to_hub(repo_id, local_model_path, checkpoint_name=None, token=None):
    """
    Push a specific checkpoint to Hugging Face Hub under a named reference.
    
    Args:
        local_model_path (str): Path to local model checkpoint directory
        repo_id (str): Destination repo ID (format: "username/repo-name")
        checkpoint_name (str): Name for this checkpoint (e.g., "v1.0", "epoch_10", etc.)
        token (str, optional): HF API token. If None, will use token from ~/.huggingface/token
    """
    #---------------------------------------------------------------------------------------
    if local_model_path in uploaded_checkpoints:
        print(f"Checkpoint {local_model_path} already uploaded. Skipping.")
        return
    if local_model_path in upload_attempts:
        if upload_attempts[local_model_path] >= max_attempts:
            print(f"Max attempts reached for {local_model_path}. Skipping.")
            return
        upload_attempts[local_model_path] += 1
    else:
        upload_attempts[local_model_path] = 1
    #---------------------------------------------------------------------------------------
    
    api = HfApi()
    
    # Create repo if it doesn't exist
    api.create_repo(repo_id=repo_id, exist_ok=True, private=True)
    
    if checkpoint_name is None:
        checkpoint_name = os.path.basename(local_model_path)
    
    try:
        # First, create the branch from the main branch
        api.create_branch(
            repo_id=repo_id,
            branch=checkpoint_name,
            exist_ok=True  # Will not fail if branch already exists
        )
    except Exception as e:
        print(f"Note: Branch creation returned: {e}")
        # Continue anyway as the error might just be that the branch already exists
        # Note: Branch creation returned: 429 Client Error: Too Many Requests for url: https://huggingface.co/api/models/davidsvaughn/math_pairs_279_Llama-3.2-3B-Instruct_vwk4bic0/branch/checkpoint-650 (Request ID: Root=1-67abfbc0-0c2f44247a8c42ab45803244;0d311147-783b-467e-b889-c25b00f5b816)

    # Upload the checkpoint to a specific reference
    print(f"Uploading checkpoint to {repo_id}:{checkpoint_name}")
    try:
        api.upload_folder(
            folder_path=local_model_path,
            repo_id=repo_id,
            repo_type="model",
            revision=checkpoint_name,  # This creates a new branch/ref
            # token=token
        )
    except Exception as e:
        print(f"Upload failed: {e}")
        return
    uploaded_checkpoints.add(local_model_path)
    print("Done")
    
    
class DeleteGlobalStepsCallback(TrainerCallback):
    def __init__(self, trainer, save_steps=500):
        self.trainer = trainer
        self.save_steps = save_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if (state.global_step + 1) % self.save_steps == 0:
            self.trainer.accelerator.wait_for_everyone()
            if self.trainer.accelerator.is_main_process:
                #---------------------------------------------------------------------------------------
                # Remove global steps
                remove_dirs(glob(os.path.join(args.output_dir, 'checkpoint-*', 'global_step*')))
                # Remove model.safetensors files, only keep adapter weights
                remove_files(glob(os.path.join(args.output_dir, 'checkpoint-*', 'model.safetensors')))
                #---------------------------------------------------------------------------------------
                
                checkpoints = glob(os.path.join(args.output_dir, 'checkpoint-*'))
                top = args.output_dir.split('/')[-1]
                # repo_id = f"davidsvaughn/{top}"
                prefix = hf_dataset_id.replace("_tokenized", "")
                repo_id = f"{prefix}_{wandb_run_id}"
                # ##push all checkpoints to hf hub
                # for ckpt_dir in checkpoints:
                #     push_checkpoint_to_hub(repo_id, ckpt_dir)
                    
                #---------------------------------------------------------------------------------------
                # TODO : fix this the right way!!!
                # remove_dirs(glob(os.path.join(args.output_dir, 'checkpoint-*')))
                # remove_files(glob(os.path.join(args.output_dir, 'checkpoint-*', 'model.safetensors')))  # remove model.safetensors files
                #---------------------------------------------------------------------------------------
            self.trainer.accelerator.wait_for_everyone()
            
            # does this help?
            clear_cuda_tensors()
            
        return control

trainer.add_callback(DeleteGlobalStepsCallback(trainer, save_steps=training_args.save_steps))
trainer.add_callback(SaveDeepSpeedPeftModelCallback(trainer, save_steps=training_args.save_steps))
#---------------------------------------------------------------------------------------

class SetEpochCallback(TrainerCallback):
    """
    Trigger re-computing subset for dataset Examples-proportional mixing, see `dataset::ProportionMixingDataset`
    A hack that modifies the train dataset, pointed by Trainer's dataloader
    """
    def __init__(self, trainer: Trainer):
        self.trainer = trainer

    def on_epoch_begin(self, args: TrainingArguments, state, control, **kwargs):
        if hasattr(self.trainer, "train_sampler") and self.trainer.train_sampler is not None:
            self.trainer.train_sampler.set_epoch(state.epoch)
        return control
    
trainer.add_callback(SetEpochCallback(trainer))
        
#--------------------------------------------------------------------------------------------------

# clear_cuda_tensors()

# train model
trainer.train()