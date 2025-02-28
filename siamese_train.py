#--------------------------------------------------------------------------------------------------
# Supervised Fine-Tuning for Siamese Networks
# This file contains the training pipeline for Siamese networks using the model defined in siamese_model.py

import os
import sys
import json
import random
import numpy as np
import torch
import torch.distributed as dist
import wandb
import shutil
from glob import glob
from jinja2 import Template
from functools import partial
from typing import Dict, List, Union, Optional
from dataclasses import dataclass, asdict, field
from sklearn.metrics import precision_recall_curve, average_precision_score

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    HfArgumentParser,
    Trainer, 
    TrainerCallback, 
    TrainerState, 
    TrainerControl,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

from ddp_utils import is_main, main, printmain, get_num_processes
from utils import clear_cuda_tensors, to_adict
from data_utils import get_config, load_items
from xgb_utils import run_xgb_on_items
from kappa import ikappa
from model_utils import apply_chat_template
from embedder.huggingface import HuggingfaceEmbedder
from pathlib import Path

# Import Siamese model components
from siamese_model import (
    SiameseCausalLM, 
    DataCollatorForSiameseNetwork, 
    tokenize_dataset,
)

#--------------------------------------------------------------------------------------------------
# Argument Parsing
# - cmd line arguments take precedence over these defaults
# - can be run with no cmd line arguments, or with any of the following
# - all arguments are logged to wandb

# Standard approach (already used)
# cur_dir = os.path.dirname(os.path.abspath(__file__))

# Alternative using pathlib (more modern, object-oriented)
# cur_dir = Path(__file__).parent.absolute()

# If you want the absolute path as string
cur_dir = str(Path(__file__).parent.absolute())
deepspeed_dir = os.path.join(cur_dir, 'deepspeed')


@dataclass
class ScriptArguments:
    
    model_id:       str             = field(default="meta-llama/Llama-3.2-3B-Instruct", metadata={"help": "The HuggingFace model id"})
    # model_id:       str             = field(default="microsoft/Phi-4-mini-instruct", metadata={"help": "The HuggingFace model id"})
    # model_id:       str             = field(default="microsoft/Phi-3-mini-128k-instruct", metadata={"help": "The HuggingFace model id"})
    # model_id:       str             = field(default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", metadata={"help": "The HuggingFace model id"})
    
    dataset_id:     List[str]       = field(default_factory=lambda: ["davidsvaughn/math_pairs_460"], metadata={"help": "The HuggingFace dataset id"})
    # dataset_id:     List[str]       = field(default_factory=lambda: ["davidsvaughn/math_pairs_1426"], metadata={"help": "The HuggingFace dataset id"})
    
    prompt_dir:     Optional[str]   = field(default="prompts", metadata={"help": "The prompt directory"})
    # data_dir:       Optional[str]   = field(default='~/data', metadata={"help": "The directory to store the training data"})
    test_data_dir:  Optional[str]   = field(default='~/embed/data', metadata={"help": "The directory where the test data is stored"})
    rand_seed:      Optional[int]   = field(default=1357, metadata={"help": "The random seed to use"})
    max_seq_length: Optional[int]   = field(default=2048, metadata={"help": "The maximum sequence length"})
    subsample_train:Optional[float] = field(default=1000000, metadata={"help": "The number of training samples to use"})
    subsample_eval: Optional[float] = field(default=20000, metadata={"help": "The number of evaluation samples to use"})
    max_samples:    Optional[int]   = field(default=1000000, metadata={"help": "The maximum number of samples to load"})
    
    # LoRA parameters
    lora_alpha:     Optional[int]   = field(default=32, metadata={"help": "The LoRA alpha parameter"})
    lora_r:         Optional[int]   = field(default=32, metadata={"help": "The LoRA r parameter"})
    lora_dropout:   Optional[float] = field(default=0.1, metadata={"help": "The LoRA dropout rate"})
    
    # Model parameters
    attn_implementation:Optional[str]   = field(default="flash_attention_2", metadata={"help": "The attention implementation to use"})
    use_4bit:           Optional[bool]  = field(default=False, metadata={"help": "Whether to use 4-bit quantization"})
    use_double_quant:   Optional[bool]  = field(default=False, metadata={"help": "Whether to use double quantization"})
    shuffle:            Optional[bool]  = field(default=True, metadata={"help": "Whether to shuffle the training data"})
    deepspeed_template: Optional[str]   = field(default="zero2_decay_template.j2", metadata={"help": "The deepspeed configuration file"})
    prompt_loss_weight: Optional[float] = field(default=1.0, metadata={"help": "The prompt loss weight"})

    # Siamese Network Arguments
    padding_side:   Optional[str]   = field(default="left", metadata={"help": "The padding side to use"})       # left | right
    projection_dim: Optional[float] = field(default=0.0, metadata={"help": "The projection dimension to use"})
    hidden_layer:   Optional[int]   = field(default=-1, metadata={"help": "The hidden layer to use"})
    use_xgb:        Optional[bool]  = field(default=True, metadata={"help": "Whether to run XGBoost"})
    similarity_type:Optional[str]   = field(default="euclidean", metadata={"help": "The similarity type to use"})  # cosine | euclidean | angular
    use_diff_norm:  Optional[bool]  = field(default=False, metadata={"help": "use normalized score diffs in loss"})
    margin:         Optional[float] = field(default=2.0, metadata={"help": "The margin to use in contrastive loss"})
    margin_mult:    Optional[float] = field(default=1.0, metadata={"help": "The margin multiplier to use in contrastive loss"})
    lm_loss_weight: Optional[float] = field(default=0.001, metadata={"help": "The language model loss weight"})
    pooling_mode:   Optional[str]   = field(default="mean", metadata={"help": "The pooling mode to use"})   # mean | lasttoken
    
    ## *prompt templates are now applied during dataset creation, not here ##
    # prompt_template:str             = field(default="prompts/math/user.j2", metadata={"help": "The prompt template to use"})
    # use_jinja2:     Optional[bool]  = field(default=True, metadata={"help": "Whether to use jinja2 templates"})

# Setup default training arguments
training_args = TrainingArguments(
    num_train_epochs                = 6,
    per_device_train_batch_size     = 4,
    per_device_eval_batch_size      = 4,
    gradient_accumulation_steps     = 8,
    remove_unused_columns           = False,
    logging_strategy                = "steps", 
    eval_strategy                   = "steps",
    output_dir                      = cur_dir+'/output',
    logging_steps                   = 10,
    eval_steps                      = 50,
    save_steps                      = 50,
    bf16                            = True,
    tf32                            = True,
    max_grad_norm                   = 0.3,
    warmup_ratio                    = 0.05,
    weight_decay                    = 0.001,
    lr_scheduler_type               = "constant_with_warmup",
    gradient_checkpointing_kwargs   = {"use_reentrant": True},
    report_to                       = "wandb",
    learning_rate                   = 1e-4,
    deepspeed = deepspeed_dir+"/zero2_decay_lr1.json" if 'LOCAL_RANK' in os.environ else None,
)

# Parse command line arguments
if '--help' in sys.argv:
    HfArgumentParser((ScriptArguments, TrainingArguments)).parse_args_into_dataclasses()

parser = HfArgumentParser(ScriptArguments)
script_args, cmd_train_args = parser.parse_known_args()

# Convert string literals to proper types
def totype(s):
    try: return eval(s)
    except: return s

# Apply command line arguments to override defaults
for k,v in zip(cmd_train_args[::2], cmd_train_args[1::2]):
    setattr(training_args, k.lstrip('-'), totype(v))

# Handle DeepSpeed configuration
if is_main():
    if 'deepspeed_template' in script_args and script_args.deepspeed_template and training_args.deepspeed:
        with open(deepspeed_dir+'/'+getattr(script_args, 'deepspeed_template'), 'r') as f:
            ds_template = Template(f.read())
        lr = training_args.learning_rate
        ds_config = ds_template.render(lr=lr)
        with open(training_args.deepspeed, 'w') as f:
            f.write(ds_config)

# # Handle prompt template files (*prompt templates are now applied during dataset creation, not here)  
# for k in ['prompt_template']:
#     if getattr(script_args, k).endswith('.j2'):
#         script_args.use_jinja2 = True
#         with open(cur_dir+'/'+getattr(script_args, k), 'r') as f:
#             setattr(script_args, k, f.read())

# Cast to dataclass
script_args = ScriptArguments(**vars(script_args))  # Cast namespace to dataclass

#---------------------------------------------------------------------------------------
# Configuration for evaluation and testing

# Test item IDs for different datasets
math_items = [123362, 29632]
bw_items = [33234]
fw_items = [57236]

# Select test items based on dataset
test_items = {}
if 'math' in script_args.dataset_id[0]:
    test_items['math'] = math_items
if 'bw' in script_args.dataset_id[0]:
    test_items['bw'] = bw_items
if 'fw' in script_args.dataset_id[0]:
    test_items['fw'] = fw_items

# Debug configurations
DEBUG = 3

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
        test_items = {'math': [123362]}
    elif DEBUG == 4:
        training_args.gradient_accumulation_steps = 2
        script_args.max_samples = 5000
        script_args.subsample_train = 500
        script_args.subsample_eval = 100
        training_args.eval_steps = 300          
        test_items = {'math': [123362]}
                    
#---------------------------------------------------------------------------------------

# Print configuration
if is_main():
    print('-'*100)
    print("\n-------- Training Arguments ---------")
    for key, value in asdict(training_args).items():
        print(f"{key}: {value}")

    print("\n-------- Script Arguments -----------")
    for key, value in asdict(script_args).items():
        print(f"{key}: {value}")
    print('-'*100)

    # Create or increment output directory
    i, prefix = 0, training_args.output_dir
    while os.path.exists(training_args.output_dir) and os.listdir(training_args.output_dir):
        i+=1
        training_args.output_dir = f"{prefix}{i}"
    os.makedirs(training_args.output_dir, exist_ok=True)
    print(f"OUTPUT DIRECTORY: {training_args.output_dir}")

#--------------------------------------------------------------------------------------------------
# Logging setup and random seed initialization

# Make data serializable for wandb
def make_json_serializable(d):
    def is_json_serializable(value):
        try:
            json.dumps(value)
            return True
        except (TypeError, OverflowError):
            return False
    return {k: v if is_json_serializable(v) else str(v) for k, v in d.items()}

# Initialize wandb logging
if is_main():
    wandb.init(project= f"Embedding-LLM--{script_args.dataset_id[0].replace('/','_')}")
    wandb.config.update(make_json_serializable(asdict(training_args)))
    wandb.config.update(make_json_serializable(asdict(script_args)))
    wandb_run_name = wandb.run.name
    wandb_run_id = wandb.run.id
    print(f"Wandb run name: {wandb_run_name}, id: {wandb_run_id}")
    print()

# Set random seeds
torch.manual_seed(script_args.rand_seed)
np.random.seed(script_args.rand_seed)
random.seed(script_args.rand_seed)

#--------------------------------------------------------------------------------------------------
# Dataset Loading and Dataset Hub Management

def check_dataset_exists(dataset_id):
    api = HfApi()
    try:
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

    # Shuffle dataset
    dataset = dataset.shuffle(seed=args.rand_seed)
    
    # Print splits and number of samples
    dataset_keys = list(dataset.keys())
    for k in dataset_keys:
        print(f"Number of {k} samples: {len(dataset[k])}")

        # Subsample if needed
        if len(dataset[k]) > args.max_samples:
            dataset[k] = dataset[k].shuffle(seed=args.rand_seed).select(range(args.max_samples))
            print(f"Randomly selected {args.max_samples} samples from {k} split")

    # Create validation split if needed
    if 'validation' not in dataset_keys:
        dataset = dataset['train'].train_test_split(test_size=0.1, seed=args.rand_seed)
        dataset['validation'] = dataset.pop('test')
        dataset_keys = list(dataset.keys())
        
    # Template for old method formatting
    def format_prompt(inputs, template_text, tokenizer):
        template = Template(template_text)
        user_text = template.render(**inputs).strip()
        msgs = [{"role": "user", "content": user_text}]
        txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return txt + "{'score':'"
    
    # Format each sample with the appropriate template
    def format_sample(sample):
        try: # now the prompt templates are applied during dataset creation, not in here
            payload1 = to_adict(json.loads(sample['payload1']))
            payload2 = to_adict(json.loads(sample['payload2']))
            sample["txt1"] = apply_chat_template(tokenizer, payload1)
            sample["txt2"] = apply_chat_template(tokenizer, payload2)
        except:
            # Fall back to old method - when prompt templates were applied here
            sample['txt1'] = format_prompt({**sample, 'text': sample['text1'] }, args.prompt_template, tokenizer)
            sample['txt2'] = format_prompt({**sample, 'text': sample['text2'] }, args.prompt_template, tokenizer)

        return sample

    # Determine which columns to keep (preserve labels)
    remove_columns = list(dataset[k].features)
    for name in ['diff', 'diff_norm', 'diff_abs']:
        try: remove_columns.remove(name)  # Keep these columns
        except: pass
    
    # Format samples and apply chat template
    print(f"\nFormatting samples (applying chat template...)")
    dataset = DatasetDict({ k : dataset[k].map(format_sample, 
                                              remove_columns=remove_columns) for k in dataset_keys })

    # Tokenize the dataset
    dataset = DatasetDict({ k : tokenize_dataset(dataset[k], tokenizer, args) for k in dataset_keys })

    # Report final dataset sizes
    for k in dataset_keys:
        print(f"Total count of {k} tokenized sequences: {len(dataset[k])}")
        
    return dataset

# Load or create tokenized dataset
def load_or_prepare_dataset(tokenizer, args):
    dataset_tag = args.dataset_id[0]
    dataset_tag = dataset_tag.replace("davidsvaughn/", "").replace("/", "_")
    model_tag = args.model_id.split('/')[-1]
    data_tag = f'{dataset_tag}_{model_tag}_tokenized'
    hf_dataset_id = f"davidsvaughn/{data_tag}"
    
    # Check if dataset already exists
    if check_dataset_exists(hf_dataset_id):
        return hf_dataset_id
    
    # Prepare and tokenize dataset
    dataset = prepare_dataset(tokenizer, args)
    
    # Upload to HuggingFace Hub
    if dataset is not None:
        print(f"\nPushing tokenized dataset to HuggingFace hub: {hf_dataset_id}")
        dataset.push_to_hub(hf_dataset_id, private=True)
        if get_num_processes() == 1:
            sys.exit()
        else:
            # Synchronize all processes
            dist.barrier()
    return hf_dataset_id

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(script_args.model_id)
tokenizer.padding_side = script_args.padding_side
tokenizer.pad_token = tokenizer.eos_token

# Load or prepare the dataset
hf_dataset_id = load_or_prepare_dataset(tokenizer, script_args)
print(f"\nLoading tokenized dataset from HuggingFace hub: {hf_dataset_id}")
llm_dataset = load_dataset(hf_dataset_id)

# Subsample dataset if needed
def random_subset(dataset, n):
    m = len(dataset)
    if n <= 0 or n >= m: return dataset
    n = int(m*n) if n < 1 else int(n)
    idx = np.random.permutation(m)
    return dataset.select(idx[:n])

# Apply subsampling to train and validation sets
for k, n in zip(['train', 'validation'], [script_args.subsample_train, script_args.subsample_eval]):
    m = len(llm_dataset[k])
    llm_dataset[k] = random_subset(llm_dataset[k], n)
    printmain(f"Using {len(llm_dataset[k])} of {m} {k} sequences")

#--------------------------------------------------------------------------------------------------
# Model Initialization

# Configure quantization if needed
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=script_args.use_double_quant,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
) if script_args.use_4bit else None

# Load pretrained model
model = AutoModelForCausalLM.from_pretrained(
    script_args.model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation=script_args.attn_implementation,
    use_cache=not training_args.gradient_checkpointing,
    quantization_config=bnb_config,
    trust_remote_code=True,
)

# Prepare model for quantized training if needed
if script_args.use_4bit:
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

# Find linear modules for LoRA
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

# Configure LoRA
peft_config = LoraConfig(
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    r=script_args.lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=target_modules,
)

# Initialize PEFT model
printmain("Initializing peft model...")
model = get_peft_model(model, peft_config)

# Wrap model in Siamese architecture
model = SiameseCausalLM(
    model, 
    similarity_type=script_args.similarity_type, 
    pooling_mode=script_args.pooling_mode,
    padding_side=script_args.padding_side,
    projection_dim=script_args.projection_dim,
    lm_loss_weight=script_args.lm_loss_weight,
    hidden_layer=script_args.hidden_layer,
    use_diff_norm=script_args.use_diff_norm,
    margin=script_args.margin,
    margin_mult=script_args.margin_mult,
)

# Initialize data collator
collator = DataCollatorForSiameseNetwork(
    tokenizer, 
    padding=True, 
    pad_to_multiple_of=4,
    padding_side=script_args.padding_side,
    use_diff_norm=script_args.use_diff_norm,
)

#--------------------------------------------------------------------------------------------------
# Metrics and Evaluation

def compute_metrics(pred):
    labels = pred.label_ids
    diffs = labels
    labels = (labels == 0)  # Convert to binary labels
    
    # Get logits and predictions
    try:
        logits = pred.predictions
        preds = logits.argmax(-1)
    except:
        logits = pred.predictions[0]
        preds = logits.argmax(-1)

    # Get similarity scores
    scores = logits[:, -1]

    # Calculate average precision
    ap = average_precision_score(labels, scores)

    # Calculate distance and correlation metrics
    dists = logits[:,0] + 1  # logits = [d-1, 1-d] or [-cos, cos]
    corr = np.corrcoef(dists, diffs)[0,1]
    
    # Analyze distance by difficulty level
    diff_vals = np.unique(diffs)
    diff_vals.sort()
    dist_means = [dists[diffs==d].mean() for d in diff_vals]
    dist_stds = [dists[diffs==d].std() for d in diff_vals]
    
    # Print distance statistics if on main process
    if is_main():
        print("\nDIFF:\tMEAN\t+/- STD")
        for v, m, s in zip(diff_vals, dist_means, dist_stds):
            print(f"{v:.4g}:\t{m:.4g}\t+/- {s:.4g}")
        print()

    # Calculate F1 score at optimal threshold
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2*recall*precision/(recall+precision)
    idx = np.argmax(f1_scores)
    prec, rec, thresh = precision[idx], recall[idx], thresholds[idx]
    f1 = np.max(f1_scores)

    # Redefine predictions using optimal threshold
    preds = (scores > thresh).astype(int)

    # Calculate quadratic weighted kappa
    kappa = ikappa(labels, preds)

    # Return metrics
    metrics = {
        'AP': round(ap, 4),
        'QWK': round(kappa, 4),
        'F1': round(f1, 4),
        'Prec': round(prec, 4),
        'Recall': round(rec, 4),
        'Corr': round(corr, 4),
    }

    return metrics

#--------------------------------------------------------------------------------------------------
# XGBoost evaluation on test items

def run_xgb(model, data, **kwargs):
    embedder = HuggingfaceEmbedder(model=model.encoder, tokenizer=tokenizer)
    
    mean_qwks = {}
    for item_type, item_data in data.items():
        qwks = run_xgb_on_items(embedder, item_data, **kwargs)
        if qwks is not None:
            mean_qwk = round(np.mean(qwks), 4)
            item_name = f'QWK_{item_type.upper()}'
            mean_qwks[item_name] = mean_qwk
            print(f"\n{item_name} : {mean_qwk}\t{qwks}\n" + '-'*50)
        
    if is_main():
        return mean_qwks
    else:
        return None

# XGBoost evaluation callback
class XGBEvalCallbackDDP(TrainerCallback):
    def __init__(self, test_items, eval_function, eval_steps=20, prompt_dir='prompts', data_dir='data'):
        self.eval_function = eval_function
        self.eval_steps = eval_steps
        
        # Load test data
        self.test_data = None
        test_data = {}
        for item_type, item_list in test_items.items():
            cfg = get_config(
                item_type, 
                items=item_list,
                root_data_dir=data_dir,
                root_prompt_dir=prompt_dir,
            )
            test_data[item_type] = load_items(cfg)
        self.test_data = test_data
        
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % self.eval_steps == 0:
            model = kwargs['model']
            model.eval()  # Set to evaluation mode
            
            # Run evaluation
            with torch.no_grad():
                metrics = self.eval_function(model, self.test_data, step=state.global_step)

            # Log metrics on main process
            if args.local_rank == 0 and metrics:
                for metric_name, value in metrics.items():
                    state.log_history.append({
                        f"{metric_name}": value,
                        "step": state.global_step
                    })

                wandb.log(
                    {f"eval/{k}": v for k, v in metrics.items()},
                    step=state.global_step
                )
                
            # Return to training mode
            model.train()
            
# Initialize XGBoost callback if needed
if script_args.use_xgb:
    xgb_eval_callback = XGBEvalCallbackDDP(
        test_items, 
        partial(run_xgb,
                padding_side=script_args.padding_side,
                pooling_mode=script_args.pooling_mode,
                hidden_layer=script_args.hidden_layer,
        ),
        eval_steps=int(training_args.eval_steps),
        prompt_dir=script_args.prompt_dir,
        data_dir=script_args.test_data_dir,
    )

#--------------------------------------------------------------------------------------------------
# Callbacks for Model Management

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

# Save model callback
class SaveDeepSpeedPeftModelCallback(TrainerCallback):
    def __init__(self, trainer, save_steps=500):
        self.trainer = trainer
        self.save_steps = save_steps

    def on_step_end(self, args, state, control, **kwargs):
        if (state.global_step) % self.save_steps == 0:
            self.trainer.accelerator.wait_for_everyone()
            
            model = self.trainer.model_wrapped if hasattr(self.trainer, "model_wrapped") else self.trainer.model
            unwrapped_model = self.trainer.accelerator.unwrap_model(model)
            
            if self.trainer.accelerator.is_main_process:
                ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
                unwrapped_model.encoder.save_pretrained(ckpt_dir, **kwargs)
                
            self.trainer.accelerator.wait_for_everyone()
        return control

# Hub upload tracking
uploaded_checkpoints = set()
upload_attempts = {}
max_attempts = 3

def push_checkpoint_to_hub(repo_id, local_model_path, checkpoint_name=None, token=None):
    """Push a checkpoint to Hugging Face Hub under a named reference."""
    # Track upload attempts
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
    
    api = HfApi()
    
    # Create repo if needed
    api.create_repo(repo_id=repo_id, exist_ok=True, private=True)
    
    if checkpoint_name is None:
        checkpoint_name = os.path.basename(local_model_path)
    
    try:
        # Create branch for this checkpoint
        api.create_branch(
            repo_id=repo_id,
            branch=checkpoint_name,
            exist_ok=True
        )
    except Exception as e:
        print(f"Note: Branch creation returned: {e}")
        # Continue anyway as the error might just be that the branch already exists

    # Upload the checkpoint to the specified branch
    print(f"Uploading checkpoint to {repo_id}:{checkpoint_name}")
    try:
        api.upload_folder(
            folder_path=local_model_path,
            repo_id=repo_id,
            repo_type="model",
            revision=checkpoint_name,
        )
    except Exception as e:
        print(f"Upload failed: {e}")
        return
    uploaded_checkpoints.add(local_model_path)
    print("Done")

# Callback to clean up checkpoints and optionally push to hub
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
                # Remove global steps and model.safetensors (keep only adapter weights)
                remove_dirs(glob(os.path.join(args.output_dir, 'checkpoint-*', 'global_step*')))
                remove_files(glob(os.path.join(args.output_dir, 'checkpoint-*', 'model.safetensors')))
                
                # Setup for potential checkpoint pushing to hub
                checkpoints = glob(os.path.join(args.output_dir, 'checkpoint-*'))
                prefix = hf_dataset_id.replace("_tokenized", "")
                repo_id = f"{prefix}_{wandb_run_id}"
                
                # Uncomment to push checkpoints to hub
                # for ckpt_dir in checkpoints:
                #     push_checkpoint_to_hub(repo_id, ckpt_dir)
            
            self.trainer.accelerator.wait_for_everyone()
            
            # Clean up memory
            clear_cuda_tensors()
            
        return control

# Callback to help with epoch tracking in dataset samplers
class SetEpochCallback(TrainerCallback):
    """
    Trigger re-computing subset for dataset Examples-proportional mixing
    """
    def __init__(self, trainer: Trainer):
        self.trainer = trainer

    def on_epoch_begin(self, args: TrainingArguments, state, control, **kwargs):
        if hasattr(self.trainer, "train_sampler") and self.trainer.train_sampler is not None:
            self.trainer.train_sampler.set_epoch(state.epoch)
        return control

#--------------------------------------------------------------------------------------------------
# Set up trainer and callbacks

# Create trainer with model and datasets
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=llm_dataset['train'],
    eval_dataset=llm_dataset['validation'],
    data_collator=collator,
    compute_metrics=compute_metrics,
    callbacks=[xgb_eval_callback] if script_args.use_xgb else None,
)

# Add callbacks for model management
trainer.add_callback(DeleteGlobalStepsCallback(trainer, save_steps=training_args.save_steps))
trainer.add_callback(SaveDeepSpeedPeftModelCallback(trainer, save_steps=training_args.save_steps))
trainer.add_callback(SetEpochCallback(trainer))

#--------------------------------------------------------------------------------------------------
# Start training

if __name__ == "__main__":
    # Begin training
    trainer.train()
    
    # Save final model if needed
    if is_main():
        print("Training completed. Final model saved to:", training_args.output_dir)

