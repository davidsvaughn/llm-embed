from dataclasses import dataclass, field
from typing import Optional
import torch
# from peft import AutoPeftModelForCausalLM
from peft import PeftModel, PeftConfig
import os, sys
from transformers import AutoTokenizer, HfArgumentParser, AutoModelForCausalLM
from utils import adict, read_json

# execute script
# python merge_adapter.py --checkpoint_dir /home/azureuser/llm-embed/output2/checkpoint-2400
# python merge_adapter.py --checkpoint_dir /home/azureuser/llm-embed/checkpoint-2400 --output_dir model-600

@dataclass
class ScriptArguments:
    checkpoint_dir: str = field(metadata={"help": "path to checkpoint"})
    output_dir: Optional[str] = field(default="model", metadata={"help": "where the merged model should be saved"})


def load_peft_model(checkpoint_dir):
    config = PeftConfig.from_pretrained(checkpoint_dir)

    # tokenizer
    parent_dir = os.path.dirname(checkpoint_dir)
    tok_dir = os.path.join(parent_dir, 'tokenizer')
    if os.path.exists(tok_dir):
        print(f"\nLoading tokenizer from {tok_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    else:
        base_model_id = config.base_model_name_or_path
        print(f"\nLoading tokenizer from {base_model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.padding_side = 'right'

    # model
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # device_map="auto",
        device_map={"":0},
    )

    n = model.get_output_embeddings().weight.shape[0]
    if n < len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        print(f"Resized model to {len(tokenizer)} tokens")

    # peft model
    model = PeftModel.from_pretrained(model, checkpoint_dir)

    return model, tokenizer

def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # get output directory name
    parent_dir = os.path.dirname(args.checkpoint_dir)
    
    # if output_dir begins with ~ then expanduser
    if args.output_dir.startswith('~'):
        args.output_dir = os.path.expanduser(args.output_dir)
    elif not args.output_dir.startswith('/'):
        output_dir = os.path.join(parent_dir, args.output_dir)

    # load LoRA (adapter) model
    config = PeftConfig.from_pretrained(args.checkpoint_dir)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # device_map={"":0},
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, args.checkpoint_dir)

    # Merge LoRA and base model and save
    model = model.merge_and_unload()
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="4GB")
    print(f'Merged model saved to {output_dir}')

    # save tokenizer
    tok_dir = os.path.join(parent_dir, 'tokenizer')
    if os.path.exists(tok_dir):
        print(f'Loading tokenizer from {tok_dir}')
        tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    else:
        # base_model_id = config.base_model_name_or_path
        base_model_id = adict(read_json(os.path.join(output_dir, 'config.json')))._name_or_path
        print(f'Loading tokenizer from {base_model_id}')
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right' # to prevent warnings
        # tokenizer = get_tokenizer(base_model_id)
    tokenizer.save_pretrained(output_dir)
    print(f'Tokenizer saved to {output_dir}')


if __name__ == "__main__":
    main()
