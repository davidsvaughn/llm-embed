import os
import sys
import torch
import time
import traceback
import argparse
from glob import glob
from sentence_transformers import SentenceTransformer, models

from utils import clear_cuda_tensors, mkdirs, tricky_traversal_order
from ddp_utils import init_distributed, is_main, printmain
from data_utils import get_config, load_items
from xgb_utils import run_xgb_on_item, run_xgb_on_items
from model_utils import load_checkpoint_model
from embedder.factory import EmbedderFactory
from embedder.huggingface import HuggingfaceEmbedder
from embedder.sent_trans import SentenceTransformerEmbedder

#---------------------------------------------------------------------------------------
'''
For DDP mode - Run this script with the following command:

    torchrun --nproc_per_node 4 siamese_test.py --scan

Or to generate predictions:

    python siamese_test.py --gen --overwrite --debug
'''
#---------------------------------------------------------------------------------------    

# save predictions for all items - useful for building "hard pairs" dataset for siamese model
def generate_predictions(args):
    overwrite = args.overwrite
    debug = args.debug
    items = [int(it.strip()) for it in args.items.split(",")] if args.items else None
    
    #-----------------------------------------------------------------------------
    # Get config for each item type
    cfg = get_config(args.item_type,
                     data_dir=args.data_dir,
                     prompt_dir=args.prompt_dir,
                     model_dir=args.model_dir,
                     model_id=args.model_id,
                     items=items,
                     filter_out_mult=args.filter_out_mult if not args.items else None,
                     extract_state_indices=args.extract_state_indices if args.extract_state_indices else None,
                    )
    
    # load items (debug mode only loads a subset of items)
    data_by_item = load_items(cfg, debug=debug)

    #-----------------------------------------------------------------------------

    # load HF model and tokenizer
    embedder = EmbedderFactory(cfg)
    
    #-----------------------------------------------------------------------------

    for i, (item, records) in enumerate(data_by_item.items()):
        printmain(f"\nProcessing item: {item} ({i+1}/{len(data_by_item)})")

        preds_file = cfg.preds_file_fmt.format(item=item)
        preds_dir = os.path.dirname(preds_file)
        mkdirs(preds_dir)

        # skip if predictions file exists... or in overwrite mode, or in debug mode
        if os.path.exists(preds_file) and not debug and not overwrite:
            printmain(f"Skipping item {item}... {preds_file} exists")
            continue
        
        try:
            # compute embeddings, run xgb k-fold, save predictions
            run_xgb_on_item(embedder, item, records, preds_file=preds_file, **cfg)
            
        except Exception as e:
            printmain(f"ERROR Processing item {item}:\n{e}")
            traceback.print_exc()
            sys.exit()

    printmain("DONE!")

#------------------------------------------------------------------------------

# output_dir contains multiple checkpoint directories
def scan_checkpoints(args):
    min_num = args.min_num
    max_num = args.max_num
    filters = args.filters
    pooling_mode = args.pooling_mode
    K = args.K
    items = [int(it.strip()) for it in args.items.split(",")] if args.items else None
    
    # Get config for each item type
    cfg = get_config(args.item_type,
                    random_state=args.random_state,
                    data_dir=args.data_dir,
                    prompt_dir=args.prompt_dir,
                    model_dir=args.model_dir,
                    items=items,
                    filter_in_mult=args.filter_in_mult if not args.items else None,
                    hh_min=args.hh_min if not args.items else None,
                    )

    # get checkpoint directories
    checkpoint_dirs = glob(args.model_dir + '/checkpoint-*')
    checkpoint_dirs = [d for d in checkpoint_dirs if int(d.split('-')[-1]) >= min_num and int(d.split('-')[-1]) <= max_num]
    checkpoint_dirs.sort(key=lambda x: int(x.split('-')[-1]))

    if filters is not None:
        checkpoint_nums = [int(d.split('-')[-1]) for d in checkpoint_dirs]
        checkpoint_dirs = [d for d in checkpoint_dirs if int(d.split('-')[-1]) in filters]
    else:
        checkpoint_dirs = tricky_traversal_order(checkpoint_dirs)

    # pop first two checkpoints and add to end
    if len(checkpoint_dirs) > 10:
        checkpoint_dirs = checkpoint_dirs[2:] + checkpoint_dirs[:2]

    #------------------------------------------------------------------------------

    # load test items
    data_by_item = load_items(cfg, debug=args.debug)

    # loop through checkpoints
    results = []
    for checkpoint_dir in checkpoint_dirs:
        printmain(f"{checkpoint_dirs}")
        printmain(f"\nRunning checkpoint: {checkpoint_dir}")
        start = time.time()
        
        embedder = EmbedderFactory(model_id=checkpoint_dir)
        
        qwks = run_xgb_on_items(embedder, data_by_item, K=K,
                                pooling_mode=pooling_mode,
                                random_state=cfg.get('random_state', args.random_state),
                                )
        
        if qwks is not None:
            qwk = qwks.mean()
            chkpt = checkpoint_dir.split('/')[-1]
            result = f'{chkpt}\t{qwk:.4f}'
            results.append(result)
            
            # sort results by checkpoint number
            results.sort(key=lambda x: int(x.split('\t')[0].split('-')[-1]))
            
            print('\n' + '-'*40)
            print(f'{args.model_dir}')
            for r in results:
                print(r)
            print('-'*40 + '\n')
            print(f"\nTIME FOR ITEM: {time.time()-start:.2f} SEC\n")
            
        if torch.distributed.is_initialized():
            torch.distributed.barrier()  # Wait for rank 0 to finish
            
        clear_cuda_tensors()

    if is_main():
        print("Done")

#------------------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Siamese model testing and prediction generation')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Common arguments
    parser.add_argument('--data-dir', default='data', help='Root data directory')
    parser.add_argument('--prompt-dir', default='prompts', help='Root prompt directory')
    parser.add_argument('--item-type', default='bw', choices=['bw', 'fw', 'math'], help='Item type')
    parser.add_argument('--pooling-mode', default='mean', choices=['mean', 'lasttoken'], help='Pooling mode')
    parser.add_argument('--random-state', type=int, default=42, help='Random state for reproducibility')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with limited items')
    
    # Generate predictions command
    gen_parser = subparsers.add_parser('gen', help='Generate predictions')
    gen_parser.add_argument('--overwrite', action='store_true', help='Overwrite existing predictions')
    gen_parser.add_argument('--model-dir', default='models', help='Directory containing downloaded models')
    gen_parser.add_argument('--model-id', default='dan-bw', help='Path to model')
    gen_parser.add_argument('--filter-out-mult', type=int, default=2, help='Filter out multiplier')
    gen_parser.add_argument('--extract-state-indices', type=int, nargs='+', help='State indices to extract')
    gen_parser.add_argument('--items', type=str, default='', help='Comma separated list of items')
    
    # Scan checkpoints command
    scan_parser = subparsers.add_parser('scan', help='Scan checkpoints')
    scan_parser.add_argument('--model-dir', default='output', help='Directory containing model checkpoints')
    scan_parser.add_argument('--min-num', type=int, default=0, help='Minimum checkpoint number')
    scan_parser.add_argument('--max-num', type=int, default=10000000, help='Maximum checkpoint number')
    scan_parser.add_argument('--K', type=int, default=5, help='Number of folds for cross-validation')
    scan_parser.add_argument('--filters', type=int, nargs='+', help='Specific checkpoint numbers to evaluate')
    scan_parser.add_argument('--filter-in-mult', type=int, default=2, help='Filter in multiplier')
    scan_parser.add_argument('--hh-min', type=float, default=0.6, help='Minimum human-human agreement')
    scan_parser.add_argument('--items', type=str, default='', help='Comma separated list of items')
    
    # For backward compatibility, also support --gen and --scan flags
    parser.add_argument('--gen', action='store_true', help='Generate predictions (legacy flag)')
    parser.add_argument('--scan', action='store_true', help='Scan checkpoints (legacy flag)')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Handle legacy flags for backward compatibility
    if args.gen:
        args.command = 'gen'
    elif args.scan:
        args.command = 'scan'
    
    if args.command == 'gen':
        generate_predictions(args)
    elif args.command == 'scan':
        scan_checkpoints(args)
    else:
        print("Please specify either 'gen' or 'scan' command")
        print("Example: python siamese_test.py gen --overwrite")
        print("      or python siamese_test.py scan --filters 2250")
        # # For backward compatibility
        # print("Legacy usage: python siamese_test.py --gen --overwrite")
        # print("         or: python siamese_test.py --scan")
        sys.exit(1)
