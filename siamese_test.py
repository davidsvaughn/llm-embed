import os
import sys
import torch
import time
import traceback
import argparse
from glob import glob
import numpy as np
# from sentence_transformers import SentenceTransformer, models

from utils import clear_cuda_tensors, mkdirs, tricky_traversal_order
from ddp_utils import init_distributed, is_main, printmain
from data_utils import get_config, load_items
from xgb_utils import run_xgb_on_item, run_xgb_on_items
from model_utils import load_checkpoint_model
from embedder.factory import EmbedderFactory
from embedder.huggingface import HuggingfaceEmbedder
from embedder.sent_trans import SentenceTransformerEmbedder

#---------------------------------------------------------------------------------------    

# save predictions for all items - useful for building "hard pairs" dataset for siamese model
def generate_predictions(args):

    args.items = [int(x.strip()) for x in args.items.split(",")] if args.items else None
    args.hh_min = args.hh_min if not args.items else None
    args.item_filter = args.item_filter if not args.items else None
    args.extract_state_indices = [int(x.strip()) for x in args.extract_state_indices.split(",")] if args.extract_state_indices else None
    
    # Filter out None and empty string values (leave '0' and 'False' as is)
    kwargs = {k: v for k, v in vars(args).items() if v is not None and v != ''}
    
    cfg = get_config(**kwargs)
    
    #-----------------------------------------------------------------------------
    # load items (debug mode only loads a subset of items)
    data_by_item = load_items(cfg, debug=args.debug)

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
        if os.path.exists(preds_file) and not args.debug and not args.overwrite:
            printmain(f"Skipping item {item}... {preds_file} exists")
            continue
        
        try:
            # compute embeddings, run xgb k-fold, save predictions
            run_xgb_on_item(embedder, item, records, preds_file=preds_file, **cfg)
            
        except Exception as e:
            printmain(f"ERROR Processing item {item}:\n{e}")
            traceback.print_exc()
            sys.exit()
            
        clear_cuda_tensors()

    printmain("DONE!")

#------------------------------------------------------------------------------

# output_dir contains multiple checkpoint directories
def scan_checkpoints(args):
    
    #------------------------------------------------------------------------------
    # # old code
    # args.chk_list = args.chk_list.split(",") if args.chk_list else None
    # pooling_mode = args.pooling_mode
    # K = args.K
    # items = [int(it.strip()) for it in args.items.split(",")] if args.items else None
    
    # cfg = get_config(args.item_type,
    #                  random_state=args.random_state,
    #                  data_dir=args.data_dir,
    #                  prompt_dir=args.prompt_dir,
    #                  model_dir=args.model_dir,
    #                  items=items,
    #                  hh_min=args.hh_min if not args.items else None,
    #                  filter_in_mult=args.filter_in_mult if not args.items else None,
    #                  item_filter=args.item_filter if not args.items else None,
    #                  )
    #------------------------------------------------------------------------------
    # new code
    
    args.items = [int(x.strip()) for x in args.items.split(",")] if args.items else None
    args.hh_min = args.hh_min if not args.items else None
    args.item_filter = args.item_filter if not args.items else None
    args.chk_list = args.chk_list.split(",") if args.chk_list else None
    
    # Filter out None and empty string values (leave '0' and 'False' as is)
    kwargs = {k: v for k, v in vars(args).items() if v is not None and v != ''}
    
    cfg = get_config(**kwargs)
    
    #------------------------------------------------------------------------------

    # get checkpoint directories
    checkpoint_dirs = glob(args.model_dir + '/checkpoint-*')
    checkpoint_dirs = [d for d in checkpoint_dirs if int(d.split('-')[-1]) >= args.chk_min and int(d.split('-')[-1]) <= args.chk_max]
    checkpoint_dirs.sort(key=lambda x: int(x.split('-')[-1]))

    if args.chk_list is not None:
        checkpoint_dirs = [d for d in checkpoint_dirs if d.split('-')[-1] in args.chk_list]
    else:
        checkpoint_dirs = tricky_traversal_order(checkpoint_dirs)
        # pop first two checkpoints and add to end
        if len(checkpoint_dirs) > 5:
            checkpoint_dirs = checkpoint_dirs[2:] + checkpoint_dirs[:2]

    #------------------------------------------------------------------------------

    # load test items
    data_by_item = load_items(cfg, debug=args.debug)
    
    #------------------------------------------------------------------------------
    
    # initialize Q table to store QWK values, for adaptive checkpoint scanning
    chk_nums = [int(d.split('-')[-1]) for d in checkpoint_dirs]
    chk_nums.sort()
    Q = np.array([[x, -1.0] for x in chk_nums])

    # loop through checkpoints
    i, results = 0, []
    while True:
        if len(checkpoint_dirs) == 0:
            break
        i += 1
        printmain(f"{checkpoint_dirs}")
        
        #------------------------------------------------------------------------------
        # adaptive checkpoint scanning
        if i<4:
            # visit first several checkpoints in order...
            checkpoint_dir = checkpoint_dirs.pop(0)
        else:
            # use Q-table to find the unvisited chkpt adjacent to the highest QWK chkpt so far...
            best = (-1,0)
            for checkpoint_dir in checkpoint_dirs:
                chk_num = int(checkpoint_dir.split('-')[-1])
                idx = np.where(Q[:,0]==chk_num)[0][0]
                qwk_left = Q[idx-1,1] if idx>0 else -1
                qwk_right = Q[idx+1,1] if idx<len(Q)-1 else -1
                qwk_max = max(qwk_left, qwk_right)
                if qwk_max > best[0]:
                    best = (qwk_max, chk_num)

            idx = [d.split('-')[-1] for d in checkpoint_dirs].index(str(best[1]))
            checkpoint_dir = checkpoint_dirs.pop(idx)
        #--------------------------------------------------------------------------
        
        start = time.time()
        printmain(f"\nRunning checkpoint: {checkpoint_dir}")
        
        # load embedder
        embedder = EmbedderFactory(model_id=checkpoint_dir)
        
        # run xgb on all items
        qwks = run_xgb_on_items(embedder, data_by_item, K=args.K,
                                pooling_mode=args.pooling_mode,
                                random_state=cfg.get('random_state', args.random_state),
                                )
        
        # only rank0 has qwks != None
        if qwks is not None:
            qwk = qwks.mean()
            chkpt = checkpoint_dir.split('/')[-1]
            result = f'{chkpt}\t{qwk:.4f}'
            results.append(result)
            
            # sort results by checkpoint number
            results.sort(key=lambda x: int(x.split('\t')[0].split('-')[-1]))
            
            # fill in Q table
            chk_num = int(checkpoint_dir.split('-')[-1])
            Q[Q[:,0]==chk_num,1] = qwk
            
            print('\n' + '-'*40)
            print(f'{args.model_dir}')
            for r in results:
                print(r)
            print('-'*40 + '\n')
            print(f"\nTIME FOR ITEM: {time.time()-start:.2f} SEC\n")
            
        # Q-table maintenance
        if torch.distributed.is_initialized():
            # broadcast Q to all ranks...
            Q = torch.tensor(Q, dtype=torch.float32)
        
            # Ensure all processes use the same device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            Q = Q.to(device)
            
            # Broadcast Q from rank 0 to all other ranks
            torch.distributed.broadcast(Q, src=0)
            
            # Convert back to numpy
            if device.type == 'cuda':
                Q = Q.cpu().numpy()
            else:
                Q = Q.numpy()
            
            # Wait for all processes to finish
            torch.distributed.barrier()
            
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
    parser.add_argument('--hh-min', type=float, help='Minimum human-human agreement')
    parser.add_argument('--random-state', type=int, default=42, help='Random state for reproducibility')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with limited items')
    
    # Generate predictions command
    gen_parser = subparsers.add_parser('gen', help='Generate predictions')
    gen_parser.add_argument('--overwrite', action='store_true', help='Overwrite existing predictions')
    gen_parser.add_argument('--model-dir', default='models', help='Directory containing downloaded models')
    gen_parser.add_argument('--model-id', default='dan-bw', help='Path to model')
    gen_parser.add_argument('--extract-state-indices', type=str, default='', help='Comma separated list of exl2 layers to extract')
    gen_parser.add_argument('--items', type=str, default='', help='Comma separated list of items')
    gen_parser.add_argument('--item-filter', type=str, default='n % 2 != 0', help='Expression to filter item numbers (e.g., "n % 2 != 0" for odd items)')
    
    # Scan checkpoints command
    scan_parser = subparsers.add_parser('scan', help='Scan checkpoints')
    scan_parser.add_argument('--model-dir', default='output', help='Directory containing model checkpoints')
    scan_parser.add_argument('--chk-min', type=int, default=0, help='Minimum checkpoint number')
    scan_parser.add_argument('--chk-max', type=int, default=10**8, help='Maximum checkpoint number')
    scan_parser.add_argument('--chk-list', type=str, default='', help='Comma separated list of checkpoint numbers to evaluate')
    scan_parser.add_argument('--K', type=int, default=5, help='Number of folds for cross-validation')
    scan_parser.add_argument('--items', type=str, default='', help='Comma separated list of items')
    scan_parser.add_argument('--item-filter', type=str, default='n % 2 == 0', help='Expression to filter item numbers (e.g., "n % 2 == 0" for even items)')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if args.command == 'gen':
        generate_predictions(args)
    elif args.command == 'scan':
        scan_checkpoints(args)
    else:
        print("Please specify either 'gen' or 'scan' command")
        print("Example: python siamese_test.py gen --overwrite")
        print("      or python siamese_test.py scan --chk-list 2250,2300,2350")
        sys.exit(1)
