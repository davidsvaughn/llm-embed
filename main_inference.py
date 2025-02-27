import os,sys
import torch
import os, sys
from glob import glob
import time
import traceback

from util import clear_cuda_tensors, mkdirs, tricky_traversal_order
from ddp_utils import init_distributed, is_main, printmain
from data_utils import get_config, load_items
from xgb_utils import run_xgb_on_item, run_xgb_on_items

# from model_utils import load_model

#---------------------------------------------------------------------------------------
'''
For DDP mode - Run this script with the following command:

    torchrun --nproc_per_node 4 ddp_inference.py

'''
#---------------------------------------------------------------------------------------    

# save predictions for all items - useful for building "hard pairs" dataset for siamese model
def save_predictions(overwrite=False, debug=False):
    root_data_dir = '/home/azureuser/embed/data'
    root_prompt_dir = '/mnt/llm-train/embed/simple/prompts'
    
    #-----------------------------------------------------------------------------
    # Get config for each item type
    cfg = get_config('bw',
                     root_data_dir=root_data_dir,
                     root_prompt_dir=root_prompt_dir,
                     filter_out_mult=2,
                     model_id='dan-bw',
                    #  model_id='llama-siam-3-exl2-q4',
                     extract_state_indices=[41, 48, 57],
                    #  K=0,
                     )
    
    # cfg = get_config('fw', trait='dev', # dev , org , con
    #                  root_data_dir=root_data_dir,
    #                  root_prompt_dir=root_prompt_dir,
    #                  filter_out_mult=2,
    #                  model_id='dan-bw', 
    #                  )
    
    # cfg = get_config('math',
    #                  root_data_dir=root_data_dir,
    #                  root_prompt_dir=root_prompt_dir,
    #                  filter_out_mult=2,
    #                  model_id='dan-siam-3',
    #                  )
    #-----------------------------------------------------------------------------

    # load items (debug mode only loads a subset of items)
    data_by_item = load_items(cfg, debug=debug)

    #-----------------------------------------------------------------------------

    # load HF model and tokenizer
    # model = load_model(cfg)
    embedder = EmbedderFactory(cfg)
    
    #-----------------------------------------------------------------------------

    for i, (item, records) in enumerate(data_by_item.items()):
        printmain(f"\nProcessing item: {item} ({i+1}/{len(data_by_item)})")

        # item_path = cfg.item_path_fmt.format(item=item)
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
def scan_checkpoints(cfg, output_dir, min_num=0, max_num=10000000, K=5, filters=None, pooling_strategy='mean'):

    #------------------------------------------------------------------------------
    # min/max checkpoint numbers
    # min_num, max_num = 2500, 3000
    #------------------------------------------------------------------------------
    # output_dir = '/home/azureuser/embed/output2'
    filters = [2750]
    #------------------------------------------------------------------------------

    # get checkpoint directories
    checkpoint_dirs = glob(output_dir + '/checkpoint-*')
    checkpoint_dirs = [d for d in checkpoint_dirs if int(d.split('-')[-1]) >= min_num and int(d.split('-')[-1]) <= max_num]
    checkpoint_dirs.sort(key=lambda x: int(x.split('-')[-1]))

    if filters is not None:
        checkpoint_nums = [int(d.split('-')[-1]) for d in checkpoint_dirs]
        checkpoint_dirs = [checkpoint_dirs[checkpoint_nums.index(n)] for n in filters]
    else:
        checkpoint_dirs = tricky_traversal_order(checkpoint_dirs)

    # pop first two checkpoints and add to end
    if len(checkpoint_dirs) > 10:
        checkpoint_dirs = checkpoint_dirs[2:] + checkpoint_dirs[:2]

    # resume from checkpoint 13...
    # checkpoint_dirs = checkpoint_dirs[13:]
    #------------------------------------------------------------------------------

    # load test items
    data_by_item = load_items(cfg)#, debug=15)

    # loop through checkpoints
    results = []
    for checkpoint_dir in checkpoint_dirs:
        printmain(f"{checkpoint_dirs}")
        printmain(f"\nRunning checkpoint: {checkpoint_dir}")
        start = time.time()
        
        # model = load_model(model_id=checkpoint_dir)
        embedder = EmbedderFactory(model_id=checkpoint_dir)
        
        qwks = run_xgb_on_items(embedder, data_by_item, K=K,
                                pooling_strategy=pooling_strategy,
                                random_state=cfg.get('random_state', 42),
                                )
        
        if qwks is not None:
            qwk = qwks.mean()
            chkpt = checkpoint_dir.split('/')[-1]
            result = f'{chkpt}\t{qwk:.4f}'
            results.append(result)
            
            # sort results by checkpoint number
            results.sort(key=lambda x: int(x.split('\t')[0].split('-')[-1]))
            
            print('\n' + '-'*40)
            print(f'{output_dir}')
            for r in results:
                print(r)
            print('-'*40 + '\n')
            print(f"\nTIME FOR ITEM: {time.time()-start:.2f} SEC\n")
            
        if torch.distributed.is_initialized():
            torch.distributed.barrier()  # Wait for rank 0 to finish
            
        clear_cuda_tensors()

    if is_main():
        print("Done")

def run_checkpoints():
    
    root_data_dir = '/home/azureuser/embed/data'
    root_prompt_dir = '/mnt/llm-train/embed/simple/prompts'
    
    #-----------------------------------------------------------------------------
    # Get config for each item type
    cfg = get_config('math',
                     random_state=42,
                     root_data_dir=root_data_dir,
                     root_prompt_dir=root_prompt_dir,
                    #  items = [123362, 33082, 13272, 27218, 29632, 31600, 52414, 78382],
                     filter_in_mult=2, hh_min=0.6,
                     )
    
    # cfg = get_config('bw',
    #                  root_data_dir=root_data_dir,
    #                  root_prompt_dir=root_prompt_dir,
    #                  items = [58151, 58155, 58425, 58587, 58937, 59181, 61047, 90881, 94225, 94237],
    #                  )
    
    # cfg = get_config('fw', trait='dev', # dev , org , con
    #                  root_data_dir=root_data_dir,
    #                  root_prompt_dir=root_prompt_dir,
    #                  )
    
    #-----------------------------------------------------------------------------
    
    # each output_dir should contain multiple checkpoint directories
    output_dirs = [
        # '/home/azureuser/embed/output',
        '/home/azureuser/embed/output1',
        # '/home/azureuser/embed/output5',
        ]

    for output_dir in output_dirs:
        scan_checkpoints(cfg, output_dir,
                         pooling_strategy='mean',
                        #  pooling_strategy='last',
                        K=0,
                         )
        
#------------------------------------------------------------------------------
from sentence_transformers import SentenceTransformer, models
from embed_utils import EmbedderFactory, HuggingfaceEmbedder

# output_dir contains multiple checkpoint directories
def test_st_checkpoint():
    
    root_data_dir = '/home/azureuser/embed/data'
    root_prompt_dir = '/mnt/llm-train/embed/simple/prompts'
    
    # Get config for math
    cfg = get_config('math',
                     batch_size=16,
                     root_data_dir=root_data_dir,
                     root_prompt_dir=root_prompt_dir,
                     items = [123362, 33082, 13272, 27218, 29632, 31600, 52414, 78382]
                     )
    # load test items
    data_by_item = load_items(cfg)
    
    #------------------------------------------------------------------------------
    # load SentenceTransformer model
    st_root = "/home/azureuser/embed/st/output/"
    
    
    #------------------------------------------------------------------------------
    # st_run = 'training_math_pairs_279_Salesforce-SFR-Embedding-Mistral_2025-02-13_07-53-43'
    # chkpt_num = 5000 # 2500  3200  5000
    # model_path = st_root + st_run + f'/checkpoint-{chkpt_num}'
    # embedder = EmbedderFactory(model_id=model_path, model_type='st')
    #------------------------------------------------------------------------------
    
    st_run = 'training_math_pairs_279_meta-llama-Llama-3.2-3B-Instruct_2025-02-13_19-44-43'
    chkpt_num = 3200
    model_path = st_root + st_run + f'/checkpoint-{chkpt_num}'
    # embedder = EmbedderFactory(model_id=model_path, model_type='st')
    
    st_model = SentenceTransformer(model_path)
    transformer_module = st_model[0]
    hf_model = transformer_module.auto_model
    hf_tokenizer = transformer_module.tokenizer
    embedder = HuggingfaceEmbedder(model=hf_model, 
                                   tokenizer=hf_tokenizer, 
                                   pooling_strategy="mean", 
                                #    padding_side="right",
                                   )
    
    # TODO: pooling_strategy --> pooling_mode
    
    #------------------------------------------------------------------------------
    
    qwks = run_xgb_on_items(embedder, data_by_item)
    
    if qwks is not None:
        qwk = qwks.mean()
        print(f'QWK: {qwk:.4f} {qwks}')
        
    print("Done")

#------------------------------------------------------------------------------
from embed_utils import SentenceTransformerEmbedder
from model_utils import load_checkpoint_model

def clm2st(clm_model_id,
           pooling_mode="mean", # mean, lasttoken
           ):
    word_embedding_model = models.Transformer(
        clm_model_id, 
    )
    # pooling_mode: mean, lasttoken, cls, max, mean_sqrt_len_tokens, weightedmean
    # see: /home/azureuser/embed/sentence-transformers/sentence_transformers/models/Pooling.py
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode=pooling_mode,
    )
    # build the SentenceTransformer pipeline
    st_model = SentenceTransformer(
        modules=[word_embedding_model, pooling_model]
    )
    # set the st_model's tokenizer's pad token to eos token
    st_model.tokenizer.pad_token = st_model.tokenizer.eos_token
    return st_model


def compare_st_checkpoint():
    
    root_data_dir = '/home/azureuser/embed/data'
    root_prompt_dir = '/mnt/llm-train/embed/simple/prompts'
    
    # Get config for math
    cfg = get_config('math',
                     batch_size=16,
                     root_data_dir=root_data_dir,
                     root_prompt_dir=root_prompt_dir,
                     items = [123362],
                    #  items = [123362, 33082, 13272, 27218, 29632, 31600, 52414, 78382]
                     )
    # load test items
    data_by_item = load_items(cfg)
    
    
    #------------------------------------------------------------------------------
    # python merge_checkpoint.py --checkpoint_dir /home/azureuser/embed/output6/checkpoint-1700
    model_path = '/home/azureuser/embed/output6/model'
    
    # location of HF adapter model
    # hf_root = '/home/azureuser/embed/output6'
    # chkpt_num = 1700
    # model_path = hf_root + f'/checkpoint-{chkpt_num}'
    
    # 1 - load as HF model
    embedder = EmbedderFactory(model_id=model_path, model_type='hf')
    
    
    # 2 - load as SentenceTransformer model
    # hf_model = load_checkpoint_model(model_path)
    
    # st_model = clm2st(model_path, pooling_mode="mean")
    # embedder = SentenceTransformerEmbedder(model=st_model)
    
    #------------------------------------------------------------------------------
    # load SentenceTransformer model
    # st_root = "/home/azureuser/embed/st/output/"
    
    
    # #------------------------------------------------------------------------------
    # # st_run = 'training_math_pairs_279_Salesforce-SFR-Embedding-Mistral_2025-02-13_07-53-43'
    # # chkpt_num = 5000 # 2500  3200  5000
    # # model_path = st_root + st_run + f'/checkpoint-{chkpt_num}'
    # # embedder = EmbedderFactory(model_id=model_path, model_type='st')
    # #------------------------------------------------------------------------------
    
    # st_run = 'training_math_pairs_279_meta-llama-Llama-3.2-3B-Instruct_2025-02-13_19-44-43'
    # chkpt_num = 3200
    # model_path = st_root + st_run + f'/checkpoint-{chkpt_num}'
    # # embedder = EmbedderFactory(model_id=model_path, model_type='st')
    
    # st_model = SentenceTransformer(model_path)
    # transformer_module = st_model[0]
    # hf_model = transformer_module.auto_model
    # hf_tokenizer = transformer_module.tokenizer
    # embedder = HuggingfaceEmbedder(model=hf_model, 
    #                                tokenizer=hf_tokenizer, 
    #                                pooling_strategy="mean", 
    #                             #    padding_side="right",)
    
    # TODO: pooling_strategy --> pooling_mode
    
    #------------------------------------------------------------------------------
    
    qwks = run_xgb_on_items(embedder, data_by_item, 
                            padding_side="left",
                            pooling_strategy="mean")
    
    if qwks is not None:
        qwk = qwks.mean()
        print(f'QWK: {qwk:.4f} {qwks}')
        
    print("Done")

#------------------------------------------------------------------------------------------
if __name__ == "__main__":
    #------------------------------------------------------------------------------
    ''' save predictions for all items - to be used for building "hard pairs" dataset for siamese model '''
    # save_predictions(overwrite=True, debug=False)
    
    #------------------------------------------------------------------------------\
    ''' run xgb on items for each checkpoint '''
    run_checkpoints()
    
    #------------------------------------------------------------------------------\
    ''' test xgb on SentenceTransformer checkpoint '''
    # test_st_checkpoint()
    
    # compare_st_checkpoint()
    