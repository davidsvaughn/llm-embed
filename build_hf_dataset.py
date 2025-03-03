# Description: Build a Hugging Face dataset from the MI data
# Upload the dataset to the Hugging Face Hub (private repo) to use for fine-tuning

import os, sys
import traceback, re
import numpy as np
import json
from glob import glob
from datasets import Dataset, DatasetDict
import random
from utils import to_adict
from data_utils import get_config, load_items


#---------------------------------------------------------------------------------------------

# function to transform list of dicts to dict of lists
# *** all dicts in list should have the same keys! ***
def dictlist_to_listdict(dictlist):
    listdict = {}
    # get keys that are common to all dicts in dictlist
    keys = set(dictlist[0].keys())
    for d in dictlist:
        keys = keys.intersection(set(d.keys()))
    
    for key in keys:
        # if key starts with 'payload', convert to json string
        if key.startswith('payload'):
            listdict[key] = [json.dumps(d[key]) for d in dictlist]
        else:
            listdict[key] = [d[key] for d in dictlist]
    return listdict

# function to transform list of dicts to dataset
def dictlist_to_dataset(dictlist):
    return Dataset.from_dict(dictlist_to_listdict(dictlist))

# function to remove a key from a dict if it exists
def try_del(d, key):
    try:
        del d[key]
    except:
        pass

#---------------------------------------------------------------------------------------------

def make_pairs(data, ct_max=2, np_min=10, easy_prob=0.5, easy_max=1000, blacklist=[], cutoff=0.4):
    n = len(data)
    perm = np.random.permutation(n*(n+1))

    # select pairs
    ct_index = {}
    pos_pairs, neg_pairs = [], []
    num_hard = 0

    z = 0
    # for i,j in idx_pairs:
    for kidx, k in enumerate(perm):
        i,j = k//n, k%n
        if i>=j: continue
        if i>=n or j>=n:continue

        if i not in ct_index: ct_index[i] = 0
        if ct_index[i] >= ct_max: continue
        if j not in ct_index: ct_index[j] = 0
        if ct_index[j] >= ct_max: continue

        rec1, rec2 = data[i], data[j]
        
        if (rec1.score, rec2.score) in blacklist or (rec2.score, rec1.score) in blacklist:
            continue

        # is it a hard pair?
        # hardness of pair
        hardness = abs(rec1.pred - rec1.score) + abs(rec2.pred - rec2.score)
        
        is_hard = rec1.pred != rec1.score or rec2.pred != rec2.score
        if not is_hard:
            if np.random.rand() > easy_prob:
                continue

        # is it a positive pair?
        is_pos = rec1.score == rec2.score

        # # get neg/pos ratio
        np_ratio = (len(neg_pairs)+1) / (len(pos_pairs)+1)
        if is_pos and np_ratio<np_min:
            continue
        
        pair = {'h': hardness, 'rec1': rec1, 'rec2': rec2}
        
        if is_pos:
            pos_pairs.append(pair)
        else:
            neg_pairs.append(pair)

        ct_index[i] += 1
        ct_index[j] += 1
        last_kidx = kidx
        num_hard += 1 if is_hard else 0
        z += 1
        if z/(ct_max*n) > cutoff:
            break
        
    #---------------------------------------------------------------------------------------------
    
    # count number of items in ct_index where ct_index[i] >0
    num_items = len([i for i in ct_index if ct_index[i]>0])
    # print percetage of items that have been covered
    print(f"Samples Covered: {num_items/n:.2f}")
    
    # count number of items in ct_index where ct_index[i] == 1,2,3,4,etc...
    ct_counts = {}
    for i in ct_index:
        ct = ct_index[i]
        if ct not in ct_counts:
            ct_counts[ct] = 0
        ct_counts[ct] += 1
    print(f"Counts: {ct_counts}")

    # # print out the number of positive and negative pairs
    print(f"Num Pos Pairs: {len(pos_pairs)}\tNum Neg Pairs: {len(neg_pairs)}")
    print(f"Total: {len(pos_pairs) + len(neg_pairs)}\tN/P={np_ratio:.2f}\t( N={len(neg_pairs)}\tP={len(pos_pairs)} )")

    # # print ratio of last_kidx to perm size
    print(f"Last kidx: {last_kidx}\tPerm size: {len(perm)}\tCovered: {last_kidx/len(perm):.2f}")

    # # print out the fraction of hard pairs
    print(f"Hard pairs: {num_hard}\tFraction: {num_hard/(len(pos_pairs)+len(neg_pairs)):.2f}")
    # print()
    
    # sort by hardness descending
    # pos_pairs = sorted(pos_pairs, key=lambda x: -x['h'])
    # neg_pairs = sorted(neg_pairs, key=lambda x: -x['h'])

    all_pairs = pos_pairs + neg_pairs
    random.shuffle(all_pairs)
    all_pairs = sorted(all_pairs, key=lambda x: -x['h'])
    
    easy_max = min(easy_max, num_hard)
    num_easy = 0

    # make pairs
    pairs = []
    for pp in all_pairs:
        rec1, rec2, h = pp['rec1'], pp['rec2'], pp['h']
        if h==0:
            num_easy += 1
            if num_easy > easy_max:
                break
    
        pair = rec1.copy()
        
        # raw student responses
        pair['text1'] = rec1['text']
        pair['text2'] = rec2['text']
        
        # responses with prompt template, etc.
        pair['payload1'] = rec1['payload']
        pair['payload2'] = rec2['payload']

        pair['score1'] = rec1['score']
        pair['score2'] = rec2['score']
        
        pair['index1'] = rec1['index']
        pair['index2'] = rec2['index']
        
        # pair['diff'] = abs(rec1['score'] - rec2['score'])
        pair['diff_abs'] = abs(rec1['score'] - rec2['score'])
        pair['diff_norm'] = abs(rec1['score_norm'] - rec2['score_norm'])
        
        try_del(pair, 'payload')
        try_del(pair, 'text')
        try_del(pair, 'index')
        try_del(pair, 'sc')
        try_del(pair, 'score')
        try_del(pair, 'score_norm')
        try_del(pair, 'pred')
        try_del(pair, 'split')
        pairs.append(pair)
        
    return pairs

#---------------------------------------------------------------------------------------------

def get_pairs(hf_cfg, dataset_cfg, debug=False):
        
    data_by_item = load_items(dataset_cfg, debug=debug)

    item_list = list(data_by_item.keys())
    random.shuffle(item_list)
    split = int(len(item_list) * hf_cfg.SPLIT_VALID)
    
    valid_items = item_list[:split]
    train_items = item_list[split:]
    train_data, valid_data = [], []
    
    #---------------------------------------------------------------------------------------------
    for n,item in enumerate(train_items):
        print('-'*100, f"\nItem {item}\t({n+1}/{len(train_items)})")
        
        print(f"\nTrain pairs ----------------------------------------------")
        train_data += make_pairs(data_by_item[item], 
                                 ct_max=hf_cfg.CT_MAX_TRAIN, 
                                 np_min=hf_cfg.NP_MIN_TRAIN, 
                                 easy_prob=hf_cfg.EASY_PROB_TRAIN,
                                 easy_max=hf_cfg.EASY_MAX_TRAIN,
                                 blacklist=hf_cfg.BLACKLIST)
        
    #---------------------------------------------------------------------------------------------
    for n,item in enumerate(valid_items):
        print('-'*100, f"\nItem {item}\t({n+1}/{len(valid_items)})\n")
        
        print(f"\nValid pairs ----------------------------------------------")
        valid_pairs = make_pairs(data_by_item[item], 
                                 ct_max=hf_cfg.CT_MAX_VALID, 
                                 np_min=hf_cfg.NP_MIN_VALID, 
                                 easy_prob=hf_cfg.EASY_PROB_VALID,
                                 easy_max=hf_cfg.EASY_MAX_VALID,
                                 blacklist=hf_cfg.BLACKLIST)
        valid_data += valid_pairs
        
    #---------------------------------------------------------------------------------------------

    print(f"\t{dataset_cfg.str}\n\tNum Valid Pairs: {len(valid_data)}\n\tNum Train Pairs: {len(train_data)}")

    return train_data, valid_data

#---------------------------------------------------------------------------------------------

def make_hf_dataset():
    
    # general parameters for building the dataset
    hf_cfg = to_adict({
        
        'FILTER_OUT_MULT': 2,
        'CT_MAX_TRAIN': 3,
        'CT_MAX_VALID': 1,
        'NP_MIN_TRAIN': 2,
        'NP_MIN_VALID': 2,
        'EASY_MAX_TRAIN': 1000,
        'EASY_MAX_VALID': 0,
        'EASY_PROB_TRAIN': 0.01,
        'EASY_PROB_VALID': 0.01,
        'SPLIT_VALID': 0.05,
        'BLACKLIST': [(0,0)],
        
        'RAND_SEED': random.randint(1000, 10000),
        # 'RAND_SEED': 932,
        
        # 'SPLIT_TEST': 0.1,
    })
    print(f"RAND_SEED: {hf_cfg.RAND_SEED}")
    random.seed(hf_cfg.RAND_SEED)
    np.random.seed(hf_cfg.RAND_SEED)
    
    root_data_dir = '/home/azureuser/embed/data'
    root_prompt_dir = '/home/azureuser/llm-embed/prompts'

    #---------------------------------------------------------------------------------------------
    dataset_cfgs = []
    #-------------------------------------------------------
    # cfg = get_config('math',
    #                  root_data_dir=root_data_dir,
    #                  root_prompt_dir=root_prompt_dir,
    #                  hh_min=0.64,
    #                  filter_out_mult=hf_cfg.FILTER_OUT_MULT,
    #                  model_id='dan-siam-3', # for predictions
    #                  )
    # dataset_cfgs.append(cfg)
    #-------------------------------------------------------
    cfg = get_config('bw',
                     root_data_dir=root_data_dir,
                     root_prompt_dir=root_prompt_dir,
                     hh_min=0.54,
                     filter_out_mult=hf_cfg.FILTER_OUT_MULT,
                     model_id='dan-bw',
                     )
    dataset_cfgs.append(cfg)
    # #-------------------------------------------------------
    # cfg = get_config('fw',
    #                  root_data_dir=root_data_dir,
    #                  root_prompt_dir=root_prompt_dir,
    #                  trait='con', hh_min=0.6,
    #                  filter_out_mult=hf_cfg.FILTER_OUT_MULT,
    #                  model_id='dan-bw',
    #                  )
    # dataset_cfgs.append(cfg)
    # #-------------------------------------------------------
    # cfg = get_config('fw',
    #                  root_data_dir=root_data_dir,
    #                  root_prompt_dir=root_prompt_dir,
    #                  trait='org', # hh_min=0.75,
    #                  filter_out_mult=hf_cfg.FILTER_OUT_MULT,
    #                  model_id='dan-bw',
    #                  )
    # dataset_cfgs.append(cfg)
    # #-------------------------------------------------------
    # cfg = get_config('fw',
    #                  root_data_dir=root_data_dir,
    #                  root_prompt_dir=root_prompt_dir,
    #                  trait='dev', # hh_min=0.75,
    #                  filter_out_mult=hf_cfg.FILTER_OUT_MULT,
    #                  model_id='dan-bw',
    #                  )
    # dataset_cfgs.append(cfg)
    #---------------------------------------------------------------------------------------------
    
    train_data, valid_data = [], []
    num_train_pairs, num_valid_pairs = {},{}
    
    name = None
    for dataset_cfg in dataset_cfgs:
        #---------------------------------
        if name is None:
            name = dataset_cfg.str
        else:
            if dataset_cfg.str not in name:
                name += f"_{dataset_cfg.str}"
        #---------------------------------
                
        t_data, v_data = get_pairs(hf_cfg, dataset_cfg)
        train_data += t_data
        valid_data += v_data
        
        num_train_pairs[dataset_cfg.str] = len(t_data)
        num_valid_pairs[dataset_cfg.str] = len(v_data)

    print(f"\nTotal Valid Pairs: {len(valid_data)}\nTotal Train Pairs: {len(train_data)}")
    #---------------------------------------------------------------------------------------------

    # print out the number of pairs for each config
    for dataset_cfg in dataset_cfgs:
        print(f"{dataset_cfg.str}\tTrain: {num_train_pairs[dataset_cfg.str]}\tValid: {num_valid_pairs[dataset_cfg.str]}")

    # shuffle
    random.shuffle(train_data)
    random.shuffle(valid_data)
    
    # subsample (debugging)
    # train_data = train_data[:100000]
    # valid_data = valid_data[:len(valid_data)//2]

    # make hf dataset
    valid_dataset = dictlist_to_dataset(valid_data)
    train_dataset = dictlist_to_dataset(train_data)
    hf_dataset = DatasetDict({'train': train_dataset, 'validation': valid_dataset})

    # save to hub
    hf_dataset.push_to_hub(f"davidsvaughn/{name}_pairs_{hf_cfg.RAND_SEED}", private=True)
    # hf_dataset.push_to_hub(f"davidsvaughn/mixed_pairs_{hf_cfg.RAND_SEED}", private=True)

    print('Done.')
    
#---------------------------------------------------------------------------------------------
if __name__ == "__main__":
    make_hf_dataset()