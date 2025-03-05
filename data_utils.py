import os, sys
import re
import pandas as pd
from glob import glob
from tqdm import tqdm
from utils import mkdirs, adict, to_adict, read_jsonl, write_jsonl, pretty_print
from ddp_utils import printmain, is_main
from model_utils import PromptBuilder

#-------------------------------------------------------------------------------------
# process common fields
def process_sample(rec):
    rec['item'] = int(rec['item'])
    rec['index'] = int(rec['index'])
    rec['grade'] = int(rec['grade'])
    
#-------------------------------------------------------------------------------------

'''
BW sample processing...

FIELD notes:
    mode: narrative | argumentative | explanatory
    section: elaboration | introduction | opening | conclusion
        
BEFORE:
{
  "index": 355,
  "grade": 11,
  "mode": "Tar 1a Elaboration",
  "item": "100728",
  "sqid": "vdb26.SBAC_Spring22-88578",
  "sc": "0",
  "text": "So i hope that with this in mind, we will beable to countiune our activites. however with this brought up i am well aware that before activites start or are put out, sections for out training will be marked so no more inncedents may acoure. If i have casued any farther harm, i apoligize for my detiction to my team and its supporters. "
}

AFTER:
{
  "index": 355,
  "grade": 11,
  "mode": "narrative",
  "item": 100728,
  "text": "So i hope that with this in mind, we will beable to countiune our activites. however with this brought up i am well aware that before activites start or are put out, sections for out training will be marked so no more inncedents may acoure. If i have casued any farther harm, i apoligize for my detiction to my team and its supporters. ",
  "section": "elaboration",
  "score": 0,
  "min_score": 0,
  "max_score": 2,
  "score_norm": 0.0
}
''' 

bw_code2mode = { '1a': 'narrative', '3a': 'explanatory', '6a': 'argumentative' }
def parse_bw_mode(mode):
    try:
        _, code, section = mode.lower().split()
        mode = bw_code2mode[code]
        return mode, section
    except Exception as e:
        print(f"Error parsing mode: {mode}\n{e}")
        raise e
 
def process_bw_sample(rec):
    process_sample(rec)
    rec['mode'], rec['section'] = parse_bw_mode(rec['mode'])
    rec['score'] = int(rec['sc'])
    del rec['sc']
    del rec['sqid']
    return to_adict(rec)
    
#-------------------------------------------------------------------------------------
'''
FW sample processing...

    FIELD notes:
        mode: narrative | argumentative | explanatory
        trait: dev | con | org
        
BEFORE:
{
  "index": 0,
  "grade": 3,
  "mode": "Narrative",
  "item": "101169",
  "sqid": "vdb26.SBAC_Spring22-3190720",
  "con": "0",
  "dev": "1",
  "org": "1",
  "text": "\n\tone day i saw something flying in the Trees it was a squirrel it was flying but it diddit have wings. and it was useing it tail to steer its body and it was juming Tree to tree\n"
}

AFTER:
{
  "index": 0,
  "grade": 3,
  "mode": "narrative",
  "item": 101169,
  "con": { "score": 0, "min_score": 0, "max_score": 2, "score_norm": 0.0 },
  "dev": { "score": 1, "min_score": 1, "max_score": 4, "score_norm": 0.0 },
  "org": { "score": 1, "min_score": 1, "max_score": 4, "score_norm": 0.0 },
  "text": "\n\tone day i saw something flying in the Trees it was a squirrel it was flying but it diddit have wings. and it was useing it tail to steer its body and it was juming Tree to tree\n"
}

'''  
def process_fw_sample(rec):
    process_sample(rec)
    rec['mode'] = rec['mode'].lower()
    for trait in ['dev', 'con', 'org']:
        rec[trait] = { 'score' : int(rec[trait]) }
    del rec['sqid']
    return to_adict(rec)

#-------------------------------------------------------------------------------------
'''
MATH sample processing...
BEFORE:
{
  "index": 0,
  "grade": 5,
  "mode": "Math 3-5",
  "item": "107069",
  "sqid": "vdb26.SBAC_Spring22-3036370",
  "sc": "0",
  "text": "Plants grow in the grasslands because there made in Africa.  Second reason is plants grow in the grasslands because grasslands are one of the world's biomes."
}

AFTER:
{
  "index": 0,
  "grade": 5,
  "mode": "math 3-5",
  "item": 107069,
  "text": "Plants grow in the grasslands because there made in Africa.          Second reason is plants grow in the grasslands because grasslands are one of the world's biomes.",
  "score": 0,
  "min_score": 0,
  "max_score": 2,
  "score_norm": 0.0
}
''' 

def process_math_sample(rec):
    process_sample(rec)
    rec['mode'] = rec['mode'].lower()
    rec['score'] = int(rec['sc'])
    del rec['sc']
    del rec['sqid']
    return to_adict(rec)

#-------------------------------------------------------------------------------------

# compute min and max scores over all records
def get_min_max_scores(records):
    if 'score' in records[0]:
        min_score = min([rec['score'] for rec in records])
        max_score = max([rec['score'] for rec in records])
        return (min_score, max_score)
    else: # fw - has 3 traits
        min_max_scores = {}
        for trait in ['dev', 'con', 'org']:
            min_score = min([rec[trait]['score'] for rec in records])
            max_score = max([rec[trait]['score'] for rec in records])
            min_max_scores[trait] = (min_score, max_score)
        return min_max_scores
            
# set min and max scores for a record, and set normalized score ( to [0,1] interval )
def set_min_max_scores(rec, min_max_scores, norm=True):
    if 'score' in rec:
        rec['min_score'], rec['max_score'] = min_max_scores
        if norm:
            rec['score_norm'] = (rec['score'] - rec['min_score']) / (rec['max_score'] - rec['min_score'])
    else: # fw - has 3 traits
        for trait in ['dev', 'con', 'org']:
            rec[trait]['min_score'], rec[trait]['max_score'] = min_max_scores[trait]
            rec[trait]['score_norm'] = (rec[trait]['score'] - rec[trait]['min_score']) / (rec[trait]['max_score'] - rec[trait]['min_score'])

#-------------------------------------------------------------------------------------
process_funcs = { 'bw': process_bw_sample, 'fw': process_fw_sample, 'math': process_math_sample }

def process_raw_data(item_type, root_data_dir, items_sub='items'):
    # item_type = 'fw' # bw | fw | math
    
    src_dir = f'{root_data_dir}/{item_type}'
    data_dir = f'{src_dir}/{items_sub}'

    src_train_file = f'{src_dir}/{item_type}_training.json'
    src_valid_file = f'{src_dir}/{item_type}_validation.json'

    min_max_data = {}
    process_func = process_funcs[item_type]
    
    for src_file, dst in zip([src_train_file, src_valid_file], ['train', 'valid']):
        print(f"Processing: {src_file}...")
        data_by_item = {}
        
        records = read_jsonl(src_file)
        for rec in records:
            rec = process_func(rec)
            item = rec['item']
            if item not in data_by_item:
                data_by_item[item] = []
            data_by_item[item].append(rec)
                    
        for item, records in data_by_item.items():
            # compute min and max scores for each item
            if item in min_max_data:
                min_max_scores = min_max_data[item]
            else:
                min_max_data[item] = min_max_scores = get_min_max_scores(records)
                
            # add min and max scores to each record
            for rec in records:
                set_min_max_scores(rec, min_max_scores)
            
            # save records to item directory    
            item_path = f'{data_dir}/{item}'
            mkdirs(item_path)
            write_jsonl(f'{item_path}/{dst}.jsonl', records)

#-------------------------------------------------------------------------------------
def expand_user(path):
    return os.path.expanduser(path) if '~' in path else path

def run_filter(N, filter_expr):
    """
    Filter a list of integers based on a Python expression.
    
    Args:
        N: List of integers to filter
        filter_expr: String containing a Python expression where 'n' represents each number
                    Examples: "n % 2 == 0", "n > 10", "n % 3 == 0 and n % 2 == 1"
    
    Returns:
        List of integers that satisfy the filter expression
    """
    filtered_N = []
    for n in N:
        try:
            # Use a restricted globals dict for security
            if eval(filter_expr, {"__builtins__": {}}, {"n": n}):
                filtered_N.append(n)
        except Exception as e:
            print(f"Error evaluating expression for n={n}: {e}")
            sys.exit(1)
    return filtered_N

def get_base_config(item_type,
                    data_dir='data',
                    # model_dir='models',
                    # prompt_dir='prompts',
                    batch_size=16,
                    max_length=8192,
                    padding_side='left',
                    hh_min=None, 
                    trait=None, 
                    **kwargs):
    cfg = adict()
    cfg.item_type = item_type
    
    # if '~' in data_dir, expand to user's home directory
    data_dir = expand_user(data_dir)
    
    # data_dir
    src_dir = f'{data_dir}/{item_type}'
    cfg.data_dir = f"{src_dir}/{kwargs.get('items_sub', 'items')}"

    # hh file
    hh_file = f'{src_dir}/hh.csv'
    df = pd.read_csv(hh_file) if os.path.exists(hh_file) else None

    if trait:
        cfg.str = f'{item_type}-{trait}'
        cfg.trait = trait
        if df is not None:
            df = df[df['trait'] == trait ]
    else:
        cfg.str = item_type
    
    if df is not None:
        cfg.item_to_hh = dict(zip(df['item'], df['qwk_hh']))
    
    cfg.batch_size = batch_size
    cfg.max_length = max_length
    cfg.padding_side = padding_side

    # add remaining kwargs to cfg as attributes... like model_id
    for k,v in kwargs.items():
        setattr(cfg, k, v)
        
    # expand all '~' in attribs ending with '_dir'
    for k in cfg.keys():
        if k.endswith('_dir'):
            setattr(cfg, k, expand_user(getattr(cfg, k)))
        
    if 'items' not in cfg:
        cfg.hh_min = hh_min
        items = []
        for item_file in glob(f"{cfg.data_dir}/*/train.jsonl"):
            item = int(re.search(r'(\d+)', item_file).group(1))
            items.append(item)
        items.sort()
        cfg['items'] = items   

    return cfg

def get_math_config(hh_min=0.7, **kwargs):
    cfg = get_base_config('math', hh_min=hh_min, **kwargs)
    return cfg

def get_bw_config(hh_min=0.58, **kwargs):
    cfg = get_base_config('bw', hh_min=hh_min, **kwargs)
    return cfg

def get_fw_config(trait, hh_min=0.75, **kwargs):
    cfg = get_base_config('fw', hh_min=hh_min, trait=trait, **kwargs)
    cfg.max_length = 8192
    return cfg


def get_config(item_type, trait=None, **kwargs):
    if item_type == 'math':
        cfg = get_math_config(**kwargs)
    elif item_type == 'bw':
        cfg = get_bw_config(**kwargs)
    elif item_type == 'fw':
        cfg = get_fw_config(trait, **kwargs)
    elif item_type.startswith('fw-'):
        item_type, trait = item_type.split('-')
        cfg = get_fw_config(trait, **kwargs)
    else:
        raise ValueError(f"Invalid item_type: {item_type}")
    
    if 'item_filter' in cfg and 'items' in cfg:
        cfg.items = run_filter(cfg.items, cfg.item_filter)
    
    # filter items by hh_min
    if 'hh_min' in cfg and 'items' in cfg and cfg.hh_min is not None:
        cfg.items = [item for item in cfg.items if cfg.item_to_hh[item] >= cfg.hh_min]
        
    if 'item_to_hh' in cfg:
        cfg.item_to_hh = { item: hh for item,hh in cfg.item_to_hh.items() if item in cfg.items }
        
    printmain(cfg)
    
    # load tokenizer
    for id in 'model_id', 'tokenizer_id':
        if id in cfg:
            # if has no '/' assume it's a local model and pre-pend model_dir
            if '/' not in cfg[id] and 'model_dir' in cfg:
                cfg[id] = f'{cfg.model_dir}/{cfg[id]}'
            
    if 'model_id' in cfg and 'tokenize' in cfg and cfg.tokenize and 'tokenizer_id' not in cfg:
        cfg.tokenizer_id = cfg.model_id
    
    # replace pred_model_id with model_id 
    if 'model_id' in cfg:
        # get rightmost part of model_id after '/'
        model_tag = cfg.model_id.split('/')[-1]
        cfg.preds_file_fmt = f'{cfg.data_dir}/{model_tag}/{{item}}/preds.csv'
        if trait:
            cfg.preds_file_fmt = cfg.preds_file_fmt.replace('.csv', f'_{trait}.csv')
        
    #-----------------------------------------------------------
    
    # prompt related fields
    if 'prompt_dir' in cfg:
        cfg.prompt_dir = os.path.join(cfg.prompt_dir, cfg.item_type)
        # truncation used for embedding models : see PromptBuilder
        cfg.truncate_to = 'score'
        
    #-----------------------------------------------------------

    return cfg

#-------------------------------------------------------------------------------------

def load_item_predictions(cfg, item):
    if 'preds_file_fmt' not in cfg:
        return None
    preds_file = cfg.preds_file_fmt.format(item=item)
    preds = None
    if os.path.exists(preds_file):
        preds = {}
        df = pd.read_csv(preds_file, header=None)
        df.columns = ['index', 'score', 'pred']
        for _,row in df.iterrows():
            preds[row['index']] = row['pred']
    return preds

#-------------------------------------------------------------------------------------
    
def load_items(cfg, debug=False):
    
    # truncate item list: if debug is integer, truncate to that number
    if debug:
        n = debug if (isinstance(debug, int) and debug>1) else 10
        cfg.items = cfg.items[:n]
        
    #-----------------------------------------------------------
    # load PromptBuilder - used to render custom prompt templates
    prompt_builder = PromptBuilder(**cfg)
        
    #-----------------------------------------------------------
    
    # data_by_item: maps each item_id to list of records, each record is single student response
    data_by_item = {}
    preds_not_found = 0
    
    for item in tqdm(cfg.items, desc="Loading items", disable=not is_main()):
        
        # load predictions if exist
        preds = load_item_predictions(cfg, item)
        
        item_path = f'{cfg.data_dir}/{item}'
        data_by_item[item] = []
        
        for split in ['train', 'valid']:
            src_file = f'{item_path}/{split}.jsonl'
            if os.path.exists(src_file):
                
                records = read_jsonl(src_file)
                # data_by_item[item].extend(records)
                
                for rec in records:
                    
                    # skip records with too many words
                    if len(rec['text'].split()) > cfg.max_length:
                        # print(f"{item}\tskipping record: len=={len(rec.text.split())}")
                        continue
                    
                    data_by_item[item].append(rec)
                    rec.split = split # set train/valid
                    rec.item_type = cfg.item_type # math, bw, fw
                    
                    # set trait, choose trait score data
                    if 'trait' in cfg:
                        rec.update(rec[cfg.trait]) # score data for trait
                        rec.trait = cfg.trait # store trait name
                    
                    #-----------------------------------------------------------
                    ''' Apply custom prompt template for each item type '''
                    # payload: {'messages' : [user:, asst:, etc.],
                    #           'target'   : (target_name, target_value) }
                    rec.payload = prompt_builder.get_payload(**rec)
                    #-----------------------------------------------------------
                    
                    # set prediction if exists
                    if preds is not None:
                        if rec.index in preds:
                            rec.pred = int(preds[rec.index])
                        else:
                            print(f"\tmissing prediction: item={item} index={rec.index}")
                            preds_not_found += 1
                            rec.pred = rec.score # just fill in with actual score

    printmain(f"\n{len(data_by_item)} items")

    if preds is not None and preds_not_found > 0:
        print(f"Missing predictions for {preds_not_found} records")

    return data_by_item

#-------------------------------------------------------------------------------------
def preprocess_datasets():
    root_data_dir = '/home/azureuser/embed/data'
    # process_raw_data('math', root_data_dir)
    # process_raw_data('bw', root_data_dir)
    process_raw_data('fw', root_data_dir)
    
#-------------------------------------------------------------------------------------
def test_load_items():
    root_data_dir = '/home/azureuser/embed/data'
    root_prompt_dir = '/mnt/llm-train/embed/simple/prompts'
    
    #-------------------------------------------------------------------
    # Get config for each item type
    cfg = get_config('bw',
                     data_dir=root_data_dir,
                     prompt_dir=root_prompt_dir,
                     filter_out_mult=2, # i.e. load only ODD items
                     model_id='dan-bw',
                    #  hh_min=0.58,
                    #  model_id = 'meta-llama/Llama-3.2-3B-Instruct', tokenize=True,
                     )
    
    # cfg = get_config('fw', trait='dev',
    #                  data_dir=root_data_dir,
    #                  prompt_dir=root_prompt_dir,
    #                  filter_out_mult=2, # i.e. load only ODD items
    #                 #  model_id='dan-bw',
    #                  )
    
    # cfg = get_config('math',
    #                  data_dir=root_data_dir,
    #                  prompt_dir=root_prompt_dir,
    #                  filter_out_mult=2, # i.e. load only ODD items
    #                  model_id='dan-siam-3',
    #                  )
    #-------------------------------------------------------------------
    
    # print cfg
    for k in cfg.keys():
        print(f"{k}: {cfg[k]}") 
    
    data_by_item = load_items(cfg, debug=False)
    for item, records in data_by_item.items():
        print(f"\n{'-'*80}\nItem: {item}\n")
        for rec in records:
            # print data with pretty indentation
            pretty_print(rec)
            break


#------------------------------------------------------------------------------------- 
if __name__ == "__main__":
    
    #---------------------------------------------------------------------------------------------
    ''' Run this *FIRST* to preprocess incoming *_training.json and *_validation.json files for each item type (bw, fw, math)
        - splits data by item_id and saves train.jsonl and valid.jsonl files in each item's own directory
        - adds min_score, max_score, score_norm fields to each record
        - adds mode, section fields for bw
        - adds dev, con, org fields for fw
    '''
    preprocess_datasets()
    
    #---------------------------------------------------------------------------------------------
    
    ''' Run this to test loading items from processed data directories
        - load items from each item directory
        - add split, item_type, trait fields to each record
        - add payload field with custom prompt template for each record
        - add pred field if predictions are available
    '''
    # test_load_items()
    