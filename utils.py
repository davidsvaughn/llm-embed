import sys, os
import time
import json
import traceback
import gc
import itertools
import torch
from glob import glob
import shutil

class adict(dict):
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self

def to_adict(d):
    if isinstance(d, dict):
        return adict({k: to_adict(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [to_adict(v) for v in d]
    else:
        return d

def mkdirs(path):
    try:
        os.makedirs(path)
    except Exception as e:
        pass

def read_json(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
    return data

def remove_glob(pattern, file=True, dir=True, verbose=False):
    for f in glob(pattern):
        if file and os.path.isfile(f):
            if verbose:
                print(f"Removing file: {f}")
            os.remove(f)
        if dir and os.path.isdir(f):
            if verbose:
                print(f"Removing directory: {f}")
            shutil.rmtree(f)

def find_cuda_tensors():
    """Find all CUDA tensors currently in memory"""
    
    for obj in itertools.chain(gc.get_objects(), locals().values(), globals().values()):
        try:
            if torch.is_tensor(obj):
                if obj.is_cuda:
                    print(f"Found CUDA tensor of size {obj.size()} at {hex(id(obj))}")
            elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
                if obj.data.is_cuda:
                    print(f"Found CUDA tensor in object of size {obj.data.size()} at {hex(id(obj))}")
        except:
            pass

def clear_cuda_tensors(target_size=None): # (1, 8192, 32, 96)
    """Clear tensors of specific size from memory"""
    count = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                if target_size is None or obj.size() == target_size:
                    del obj
                    count += 1
        except: 
            pass
    
    torch.cuda.empty_cache()
    gc.collect()
    # printmain(f"Cleared {count} tensors")
    print(f"Cleared {count} tensors")

def read_jsonl(file_path):
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except Exception as e:
                print(f"Error reading line: {line}")
                print(traceback.format_exc())
                # throw exception
                raise e
    return to_adict(records)
    
def write_jsonl(file_path, records):
    with open(file_path, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')

# def save_gzip_data(dataset, file_path, timeout=5):
#         # Acquire a write lock using portalocker
#         with portalocker.Lock(file_path, 'w', timeout=timeout) as lockfile:
#             with gzip.open(file_path, 'wt', encoding='UTF-8') as zipfile:
#                 json_tricks.dump(dataset, zipfile)

# def load_gzip_data(file_path, timeout=30):
#     # Acquire a read lock using portalocker
#     with portalocker.Lock(file_path, 'r', timeout=timeout) as lockfile:
#         with gzip.open(file_path, 'rt', encoding='UTF-8') as zipfile:
#             return json_tricks.load(zipfile)

def verify(tokenized_batch):
    """
    Verifies all sequences in batch start with same token that doesn't repeat.
    Args:
        tokenized_batch: Batch dictionary with 'input_ids' for multiple sequences
    Returns:
        tuple: (bool, int or None) - (is_valid, start_token_id)
    """
    if not tokenized_batch['input_ids']:
        return None
        
    # Get first token of first sequence as reference
    start_token = tokenized_batch['input_ids'][0][0]
    
    for sequence in tokenized_batch['input_ids']:
        # Check if sequence starts with reference token
        if sequence[0] != start_token:
            return False, None
            
        # Check if token appears again in this sequence
        if sequence[1:].count(start_token) > 0:
            return None
    
    return start_token
           
def pretty_print_old(data, indent=4, depth=0):
   if isinstance(data, dict):
       for key, value in data.items():
           print(' ' * depth + str(key) + ':', end=' ')
           if isinstance(value, (dict, list)):
               print()
               pretty_print(value, indent, depth + indent)
           else:
               print(str(value))
   elif isinstance(data, list):
       for item in data:
           if isinstance(item, (dict, list)):
               pretty_print(item, indent, depth)
           else:
               print(' ' * depth + str(item))
               
def pretty_print(data, indent=4, depth=0):
   if isinstance(data, dict):
       print(' ' * depth + '{')
       for key, value in data.items():
           print(' ' * (depth + indent) + str(key) + ':', end=' ')
           if isinstance(value, (dict, list)):
               print()
               pretty_print(value, indent, depth + indent)
           else:
               print(str(value))
       print(' ' * depth + '}')
   elif isinstance(data, list):
       print(' ' * depth + '[')
       for item in data:
           if isinstance(item, (dict, list)):
               pretty_print(item, indent, depth + indent)
           else:
               print(' ' * (depth + indent) + str(item))
       print(' ' * depth + ']')

def tricky_traversal_order(arr, rev=True):
    if not arr: return []
    if len(arr) < 4: return arr
    # reverse array
    if rev: arr = arr[::-1]
    # pop the first and last elements and add them to the result
    result = [arr.pop(0)]
    if len(arr) > 0: result.append(arr.pop(-1))

    #loop through the array
    X = [arr]
    while X:
        x = X.pop(0)
        # get index of middle element
        i = len(x) // 2
        # add middle element to result
        result.append(x[i])
        # add left and right halves to X
        if x[:i]:
            X.append(x[:i])
        if x[i+1:]:
            X.append(x[i+1:])
    return result

def longest_repeated_char_substring(s):
    if not s:
        return ""
    
    max_length = 0
    max_char = s[0]
    current_length = 1
    current_char = s[0]
    
    for i in range(1, len(s)):
        if s[i] == current_char:
            current_length += 1
        else:
            if current_length > max_length:
                max_length = current_length
                max_char = current_char
            current_char = s[i]
            current_length = 1
    
    # Check one last time after the loop ends
    if current_length > max_length:
        max_length = current_length
        max_char = current_char
    
    return max_char * max_length

def fix_repeats(s, min_length=10):
    while True:
        sub = longest_repeated_char_substring(s)
        if len(sub) < min_length:
            break
        s = s.replace(sub, sub[0:2])
    return s