from datetime import datetime
import os,sys

# make sure to set the environment variable before importing torch
if 'LOCAL_RANK' not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"]="0" # non-DDP training (single GPU)
    
import torch
import torch.distributed as dist
    
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

def is_main():
    if 'LOCAL_RANK' in os.environ:
        return int(os.environ['LOCAL_RANK']) == 0
    # also return True if not using DDP, or only one GPU is available
    return True

def print_gpus():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
        print(f"Number of CUDA devices: {num_gpus}")
        for idx, name in enumerate(gpu_names):
            print(f"GPU {idx}: {name}")
    else:
        print("CUDA is not available.")
        
def init_distributed():
    if is_main():
        print_gpus()
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)  # Very important to set the device
        dist.init_process_group(
            backend='nccl',
            init_method='env://'
        )
        print(f"DDP training on GPU {local_rank}")
        print(f"Initialized process group - rank {dist.get_rank()}/{dist.get_world_size()}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        print("non-DDP training (single GPU)")
        
init_distributed()

def is_main_process():
    is_main_process = (dist.get_rank() == 0) if dist.is_initialized() else True
    return is_main_process

def get_num_processes():
    return dist.get_world_size() if dist.is_initialized() else 1

# decorator to run 'func' only on main (rank=0) GPU process
# - all other GPU processes will just return None
# - for DDP / multi-GPU training
# def main(func):
#     def wrapper(*args, **kwargs):
#         if is_main():
#             result = func(*args, **kwargs)
#             # Synchronize all the processes
#             if dist.is_initialized():
#                 # Specify the device in barrier call
#                 dist.barrier(device_ids=[torch.cuda.current_device()])
#             return result
#         else:
#             # If not rank 0, wait for rank 0 to finish
#             if dist.is_initialized():
#                 dist.barrier(device_ids=[torch.cuda.current_device()])
#             return None
#     return wrapper

def main(func=None, *, debug=False):
    def decorator(func):
        def wrapper(*args, **kwargs):
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            
            if debug:
                timestamp = datetime.now().strftime('%H:%M:%S.%f')
                print(f"\n[DEBUG {timestamp}] Rank {local_rank}: Entering wrapper")
            
            if is_main():
                if debug:
                    print(f"\n[DEBUG] Rank {local_rank}: Is main process")
                try:
                    if debug:
                        print(f"\n[DEBUG] Rank {local_rank}: About to execute function {func.__name__}")
                    result = func(*args, **kwargs)
                    if debug:
                        print(f"\n[DEBUG] Rank {local_rank}: Function {func.__name__} completed")
                    
                    if torch.distributed.is_initialized():
                        if debug:
                            print(f"\n[DEBUG] Rank {local_rank}: DDP initialized, about to hit barrier")
                        # torch.distributed.barrier()#device_ids=[torch.cuda.current_device()])
                        torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
                        if debug:
                            print(f"\n[DEBUG] Rank {local_rank}: Passed barrier")
                    elif debug:
                        print(f"\n[DEBUG] Rank {local_rank}: DDP not initialized")
                    
                    if debug:
                        print(f"\n[DEBUG] Rank {local_rank}: Returning result")
                    return result
                    
                except Exception as e:
                    if debug:
                        print(f"\n[DEBUG] Rank {local_rank}: Error occurred: {str(e)}")
                    raise
            else:
                if debug:
                    print(f"\n[DEBUG] Rank {local_rank}: Not main process")
                try:
                    if torch.distributed.is_initialized():
                        if debug:
                            print(f"\n[DEBUG] Rank {local_rank}: Non-main about to hit barrier")
                        # torch.distributed.barrier()#device_ids=[torch.cuda.current_device()])
                        torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
                        if debug:
                            print(f"\n[DEBUG] Rank {local_rank}: Non-main passed barrier")
                    elif debug:
                        print(f"\n[DEBUG] Rank {local_rank}: Non-main, DDP not initialized")
                    
                    if debug:
                        print(f"\n[DEBUG] Rank {local_rank}: Non-main returning None")
                    return None
                    
                except Exception as e:
                    if debug:
                        print(f"\n[DEBUG] Rank {local_rank}: Non-main error occurred: {str(e)}")
                    raise
                    
        return wrapper
    
    if func is None:
        return decorator
    return decorator(func)

# print to stdout only on main GPU process
@main
def printmain(s):
    print(s)
