import os, sys
import numpy as np
import torch
import numpy as np

import xgboost as xgb
import time
from sklearn.model_selection import KFold, cross_val_predict
from kappa import ikappa

from util import clear_cuda_tensors
from ddp_utils import is_main, is_main_process
from embed_utils import compute_hf_embeddings_ddp


def run_xgb_on_item(embedder, item, records, preds_file=None, K=5, step=None, random_state=42, **kwargs):
    
    best_params = {'learning_rate': 0.05, 'n_estimators': 200, 'max_depth': 3, 
                  'min_child_weight': 1, 'subsample': 0.3, 'colsample_bytree': 0.9}
    
    item_start = time.time()
    
    # Compute embeddings just for this item
    emb_data = embedder.compute_embeddings({item: records}, **kwargs)
    
    if is_main():
        # Run XGBoost for this item
        print(f"Time to compute embeddings for item {item}: {time.time()-item_start:.2f} seconds")
        xgb_start = time.time()
        
        # get embeddings
        X = emb_data[item]['x']
        y = emb_data[item]['y']
        idx = emb_data[item]['idx']
        
        # Clear memory
        del emb_data[item]['x']
        del emb_data[item]['y']
        del emb_data[item]['idx']
        del emb_data
        
        ##############################################
        ## save embeddings to file
        # emb_dir = '/home/azureuser/embed/tmp'
        # emb_dir = '/mnt/llm-train/embed/tmp'
        # # concat idx, y, x horizontally... X is a 2dim matrix, rest are vectors...
        # emb_table = np.column_stack((idx, y, X))
        # # sort table by first column
        # emb_table = emb_table[emb_table[:,0].argsort()]
        # # save as numpy npy file
        # fn = os.path.join(emb_dir, f"hf_emb.npy")
        # # fn = os.path.join(emb_dir, f"st_emb.npy")
        # np.save(fn, emb_table)
        # print(f"Embeddings saved to {fn}")
        
        # compute norm of X
        # X_norm = np.linalg.norm(X, axis=1)
        
        ##############################################
        
        # Run XGBoost
        xgb_mod = xgb.XGBRegressor(objective='reg:squarederror', 
                                    eval_metric='rmse', 
                                    verbosity=0, 
                                    **best_params)
        
        # use K-fold cross-validation
        if K>1:
            kfold = KFold(n_splits=K, shuffle=True, random_state=random_state)
            y_pred = cross_val_predict(xgb_mod, X, y, cv=kfold, n_jobs=-1)
            qwk = ikappa(y, y_pred)
            
        # use originl train/valid split
        else:
            train_idx, valid_idx = [], []
            # get train/valid indices
            for rec in records:
                if rec.split.startswith('train'): train_idx.append(rec.index)
                elif rec.split.startswith('valid'): valid_idx.append(rec.index)
            train_idx, valid_idx = np.array(train_idx), np.array(valid_idx)
            X_train = X[np.in1d(idx, train_idx)]
            y_train = y[np.in1d(idx, train_idx)]
            X_valid = X[np.in1d(idx, valid_idx)]
            y_valid = y[np.in1d(idx, valid_idx)]
            
            xgb_mod.fit(X_train, y_train)
            y_pred = xgb_mod.predict(X_valid)
            qwk = ikappa(y_valid, y_pred)
        
        print(f"Time to run XGBoost for item {item}: {time.time()-xgb_start:.2f} seconds")
        print(f"Total time for item {item}: {time.time()-item_start:.2f} seconds")
        # print(f"QWK = {qwk:.4f}")
        
        # save K-fold predictions for entire dataset
        if preds_file and K>1:
            preds_table = np.column_stack((idx, y, y_pred.round().astype(int)))
            # sort table by first column
            preds_table = preds_table[preds_table[:,0].argsort()]
            np.savetxt(preds_file, preds_table, delimiter=',', fmt='%d,%d,%d')
            print(f"Predictions saved to {preds_file}")

        # Clear memory
        del X, y, y_pred, xgb_mod, idx

    else:
        qwk = None

    if torch.distributed.is_initialized(): torch.distributed.barrier(device_ids=[torch.cuda.current_device()])  # Wait for rank 0 to finish

    # clear_cuda_tensors()
    
    if is_main():
        if step is not None:
            print(f"QWK_{item}\t{step}\t{qwk:.4f}")
        else:
            print(f"QWK_{item}\t{qwk:.4f}")

    return qwk


def run_xgb_on_items(embedder, data_by_item, **kwargs):
    qwks = []
    for i, (item, records) in enumerate(data_by_item.items()):
        if is_main():
            print(f"\nProcessing item: {item} ({i+1}/{len(data_by_item)})")
        
        qwk = run_xgb_on_item(embedder, item, records, **kwargs)
        if qwk is not None:
            qwks.append(qwk)
            
    # if torch.distributed.is_initialized(): torch.distributed.barrier()

    if is_main():
        return np.array(qwks).round(4)
    else:
        return None